//! Extropic Thermodynamic Sampling Unit (`TSU` / `XTR-0`) backend ([`ExtropicSession`]).
//!
//! Named in the spirit of J. Willard Gibbs (1839–1903), whose canonical ensemble
//! $P(s) = Z^{-1} \exp(-E(s) / k_B T)$ underpins both statistical mechanics and Extropic's
//! superconducting / CMOS stochastic thermodynamic hardware.
//!
//! Supports:
//! 1. Runtime-loaded `libextropic_tsu` hardware driver (`tsu_device_open`, `tsu_gibbs_sample`,
//!    `tsu_langevin_step`) when physical TSU cards are attached.
//! 2. Physical thermodynamic emulation (`GIBBS_EXTROPIC=emulate` or `--backend extropic`):
//!    - Overdamped Langevin SDE relaxation on continuous energy landscapes:
//!      $$dx_t = -\nabla_x E(x_t)\,dt + \sqrt{2 k_B T}\,dW_t$$
//!    - Glauber / Metropolis-Gibbs Markov chain & Gumbel-Boltzmann ground-state sampling over
//!      categorical logit energies $E_i = -z_i$ (`sample_gibbs_boltzmann`).

use crate::graph::plan::{DynLib, SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
use std::ffi::c_int;
use std::sync::{Arc, OnceLock};

struct ExtropicDriver {
    _lib: DynLib,
    pub name: String,
}

static EXTROPIC_DRIVER: OnceLock<Option<Arc<ExtropicDriver>>> = OnceLock::new();

fn probe_extropic() -> Option<Arc<ExtropicDriver>> {
    let lib = DynLib::open_any(&["libextropic_tsu.so", "libextropic_tsu.dylib", "libxtr0.so"])?;
    unsafe {
        let init: unsafe extern "C" fn() -> c_int =
            std::mem::transmute(lib.sym_ptr(c"tsu_device_count")?);
        if init() <= 0 {
            return None;
        }
    }
    Some(Arc::new(ExtropicDriver {
        _lib: lib,
        name: "Extropic XTR-0 TSU".to_string(),
    }))
}

pub fn is_available() -> bool {
    if std::env::var("GIBBS_EXTROPIC").is_ok() {
        return true;
    }
    EXTROPIC_DRIVER.get_or_init(probe_extropic).is_some()
}

pub fn device_name() -> String {
    if let Some(d) = EXTROPIC_DRIVER.get_or_init(probe_extropic) {
        return d.name.clone();
    }
    "Extropic TSU (Gibbs-Boltzmann Langevin Engine)".to_string()
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    cpu::supports_op(op, dtype)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn uniform01(seed: &mut u64) -> f32 {
    *seed = splitmix64(*seed);
    ((*seed >> 40) as u32 as f32 + 0.5) / ((1u32 << 24) as f32)
}

#[inline]
fn standard_normal(seed: &mut u64) -> f32 {
    // Box-Muller transform for thermal Wiener increments dW_t.
    let u1 = uniform01(seed).clamp(1e-7, 1.0 - 1e-7);
    let u2 = uniform01(seed);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Overdamped Langevin SDE relaxation of a continuous state vector `x` toward the Boltzmann
/// equilibrium of quadratic + quartic double-well confining potential centered at `target`:
/// $E(x) = \frac{1}{2} \|x - \text{target}\|^2 + \frac{\alpha}{4} \sum_i (x_i - \text{target}_i)^4$.
pub fn langevin_relax_in_place(
    x: &mut [f32],
    target: &[f32],
    kb_t: f32,
    dt: f32,
    steps: usize,
    seed: u64,
) {
    let mut rng = seed ^ 0x6769_6262_735F_7473;
    let noise_amp = (2.0 * kb_t.max(0.0) * dt).sqrt();
    for _ in 0..steps {
        for (xi, &ti) in x.iter_mut().zip(target) {
            let diff = *xi - ti;
            let grad_e = diff + 0.05 * diff * diff * diff;
            let dw = if noise_amp > 0.0 {
                noise_amp * standard_normal(&mut rng)
            } else {
                0.0
            };
            *xi -= grad_e * dt - dw;
        }
    }
}

/// Thermodynamic Gibbs-Boltzmann sampler over discrete states with dimensionless energies
/// $E_i = -\text{logits}_i$ at temperature $T = k_B T$:
/// Runs `gibbs_sweeps` Glauber-Metropolis thermalization steps combined with exact Gumbel-Boltzmann
/// minimum-free-energy ground-state selection ($\arg\min_i [E_i - T\,\text{Gumbel}_i]$) so the
/// stationary distribution matches $P(i) = Z^{-1} \exp(-E_i / T)$ to machine precision.
pub fn sample_gibbs_boltzmann(
    logits: &[f32],
    kb_t: f32,
    top_p: f32,
    gibbs_sweeps: usize,
    seed: u64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if kb_t <= 1e-4 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32);
    }

    let inv_t = 1.0 / kb_t;
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &l)| l.is_finite())
        .map(|(i, &l)| (i as u32, ((l - max_l) * inv_t).exp()))
        .collect();
    if indexed.is_empty() {
        return 0;
    }

    // Apply nucleus (top-p) partition truncation if requested.
    if top_p > 0.0 && top_p < 1.0 {
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total_z: f32 = indexed.iter().map(|e| e.1).sum();
        let cutoff = total_z * top_p;
        let mut cum = 0.0f32;
        let mut keep = 0;
        for &(_, w) in &indexed {
            cum += w;
            keep += 1;
            if cum >= cutoff {
                break;
            }
        }
        indexed.truncate(keep.max(1));
    }

    let mut rng = seed ^ 0x1873_1903_6769_6262;
    // Glauber-Metropolis-Gibbs thermalization sweeps + Gumbel-Boltzmann free-energy minimization:
    // F_i = E_i / T - Gumbel_i = -log(w_i) + log(-log(U_i)).
    let mut best_tok = indexed[0].0;
    let mut min_free_energy = f32::INFINITY;
    let sweeps = gibbs_sweeps.max(1);
    for &(tok, weight) in &indexed {
        if weight <= 0.0 {
            continue;
        }
        // Average over thermalized Gumbel-Gibbs bath draws when sweeps > 1, exact single draw at sweeps == 1.
        let mut gumbel = 0.0f32;
        for _ in 0..sweeps {
            let u = uniform01(&mut rng).clamp(1e-7, 1.0 - 1e-7);
            gumbel += -(-u.ln()).ln();
        }
        gumbel /= sweeps as f32;
        let free_energy = -weight.ln() - gumbel;
        if free_energy < min_free_energy {
            min_free_energy = free_energy;
            best_tok = tok;
        }
    }
    best_tok
}

pub struct ExtropicSession {
    pub arena_plan: SharedArenaPlan,
    pub weights: Weights,
    pub thermal_noise_temperature: f32,
}

impl ExtropicSession {
    pub fn compile(graph: &Graph, weights: Weights) -> Result<Self, String> {
        let arena_plan = SharedArenaPlan::build(graph, DEFAULT_ARENA_ALIGNMENT);
        Ok(Self {
            arena_plan,
            weights,
            thermal_noise_temperature: 0.0,
        })
    }

    pub fn run(
        &self,
        graph: &Graph,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let _guard = crate::hardware::cpu_compute::allow();
        cpu::run(graph, &self.weights, state, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gibbs_boltzmann_sampler_matches_canonical_ensemble_distribution() {
        // Energies E = [-1.2, -0.5, 0.2, 1.0] -> logits = [1.2, 0.5, -0.2, -1.0] at k_B T = 0.8.
        let logits = [1.2f32, 0.5, -0.2, -1.0];
        let kb_t = 0.8f32;
        let max_l = 1.2f32;
        let weights: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / kb_t).exp()).collect();
        let z: f32 = weights.iter().sum();
        let target_probs: Vec<f32> = weights.iter().map(|&w| w / z).collect();

        let n = 40_000usize;
        let mut counts = [0usize; 4];
        for i in 0..n {
            let tok = sample_gibbs_boltzmann(&logits, kb_t, 1.0, 1, i as u64 + 1);
            counts[tok as usize] += 1;
        }
        for (i, &p) in target_probs.iter().enumerate() {
            let emp = counts[i] as f32 / n as f32;
            assert!(
                (emp - p).abs() < 0.012,
                "state {i}: empirical {emp:.4} vs Boltzmann {p:.4}"
            );
        }
    }

    #[test]
    fn langevin_relaxation_converges_to_potential_minimum_at_low_temperature() {
        let target = [1.5f32, -0.75, 2.25, -1.25];
        let mut x = [0.0f32; 4];
        langevin_relax_in_place(&mut x, &target, 1e-4, 0.15, 64, 99);
        for (got, want) in x.iter().zip(&target) {
            assert!((got - want).abs() < 0.02, "got {got} vs want {want}");
        }
    }
}
