//! diffusers `FlowMatchEulerDiscreteScheduler` (num_train_timesteps 1000, no karras/exponential/beta
//! sigmas, no terminal stretch): rectified-flow sigma schedules and the Euler step. The model's
//! timestep input is `sigma * 1000`.

const TRAIN_STEPS: f64 = 1000.0;

/// Static shift `s * sigma / (1 + (s - 1) * sigma)` (`use_dynamic_shifting: false`).
pub fn shift_sigma(sigma: f64, shift: f64) -> f64 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// Exponential time shift with exponent 1: `e^mu / (e^mu + (1/t - 1))` (`use_dynamic_shifting: true`).
pub fn time_shift(mu: f64, t: f64) -> f64 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0))
}

/// Resolution-dependent `mu`: linear in the image token count through (base_seq, base_shift) and
/// (max_seq, max_shift); diffusers defaults are 256, 4096, 0.5, 1.15.
pub fn calculate_shift(
    seq_len: usize,
    base_seq: f64,
    max_seq: f64,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq - base_seq);
    seq_len as f64 * m + base_shift - m * base_seq
}

/// `np.linspace(1, 1/steps, steps)`: the custom sigmas FLUX and Z-Image pass to `set_timesteps`.
pub fn linspace_sigmas(steps: usize) -> Vec<f64> {
    let end = 1.0 / steps as f64;
    (0..steps)
        .map(|i| {
            if steps == 1 {
                1.0
            } else {
                1.0 + (end - 1.0) * i as f64 / (steps - 1) as f64
            }
        })
        .collect()
}

/// `set_timesteps(steps)` without custom sigmas and with a static shift (SD3): `linspace(sigma_max,
/// sigma_min)` of the scheduler's own (already shifted) training sigmas, shifted again, then 0 appended.
pub fn default_sigmas(steps: usize, shift: f64) -> Vec<f32> {
    let (hi, lo) = (
        shift_sigma(1.0, shift),
        shift_sigma(1.0 / TRAIN_STEPS, shift),
    );
    let base: Vec<f64> = (0..steps)
        .map(|i| {
            if steps == 1 {
                hi
            } else {
                hi + (lo - hi) * i as f64 / (steps - 1) as f64
            }
        })
        .collect();
    finish(base.into_iter().map(|s| shift_sigma(s, shift)))
}

/// `set_timesteps(sigmas = base)` with a static shift (Z-Image), 0 appended.
pub fn shifted_sigmas(base: &[f64], shift: f64) -> Vec<f32> {
    finish(base.iter().map(|&s| shift_sigma(s, shift)))
}

/// `set_timesteps(sigmas = base, mu)` with dynamic shifting (FLUX), 0 appended.
pub fn dynamic_sigmas(base: &[f64], mu: f64) -> Vec<f32> {
    finish(base.iter().map(|&s| time_shift(mu, s)))
}

/// `stretch_shift_to_terminal` (`shift_terminal`): rescales `1 - sigma` so the last sigma is `terminal`.
pub fn stretch_to_terminal(sigmas: &mut [f64], terminal: f64) {
    let Some(&last) = sigmas.last() else { return };
    let scale = (1.0 - last) / (1.0 - terminal);
    sigmas
        .iter_mut()
        .for_each(|s| *s = 1.0 - (1.0 - *s) / scale);
}

fn finish(s: impl Iterator<Item = f64>) -> Vec<f32> {
    s.map(|v| v as f32).chain([0.0]).collect()
}

/// Model timestep for `sigma`.
pub fn timestep(sigma: f32) -> f32 {
    sigma * TRAIN_STEPS as f32
}

/// Euler step of the flow ODE: `x += (sigma_next - sigma) * v`.
pub fn euler_step(x: &mut [f32], v: &[f32], sigma: f32, sigma_next: f32) {
    let dt = sigma_next - sigma;
    for (x, v) in x.iter_mut().zip(v) {
        *x += dt * v;
    }
}

/// `DPMSolverMultistepScheduler(use_flow_sigmas, flow_shift)` sigmas: `1 - linspace(1, 1/1000, N+1)`
/// shifted, descending, the last (0) dropped, then the final sigma 0 appended (`final_sigmas_type: zero`).
/// The model timestep is `floor(sigma * 1000)` (the scheduler stores int64 timesteps).
pub fn dpm_flow_sigmas(steps: usize, shift: f64) -> Vec<f32> {
    let n = steps + 1;
    let alphas = (0..n).map(|i| 1.0 + (1.0 / TRAIN_STEPS - 1.0) * i as f64 / (n - 1) as f64);
    let mut s: Vec<f64> = alphas.map(|a| shift_sigma(1.0 - a, shift)).collect();
    s.reverse();
    s.pop();
    finish(s.into_iter())
}

/// DPM-Solver++ (2M, midpoint) on flow predictions: x0 = x - sigma * v, alpha = 1 - sigma.
/// First order on the first step and on the final step to sigma 0.
#[derive(Default)]
pub struct DpmSolver2 {
    prev_x0: Option<Vec<f32>>,
}

impl DpmSolver2 {
    /// Steps `x` from `sigmas[i]` to `sigmas[i + 1]` given the model's flow prediction `v` at `sigmas[i]`.
    pub fn step(&mut self, x: &mut [f32], v: &[f32], sigmas: &[f32], i: usize) {
        let (s0, t) = (sigmas[i] as f64, sigmas[i + 1] as f64);
        let x0: Vec<f32> = x.iter().zip(v).map(|(x, v)| x - s0 as f32 * v).collect();
        let lambda = |s: f64| (1.0 - s).ln() - s.ln();
        if t == 0.0 {
            x.copy_from_slice(&x0);
        } else {
            let h = lambda(t) - lambda(s0);
            let (a, b) = ((t / s0) as f32, ((1.0 - t) * ((-h).exp() - 1.0)) as f32);
            match &self.prev_x0 {
                Some(m1) => {
                    let r0 = (lambda(s0) - lambda(sigmas[i - 1] as f64)) / h;
                    let k = (0.5 / r0) as f32;
                    for ((x, &d0), &m1) in x.iter_mut().zip(&x0).zip(m1) {
                        *x = a * *x - b * d0 - b * k * (d0 - m1);
                    }
                }
                None => x
                    .iter_mut()
                    .zip(&x0)
                    .for_each(|(x, &d0)| *x = a * *x - b * d0),
            }
        }
        self.prev_x0 = Some(x0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f64) {
        assert!((a as f64 - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn sd3_default_schedule_double_shifts_sigma_min() {
        // sigma_min = shift(0.001) = 0.002994012; shifted again = 0.008928571.
        let s = default_sigmas(2, 3.0);
        assert_eq!(s.len(), 3);
        near(s[0], 1.0);
        near(s[1], 3.0 * 0.002994012 / (1.0 + 2.0 * 0.002994012));
        assert_eq!(s[2], 0.0);
        // 28 steps: strictly decreasing from 1 to the final 0.
        let s = default_sigmas(28, 3.0);
        assert_eq!(s.len(), 29);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        near(timestep(s[0]), 1000.0);
    }

    #[test]
    fn z_image_schedule_is_shifted_linspace() {
        let base = linspace_sigmas(8);
        assert_eq!(base.len(), 8);
        assert!((base[7] - 0.125).abs() < 1e-12);
        let s = shifted_sigmas(&base, 3.0);
        near(s[0], 1.0);
        near(s[7], 3.0 * 0.125 / (1.0 + 2.0 * 0.125)); // 0.3
        assert_eq!(s[8], 0.0);
    }

    #[test]
    fn flux_mu_and_time_shift() {
        assert!((calculate_shift(256, 256.0, 4096.0, 0.5, 1.15) - 0.5).abs() < 1e-12);
        assert!((calculate_shift(4096, 256.0, 4096.0, 0.5, 1.15) - 1.15).abs() < 1e-12);
        let mu = 1.15f64;
        let s = dynamic_sigmas(&[1.0, 0.5], mu);
        near(s[0], 1.0);
        near(s[1], mu.exp() / (mu.exp() + 1.0));
    }

    #[test]
    fn euler_step_reaches_data_from_noise_with_exact_velocity() {
        // x_sigma = (1 - sigma) x0 + sigma n; v = n - x0. Stepping sigma 1 -> 0 lands on x0.
        let (x0, n) = ([0.5f32, -2.0], [1.0f32, 0.25]);
        let v: Vec<f32> = n.iter().zip(&x0).map(|(n, x)| n - x).collect();
        let mut x = n.to_vec();
        for w in shifted_sigmas(&linspace_sigmas(4), 3.0).windows(2) {
            euler_step(&mut x, &v, w[0], w[1]);
        }
        assert!(
            x.iter().zip(&x0).all(|(a, b)| (a - b).abs() < 1e-6),
            "{x:?}"
        );
    }

    #[test]
    fn sana_dpm_sigmas() {
        let s = dpm_flow_sigmas(2, 3.0);
        assert_eq!(s.len(), 3);
        near(s[0], 3.0 * 0.999 / (1.0 + 2.0 * 0.999));
        near(s[1], 3.0 * 0.4995 / (1.0 + 2.0 * 0.4995));
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn dpm_solver_stays_on_the_straight_path() {
        // Exact flow prediction on x_sigma = (1 - sigma) x0 + sigma n: every DPM-Solver++ step (first
        // and second order) lands on the path, and the last one on x0.
        let (x0, n) = ([0.5f32, -2.0], [1.0f32, 0.25]);
        let v: Vec<f32> = n.iter().zip(&x0).map(|(n, x)| n - x).collect();
        let sigmas = dpm_flow_sigmas(6, 3.0);
        let mut x: Vec<f32> = x0
            .iter()
            .zip(&n)
            .map(|(a, b)| (1.0 - sigmas[0]) * a + sigmas[0] * b)
            .collect();
        let mut dpm = DpmSolver2::default();
        for i in 0..6 {
            dpm.step(&mut x, &v, &sigmas, i);
            let s = sigmas[i + 1];
            for k in 0..2 {
                let want = (1.0 - s) * x0[k] + s * n[k];
                assert!((x[k] - want).abs() < 1e-5, "step {i}: {} vs {want}", x[k]);
            }
        }
    }
}
