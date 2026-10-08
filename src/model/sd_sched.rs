//! diffusers noise schedulers for the Stable Diffusion pipelines, configured from a repo's
//! `scheduler/scheduler_config.json`: Euler, Euler ancestral, PNDM (PLMS, `skip_prk_steps`), DDIM
//! (eta 0), LCM and DPM-Solver++ (multistep, order <= 2, optional Karras sigmas). Formulas follow
//! diffusers' `scheduling_*.py`; every option outside these is an Err naming it.

use crate::model::sd::Rng;
use serde_json::Value;

/// What the model predicts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prediction {
    Epsilon,
    V,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Euler,
    EulerAncestral,
    Pndm,
    Ddim,
    Lcm,
    DpmPP,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Spacing {
    Leading,
    Trailing,
    Linspace,
}

/// A parsed `scheduler_config.json`.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub kind: Kind,
    pub class: String,
    pub prediction: Prediction,
    spacing: Spacing,
    steps_offset: usize,
    set_alpha_to_one: bool,
    karras: bool,
    solver_order: usize,
    original_steps: usize,
    timestep_scaling: f64,
    alphas_cumprod: Vec<f64>,
}

impl SchedulerConfig {
    pub fn from_json(cfg: &Value) -> Result<Self, String> {
        let class = cfg["_class_name"].as_str().unwrap_or("").to_string();
        let kind = match class.as_str() {
            "EulerDiscreteScheduler" => Kind::Euler,
            "EulerAncestralDiscreteScheduler" => Kind::EulerAncestral,
            "PNDMScheduler" => Kind::Pndm,
            "DDIMScheduler" => Kind::Ddim,
            "LCMScheduler" => Kind::Lcm,
            "DPMSolverMultistepScheduler" => Kind::DpmPP,
            c => {
                return Err(format!(
                "scheduler '{c}' is not implemented (Euler, EulerAncestral, PNDM, DDIM, LCM, DPMSolverMultistep are)"
            ))
            }
        };
        let s = |k: &str| cfg[k].as_str();
        let b = |k: &str, d: bool| cfg[k].as_bool().unwrap_or(d);
        let f = |k: &str, d: f64| cfg[k].as_f64().unwrap_or(d);
        let u = |k: &str, d: usize| cfg[k].as_u64().map_or(d, |v| v as usize);
        let prediction = match s("prediction_type").unwrap_or("epsilon") {
            "epsilon" => Prediction::Epsilon,
            "v_prediction" => Prediction::V,
            p => return Err(format!("{class}: prediction_type '{p}' is not supported")),
        };
        let spacing = match s("timestep_spacing").unwrap_or(
            if matches!(kind, Kind::Pndm | Kind::Lcm | Kind::Ddim) {
                "leading"
            } else {
                "linspace"
            },
        ) {
            "leading" => Spacing::Leading,
            "trailing" => Spacing::Trailing,
            "linspace" => Spacing::Linspace,
            p => return Err(format!("{class}: timestep_spacing '{p}' is not supported")),
        };
        for (key, bad) in [
            ("rescale_betas_zero_snr", b("rescale_betas_zero_snr", false)),
            ("thresholding", b("thresholding", false)),
            ("use_exponential_sigmas", b("use_exponential_sigmas", false)),
            ("use_beta_sigmas", b("use_beta_sigmas", false)),
            ("use_lu_lambdas", b("use_lu_lambdas", false)),
            ("trained_betas", !cfg["trained_betas"].is_null()),
            (
                "skip_prk_steps=false",
                kind == Kind::Pndm && !b("skip_prk_steps", false),
            ),
            (
                "clip_sample",
                kind != Kind::Euler
                    && kind != Kind::EulerAncestral
                    && b("clip_sample", kind == Kind::Ddim),
            ),
        ] {
            if bad {
                return Err(format!("{class}: {key} is not supported"));
            }
        }
        if kind == Kind::Euler && s("interpolation_type").is_some_and(|t| t != "linear") {
            return Err(format!(
                "{class}: interpolation_type {:?} is not supported",
                s("interpolation_type")
            ));
        }
        if kind == Kind::Euler && s("timestep_type").is_some_and(|t| t != "discrete") {
            return Err(format!(
                "{class}: timestep_type {:?} is not supported",
                s("timestep_type")
            ));
        }
        let karras = b("use_karras_sigmas", false);
        if karras && !matches!(kind, Kind::Euler | Kind::DpmPP) {
            return Err(format!("{class}: use_karras_sigmas is not supported"));
        }
        if kind == Kind::DpmPP {
            let algo = s("algorithm_type").unwrap_or("dpmsolver++");
            let solver = s("solver_type").unwrap_or("midpoint");
            if algo != "dpmsolver++"
                || solver != "midpoint"
                || u("solver_order", 2) > 2
                || b("euler_at_final", false)
            {
                return Err(format!(
                    "{class}: only algorithm_type dpmsolver++, solver_type midpoint, solver_order <= 2 are supported"
                ));
            }
            if s("final_sigmas_type").is_some_and(|t| t != "zero")
                || f("lambda_min_clipped", f64::NEG_INFINITY).is_finite()
            {
                return Err(format!(
                    "{class}: final_sigmas_type / lambda_min_clipped variants are not supported"
                ));
            }
            if b("use_flow_sigmas", false) || !cfg["variance_type"].is_null() {
                return Err(format!(
                    "{class}: flow sigmas / learned variance are not supported"
                ));
            }
        }
        let n = u("num_train_timesteps", 1000);
        let (b0, b1) = (f("beta_start", 0.0001), f("beta_end", 0.02));
        let betas: Vec<f64> = match s("beta_schedule").unwrap_or("linear") {
            "scaled_linear" => {
                let (a, z) = (b0.sqrt(), b1.sqrt());
                (0..n)
                    .map(|i| (a + (z - a) * i as f64 / (n - 1) as f64).powi(2))
                    .collect()
            }
            "linear" => (0..n)
                .map(|i| b0 + (b1 - b0) * i as f64 / (n - 1) as f64)
                .collect(),
            other => return Err(format!("{class}: beta_schedule '{other}' is not supported")),
        };
        let mut acc = 1.0;
        let alphas_cumprod = betas
            .iter()
            .map(|beta| {
                acc *= 1.0 - beta;
                acc
            })
            .collect();
        Ok(Self {
            kind,
            class,
            prediction,
            spacing,
            steps_offset: u("steps_offset", 0),
            set_alpha_to_one: b("set_alpha_to_one", kind == Kind::Lcm || kind == Kind::Ddim),
            karras,
            solver_order: u("solver_order", 2),
            original_steps: u("original_inference_steps", 50),
            timestep_scaling: f("timestep_scaling", 10.0),
            alphas_cumprod,
        })
    }

    fn train_steps(&self) -> usize {
        self.alphas_cumprod.len()
    }

    /// Runs `steps` inference steps.
    pub fn build(&self, steps: usize) -> Result<Scheduler, String> {
        if steps == 0 || steps > self.train_steps() {
            return Err(format!(
                "steps must be in 1..={}, got {steps}",
                self.train_steps()
            ));
        }
        let ac = &self.alphas_cumprod;
        let n_train = self.train_steps();
        let sigma_of = |t: f64| interp(t, ac);
        let (timesteps, sigmas): (Vec<f64>, Vec<f64>) = match self.kind {
            Kind::Euler | Kind::EulerAncestral => {
                let mut ts = self.spaced(steps, false);
                let mut sig: Vec<f64> = ts.iter().map(|&t| sigma_of(t)).collect();
                if self.karras {
                    sig = karras(sig[sig.len() - 1], sig[0], steps);
                    ts = sig.iter().map(|&s| sigma_to_t(s, ac)).collect();
                }
                sig.push(0.0);
                (ts, sig)
            }
            Kind::Pndm => {
                let mut t = self.spaced(steps, true);
                t.reverse(); // ascending _timesteps
                let n = t.len();
                let mut plms: Vec<f64> = t[..n - 1].to_vec();
                if n >= 2 {
                    plms.push(t[n - 2]);
                }
                plms.push(t[n - 1]);
                plms.reverse();
                (plms, Vec::new())
            }
            Kind::Ddim => (self.spaced(steps, true), Vec::new()),
            Kind::Lcm => {
                let k = n_train / self.original_steps;
                let origin: Vec<f64> = (1..=self.original_steps)
                    .rev()
                    .map(|i| (i * k - 1) as f64)
                    .collect();
                if steps > self.original_steps {
                    return Err(format!(
                        "LCM steps must be <= original_inference_steps {}",
                        self.original_steps
                    ));
                }
                let ts = (0..steps)
                    .map(|i| origin[(i * origin.len()) / steps])
                    .collect();
                (ts, Vec::new())
            }
            Kind::DpmPP => {
                let last = n_train;
                let mut ts: Vec<f64> = match self.spacing {
                    Spacing::Linspace => {
                        let v = linspace(0.0, (last - 1) as f64, steps + 1);
                        v.iter().rev().take(steps).map(|x| x.round()).collect()
                    }
                    Spacing::Leading => {
                        let r = last / (steps + 1);
                        (0..=steps)
                            .rev()
                            .take(steps)
                            .map(|i| (i * r + self.steps_offset) as f64)
                            .collect()
                    }
                    Spacing::Trailing => {
                        let r = n_train as f64 / steps as f64;
                        arange_desc(last as f64, r)
                            .iter()
                            .map(|x| x.round() - 1.0)
                            .collect()
                    }
                };
                let mut sig: Vec<f64>;
                if self.karras {
                    let all: Vec<f64> = (0..n_train).map(|t| sigma_of(t as f64)).collect();
                    sig = karras(all[0], all[n_train - 1], steps);
                    ts = sig.iter().map(|&s| sigma_to_t(s, ac).round()).collect();
                } else {
                    sig = ts.iter().map(|&t| sigma_of(t)).collect();
                }
                sig.push(0.0);
                (ts, sig)
            }
        };
        Ok(Scheduler {
            cfg: self.clone(),
            steps,
            timesteps,
            sigmas,
            ets: Vec::new(),
            counter: 0,
            cur_sample: None,
            prev_x0: None,
            lower_order_nums: 0,
        })
    }

    /// Leading / trailing / linspace timesteps, descending; integer schedulers (PNDM, DDIM) round linspace.
    fn spaced(&self, n: usize, integer: bool) -> Vec<f64> {
        let t = self.train_steps();
        match self.spacing {
            Spacing::Leading => {
                let r = t / n;
                (0..n)
                    .rev()
                    .map(|i| (i * r + self.steps_offset) as f64)
                    .collect()
            }
            Spacing::Trailing => arange_desc(t as f64, t as f64 / n as f64)
                .iter()
                .map(|x| x.round() - 1.0)
                .collect(),
            Spacing::Linspace => {
                let v = linspace(0.0, (t - 1) as f64, n);
                v.iter()
                    .rev()
                    .map(|&x| if integer { x.round() } else { x })
                    .collect()
            }
        }
    }
}

/// `np.arange(start, 0, -step)`.
fn arange_desc(start: f64, step: f64) -> Vec<f64> {
    let n = (start / step).ceil() as usize;
    (0..n).map(|i| start - i as f64 * step).collect()
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

/// `np.interp(t, arange, sigmas)` with sigmas = sqrt((1 - ac) / ac).
fn interp(t: f64, ac: &[f64]) -> f64 {
    let s = |i: usize| ((1.0 - ac[i]) / ac[i]).sqrt();
    let t = t.clamp(0.0, (ac.len() - 1) as f64);
    let (i, f) = (t.floor() as usize, t - t.floor());
    if i + 1 >= ac.len() {
        return s(ac.len() - 1);
    }
    s(i) * (1.0 - f) + s(i + 1) * f
}

/// Karras et al. (2022) sigmas, rho 7, from `max` down to `min`.
fn karras(min: f64, max: f64, n: usize) -> Vec<f64> {
    let (lo, hi) = (min.powf(1.0 / 7.0), max.powf(1.0 / 7.0));
    linspace(0.0, 1.0, n)
        .iter()
        .map(|r| (hi + r * (lo - hi)).powi(7))
        .collect()
}

/// diffusers `_sigma_to_t`: log-sigma interpolation onto the training timesteps.
fn sigma_to_t(sigma: f64, ac: &[f64]) -> f64 {
    let logs: Vec<f64> = ac.iter().map(|a| ((1.0 - a) / a).sqrt().ln()).collect();
    let ls = sigma.max(1e-10).ln();
    let low = logs
        .iter()
        .filter(|&&l| ls - l >= 0.0)
        .count()
        .saturating_sub(1)
        .min(logs.len() - 2);
    let (lo, hi) = (logs[low], logs[low + 1]);
    let w = ((lo - ls) / (lo - hi)).clamp(0.0, 1.0);
    (1.0 - w) * low as f64 + w * (low + 1) as f64
}

/// One configured denoising run; [`Scheduler::step`] advances it.
pub struct Scheduler {
    cfg: SchedulerConfig,
    steps: usize,
    /// UNet timesteps in order (PNDM: `steps + 1` of them).
    pub timesteps: Vec<f64>,
    /// Euler / DPM: per-step sigmas with the trailing 0.
    sigmas: Vec<f64>,
    ets: Vec<Vec<f32>>,
    counter: usize,
    cur_sample: Option<Vec<f32>>,
    prev_x0: Option<Vec<f32>>,
    lower_order_nums: usize,
}

impl Scheduler {
    fn ac(&self, t: f64) -> f64 {
        self.cfg.alphas_cumprod[t as usize]
    }

    fn final_ac(&self) -> f64 {
        if self.cfg.set_alpha_to_one {
            1.0
        } else {
            self.cfg.alphas_cumprod[0]
        }
    }

    fn ve(&self) -> bool {
        matches!(self.cfg.kind, Kind::Euler | Kind::EulerAncestral)
    }

    /// Index of the first step an img2img run of `strength` denoises from (diffusers `get_timesteps`).
    pub fn start_index(&self, strength: f32) -> Result<usize, String> {
        let init = ((self.steps as f32 * strength.max(0.0)) as usize).min(self.steps);
        if init == 0 {
            return Err(format!(
                "strength {strength} with {} steps yields zero denoising steps",
                self.steps
            ));
        }
        Ok(self.steps - init)
    }

    /// Std of the initial latent noise.
    pub fn init_sigma(&self) -> f64 {
        if !self.ve() {
            return 1.0;
        }
        let max = self.sigmas.iter().cloned().fold(0.0, f64::max);
        match self.cfg.spacing {
            Spacing::Trailing | Spacing::Linspace => max,
            Spacing::Leading => (max * max + 1.0).sqrt(),
        }
    }

    /// Factor on the latents before the UNet at step `i` (`scale_model_input`).
    pub fn input_scale(&self, i: usize) -> f32 {
        if self.ve() {
            (1.0 / (self.sigmas[i] * self.sigmas[i] + 1.0).sqrt()) as f32
        } else {
            1.0
        }
    }

    /// img2img start: `x0` noised to step `i` (`add_noise`).
    pub(crate) fn add_noise(&self, x0: &mut [f32], i: usize, rng: &mut Rng) {
        if self.ve() {
            let s = self.sigmas[i] as f32;
            x0.iter_mut().for_each(|v| *v += rng.gaussian() * s);
        } else {
            let a = self.ac(self.timesteps[i]);
            let (sa, sb) = (a.sqrt() as f32, (1.0 - a).sqrt() as f32);
            x0.iter_mut()
                .for_each(|v| *v = sa * *v + sb * rng.gaussian());
        }
    }

    /// Model output -> (eps, x0) at VP alpha_cumprod `a` for sample `x`.
    fn eps_x0(&self, out: &[f32], x: &[f32], a: f64) -> (Vec<f32>, Vec<f32>) {
        let (sa, sb) = (a.sqrt() as f32, (1.0 - a).sqrt() as f32);
        match self.cfg.prediction {
            Prediction::Epsilon => (
                out.to_vec(),
                x.iter().zip(out).map(|(x, e)| (x - sb * e) / sa).collect(),
            ),
            Prediction::V => (
                x.iter().zip(out).map(|(x, v)| sa * v + sb * x).collect(),
                x.iter().zip(out).map(|(x, v)| sa * x - sb * v).collect(),
            ),
        }
    }

    /// Step `i` (index into `timesteps`): `x` <- previous sample.
    pub(crate) fn step(&mut self, i: usize, out: &[f32], x: &mut Vec<f32>, rng: &mut Rng) {
        match self.cfg.kind {
            Kind::Euler | Kind::EulerAncestral => {
                let (s, s_next) = (self.sigmas[i], self.sigmas[i + 1]);
                let x0: Vec<f32> = match self.cfg.prediction {
                    Prediction::Epsilon => {
                        x.iter().zip(out).map(|(x, e)| x - s as f32 * e).collect()
                    }
                    Prediction::V => {
                        let (c_out, c_skip) = (
                            (-s / (s * s + 1.0).sqrt()) as f32,
                            (1.0 / (s * s + 1.0)) as f32,
                        );
                        x.iter()
                            .zip(out)
                            .map(|(x, v)| v * c_out + x * c_skip)
                            .collect()
                    }
                };
                let (down, up) = if self.cfg.kind == Kind::Euler {
                    (s_next, 0.0)
                } else {
                    let up = (s_next * s_next * (s * s - s_next * s_next) / (s * s)).sqrt();
                    ((s_next * s_next - up * up).sqrt(), up)
                };
                let dt = ((down - s) / s) as f32;
                for (v, p) in x.iter_mut().zip(&x0) {
                    *v += (*v - p) * dt;
                }
                if up > 0.0 {
                    x.iter_mut().for_each(|v| *v += rng.gaussian() * up as f32);
                }
            }
            Kind::Pndm => {
                let ratio = (self.cfg.train_steps() / self.steps) as f64;
                let mut t = self.timesteps[i];
                let mut prev = t - ratio;
                if self.counter != 1 {
                    if self.ets.len() > 3 {
                        self.ets.remove(0);
                    }
                    self.ets.push(out.to_vec());
                } else {
                    prev = t;
                    t += ratio;
                }
                let e = &self.ets;
                let n = e.len();
                let comb = |w: &[f32]| -> Vec<f32> {
                    (0..out.len())
                        .map(|k| w.iter().enumerate().map(|(j, c)| c * e[n - 1 - j][k]).sum())
                        .collect()
                };
                let mut sample = x.clone();
                let m: Vec<f32> = if n == 1 && self.counter == 0 {
                    self.cur_sample = Some(x.clone());
                    out.to_vec()
                } else if n == 1 && self.counter == 1 {
                    sample = self.cur_sample.take().unwrap_or_else(|| x.clone());
                    out.iter().zip(&e[0]).map(|(a, b)| (a + b) / 2.0).collect()
                } else if n == 2 {
                    comb(&[1.5, -0.5])
                } else if n == 3 {
                    comb(&[23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0])
                } else {
                    comb(&[55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0])
                };
                let a = self.ac(t);
                let ap = if prev >= 0.0 {
                    self.ac(prev)
                } else {
                    self.final_ac()
                };
                let eps: Vec<f32> = match self.cfg.prediction {
                    Prediction::Epsilon => m,
                    Prediction::V => m
                        .iter()
                        .zip(&sample)
                        .map(|(v, s)| (a.sqrt() as f32) * v + ((1.0 - a).sqrt() as f32) * s)
                        .collect(),
                };
                let coeff = (ap / a).sqrt() as f32;
                let denom = a * (1.0 - ap).sqrt() + (a * (1.0 - a) * ap).sqrt();
                let k = ((ap - a) / denom) as f32;
                *x = sample
                    .iter()
                    .zip(&eps)
                    .map(|(s, e)| coeff * s - k * e)
                    .collect();
                self.counter += 1;
            }
            Kind::Ddim => {
                let t = self.timesteps[i];
                let prev = t - (self.cfg.train_steps() / self.steps) as f64;
                let a = self.ac(t);
                let ap = if prev >= 0.0 {
                    self.ac(prev)
                } else {
                    self.final_ac()
                };
                let (eps, x0) = self.eps_x0(out, x, a);
                let (sa, sb) = (ap.sqrt() as f32, (1.0 - ap).sqrt() as f32);
                *x = x0.iter().zip(&eps).map(|(p, e)| sa * p + sb * e).collect();
            }
            Kind::Lcm => {
                let t = self.timesteps[i];
                let prev = self.timesteps.get(i + 1).copied().unwrap_or(t);
                let a = self.ac(t);
                let ap = if prev >= 0.0 {
                    self.ac(prev)
                } else {
                    self.final_ac()
                };
                let st = t * self.cfg.timestep_scaling;
                let c_skip = (0.25 / (st * st + 0.25)) as f32;
                let c_out = (st / (st * st + 0.25).sqrt()) as f32;
                let (_, x0) = self.eps_x0(out, x, a);
                let den: Vec<f32> = x0
                    .iter()
                    .zip(x.iter())
                    .map(|(p, s)| c_out * p + c_skip * s)
                    .collect();
                *x = if i + 1 < self.timesteps.len() {
                    let (sa, sb) = (ap.sqrt() as f32, (1.0 - ap).sqrt() as f32);
                    den.iter().map(|d| sa * d + sb * rng.gaussian()).collect()
                } else {
                    den
                };
            }
            Kind::DpmPP => {
                let vp = |s: f64| {
                    let a = 1.0 / (s * s + 1.0).sqrt();
                    (a, s * a)
                };
                let (a_s0, s_s0) = vp(self.sigmas[i]);
                let (a_t, s_t) = vp(self.sigmas[i + 1]);
                let x0: Vec<f32> = match self.cfg.prediction {
                    Prediction::Epsilon => x
                        .iter()
                        .zip(out)
                        .map(|(x, e)| (x - s_s0 as f32 * e) / a_s0 as f32)
                        .collect(),
                    Prediction::V => x
                        .iter()
                        .zip(out)
                        .map(|(x, v)| a_s0 as f32 * x - s_s0 as f32 * v)
                        .collect(),
                };
                let n = self.timesteps.len();
                // diffusers: the final step is first order (final sigma 0); order 2 needs one previous x0.
                let lower_final = i == n - 1;
                let lam = |a: f64, s: f64| a.ln() - s.ln();
                let h = lam(a_t, s_t) - lam(a_s0, s_s0);
                let first = self.cfg.solver_order == 1 || self.lower_order_nums < 1 || lower_final;
                let e = (-h).exp() - 1.0;
                let mut next: Vec<f32> = x
                    .iter()
                    .zip(&x0)
                    .map(|(x, d)| ((s_t / s_s0) as f32) * x - ((a_t * e) as f32) * d)
                    .collect();
                if let (false, Some(m1)) = (first, &self.prev_x0) {
                    // Midpoint: - 0.5 * alpha_t (e^-h - 1) * (m0 - m1) / r0, r0 = h_0 / h.
                    let (a_s1, s_s1) = vp(self.sigmas[i - 1]);
                    let r0 = (lam(a_s0, s_s0) - lam(a_s1, s_s1)) / h;
                    let c = (0.5 * a_t * e / r0) as f32;
                    next.iter_mut()
                        .zip(x0.iter().zip(m1))
                        .for_each(|(n, (m0, m1))| *n -= c * (m0 - m1));
                }
                *x = next;
                self.prev_x0 = Some(x0);
                if self.lower_order_nums < self.cfg.solver_order {
                    self.lower_order_nums += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(class: &str, extra: Value) -> SchedulerConfig {
        let mut c = json!({"_class_name": class, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
            "beta_end": 0.012, "num_train_timesteps": 1000, "steps_offset": 1, "clip_sample": false});
        for (k, v) in extra.as_object().unwrap() {
            c[k] = v.clone();
        }
        SchedulerConfig::from_json(&c).unwrap()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol * y.abs().max(1.0), "{a:?} vs {b:?}");
        }
    }

    /// diffusers EulerDiscreteScheduler.set_timesteps (stabilityai/sd-turbo: trailing; SDXL base: leading).
    #[test]
    fn euler_timesteps_and_sigmas() {
        let t = cfg(
            "EulerDiscreteScheduler",
            json!({"timestep_spacing": "trailing"}),
        )
        .build(4)
        .unwrap();
        close(&t.timesteps, &[999.0, 749.0, 499.0, 249.0], 0.0);
        // sigma(999) = 14.6146 (diffusers' sigmas.max() for scaled_linear 0.00085..0.012).
        assert!((t.sigmas[0] - 14.6146).abs() < 1e-3 && t.sigmas[4] == 0.0);
        assert!((t.init_sigma() - 14.6146).abs() < 1e-3);
        let l = cfg(
            "EulerDiscreteScheduler",
            json!({"timestep_spacing": "leading"}),
        )
        .build(4)
        .unwrap();
        close(&l.timesteps, &[751.0, 501.0, 251.0, 1.0], 0.0);
        let s0 = l.sigmas[0];
        assert!((l.init_sigma() - (s0 * s0 + 1.0).sqrt()).abs() < 1e-9);
        let lin = cfg(
            "EulerDiscreteScheduler",
            json!({"timestep_spacing": "linspace"}),
        )
        .build(3)
        .unwrap();
        close(&lin.timesteps, &[999.0, 499.5, 0.0], 1e-12);
    }

    /// PNDMScheduler(skip_prk_steps=True), stable-diffusion-v1-5: [::-1] of
    /// concat(_timesteps[:-1], _timesteps[-2:-1], _timesteps[-1:]) with leading _timesteps + offset 1.
    #[test]
    fn pndm_plms_timesteps() {
        let p = cfg("PNDMScheduler", json!({"skip_prk_steps": true}))
            .build(4)
            .unwrap();
        close(&p.timesteps, &[751.0, 501.0, 501.0, 251.0, 1.0], 0.0);
        assert_eq!(
            cfg("PNDMScheduler", json!({"skip_prk_steps": true}))
                .build(50)
                .unwrap()
                .timesteps
                .len(),
            51
        );
    }

    /// LCMScheduler with original_inference_steps 50: origin = [19, 39, ..., 999] reversed, indices
    /// floor(linspace(0, 50, n, endpoint=False)).
    #[test]
    fn lcm_timesteps() {
        let l = cfg("LCMScheduler", json!({"original_inference_steps": 50}))
            .build(4)
            .unwrap();
        close(&l.timesteps, &[999.0, 759.0, 499.0, 259.0], 0.0);
    }

    #[test]
    fn ddim_and_dpm_timesteps() {
        let d = cfg("DDIMScheduler", json!({"timestep_spacing": "leading"}))
            .build(5)
            .unwrap();
        close(&d.timesteps, &[801.0, 601.0, 401.0, 201.0, 1.0], 0.0);
        // DPMSolverMultistep linspace: linspace(0, 999, n + 1).round()[::-1][:-1].
        let m = cfg(
            "DPMSolverMultistepScheduler",
            json!({"timestep_spacing": "linspace"}),
        )
        .build(4)
        .unwrap();
        close(&m.timesteps, &[999.0, 749.0, 500.0, 250.0], 0.0);
        let k = cfg(
            "DPMSolverMultistepScheduler",
            json!({"use_karras_sigmas": true}),
        )
        .build(10)
        .unwrap();
        assert!(
            (k.sigmas[0] - 14.6146).abs() < 1e-3 && (k.sigmas[9] - 0.0292).abs() < 1e-3,
            "{:?}",
            k.sigmas
        );
        assert_eq!(k.timesteps[0], 999.0);
    }

    /// One exact step of each update rule against diffusers' formula on scalars.
    #[test]
    fn step_formulas() {
        let mut rng = Rng::new(0);
        // Euler epsilon: x' = x + (sigma_next - sigma) * eps.
        let mut e = cfg(
            "EulerDiscreteScheduler",
            json!({"timestep_spacing": "trailing"}),
        )
        .build(2)
        .unwrap();
        let (s0, s1) = (e.sigmas[0], e.sigmas[1]);
        let mut x = vec![3.0f32];
        e.step(0, &[0.5], &mut x, &mut rng);
        assert!((x[0] as f64 - (3.0 + (s1 - s0) * 0.5)).abs() < 1e-4);
        // DDIM eta 0: x' = sqrt(a_prev) x0 + sqrt(1 - a_prev) eps.
        let mut d = cfg("DDIMScheduler", json!({"timestep_spacing": "leading"}))
            .build(2)
            .unwrap();
        let (a, ap) = (d.ac(d.timesteps[0]), d.ac(d.timesteps[1]));
        let mut x = vec![1.0f32];
        d.step(0, &[0.2], &mut x, &mut rng);
        let x0 = (1.0 - (1.0 - a).sqrt() * 0.2) / a.sqrt();
        assert!((x[0] as f64 - (ap.sqrt() * x0 + (1.0 - ap).sqrt() * 0.2)).abs() < 1e-5);
        // LCM last step returns the boundary-condition denoised sample.
        let mut l = cfg("LCMScheduler", json!({})).build(1).unwrap();
        let t = l.timesteps[0];
        let a = l.ac(t);
        let mut x = vec![1.0f32];
        l.step(0, &[0.3], &mut x, &mut rng);
        let st = t * 10.0;
        let (cs, co) = (0.25 / (st * st + 0.25), st / (st * st + 0.25f64).sqrt());
        let want = co * (1.0 - (1.0 - a).sqrt() * 0.3) / a.sqrt() + cs;
        assert!((x[0] as f64 - want).abs() < 1e-5);
        // DPM-Solver++ final step (sigma 0) lands exactly on x0.
        let mut m = cfg("DPMSolverMultistepScheduler", json!({}))
            .build(1)
            .unwrap();
        let (av, sv) = (
            1.0 / (m.sigmas[0].powi(2) + 1.0).sqrt(),
            m.sigmas[0] / (m.sigmas[0].powi(2) + 1.0).sqrt(),
        );
        let mut x = vec![0.7f32];
        m.step(0, &[0.1], &mut x, &mut rng);
        assert!((x[0] as f64 - (0.7 - sv * 0.1) / av).abs() < 1e-4);
    }

    #[test]
    fn unsupported_options_are_errors() {
        let base =
            json!({"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"});
        let mut c = base.clone();
        c["rescale_betas_zero_snr"] = json!(true);
        assert!(SchedulerConfig::from_json(&c).is_err());
        let mut c = base.clone();
        c["_class_name"] = json!("UniPCMultistepScheduler");
        assert!(SchedulerConfig::from_json(&c)
            .unwrap_err()
            .contains("UniPC"));
        let mut c = base;
        c["beta_schedule"] = json!("squaredcos_cap_v2");
        assert!(SchedulerConfig::from_json(&c).is_err());
    }
}
