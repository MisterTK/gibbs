//! Gemma 4 audio input: the `Gemma4AudioFeatureExtractor` log-mel spectrogram over WAV input
//! resampled to 16 kHz ([`crate::model::audio`]), and the `gemma4_audio` conformer (2-D conv
//! subsampling, chunked local attention with relative positions, light conv) followed by
//! `embed_audio` into the text width.

use super::ops::{linear, rms_norm, Clipped};
use crate::graph::{DType, Graph, Op, Tensor, UnaryKind, ValueId, Weights as GWeights};
use crate::model::nn;
use crate::model::vision_graph::{Gb, ProgramCache};
use crate::model::par::{par_for, OutPtr};
use crate::model::weights::{Mat, Weights};
use serde_json::Value;

use crate::model::audio::{read_wav, resample, resampled_len};

// ---------------------------------------------------------------------------
// Feature extractor
// ---------------------------------------------------------------------------

pub struct Features {
    rate: u32,
    frame: usize,
    hop: usize,
    bins: usize,
    mel: usize,
    floor: f32,
    max_samples: usize,
    pad_multiple: usize,
    /// `[2 * bins, frame]`: windowed cos rows then sin rows (the real DFT as one GEMM).
    pub(crate) dft: Vec<f32>,
    /// `[mel, bins]` HTK triangular filters, unnormalised.
    pub(crate) filters: Vec<f32>,
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

impl Features {
    pub fn new(fe: &Value) -> Result<Self, String> {
        let u = |k: &str, d: u64| fe[k].as_u64().unwrap_or(d) as usize;
        let num = |k: &str, d: f64| fe[k].as_f64().unwrap_or(d);
        if num("preemphasis", 0.0) != 0.0
            || num("dither", 0.0) != 0.0
            || num("input_scale_factor", 1.0) != 1.0
        {
            return Err(
                "audio feature_extractor: preemphasis/dither/input scaling are not supported"
                    .into(),
            );
        }
        if !fe["per_bin_mean"].is_null() || !fe["per_bin_stddev"].is_null() {
            return Err("audio feature_extractor: per-bin normalisation is not supported".into());
        }
        let rate = u("sampling_rate", 16000) as u32;
        let frame = u("frame_length", 320);
        let fft = u("fft_length", 512);
        let mel = u("feature_size", 128);
        let bins = fft / 2 + 1;
        let tau = 2.0 * std::f64::consts::PI;
        let mut dft = vec![0.0f32; 2 * bins * frame];
        for k in 0..bins {
            for n in 0..frame {
                let win = 0.5 - 0.5 * (tau * n as f64 / frame as f64).cos(); // periodic Hann
                let a = tau * (k * n) as f64 / fft as f64;
                dft[k * frame + n] = (win * a.cos()) as f32;
                dft[(bins + k) * frame + n] = (win * a.sin()) as f32;
            }
        }
        let (fmin, fmax) = (num("min_frequency", 0.0), num("max_frequency", 8000.0));
        let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
        let pts: Vec<f64> = (0..mel + 2)
            .map(|i| mmin + (mmax - mmin) * i as f64 / (mel + 1) as f64)
            .map(|m| 700.0 * (10f64.powf(m / 2595.0) - 1.0))
            .collect();
        let mut filters = vec![0.0f32; mel * bins];
        for j in 0..mel {
            for i in 0..bins {
                let f = (rate / 2) as f64 * i as f64 / (bins - 1) as f64;
                let down = (f - pts[j]) / (pts[j + 1] - pts[j]);
                let up = (pts[j + 2] - f) / (pts[j + 2] - pts[j + 1]);
                filters[j * bins + i] = down.min(up).max(0.0) as f32;
            }
        }
        Ok(Self {
            rate,
            frame,
            hop: u("hop_length", 160),
            bins,
            mel,
            floor: num("mel_floor", 1e-3) as f32,
            max_samples: 480_000, // the extractor's max_length (30 s)
            pad_multiple: 128,
            dft,
            filters,
        })
    }

    /// Frame validity for a clip of `len` samples at `rate` Hz (one entry per frame), checked
    /// against the 30 s limit before anything is resampled.
    fn mask(&self, len: usize, rate: u32) -> Result<Vec<bool>, String> {
        let real = resampled_len(len, rate, self.rate);
        if real > self.max_samples {
            return Err(format!(
                "audio is {:.1} s; Gemma 4 takes at most 30 s per clip",
                len as f64 / rate as f64
            ));
        }
        // Right-pad to the extractor's multiple, then prepend frame/2 zeros (semicausal framing).
        let left = self.frame / 2;
        let padded = left + real.div_ceil(self.pad_multiple) * self.pad_multiple;
        let span = self.frame + 1;
        if padded < span {
            return Err("audio clip is too short".into());
        }
        let t = (padded - span) / self.hop + 1;
        let valid = |i: usize| i >= left && i < left + real;
        Ok((0..t).map(|i| valid(i * self.hop + span - 1)).collect())
    }

    /// The windowing input `[t, frame]` (semicausal framing of the resampled clip) and each frame's
    /// validity.
    pub(crate) fn frames(&self, wav: &[u8]) -> Result<(Vec<f32>, Vec<bool>), String> {
        let (pcm, rate) = read_wav(wav)?;
        let mask = self.mask(pcm.len(), rate)?;
        let t = mask.len();
        let mut x = resample(&pcm, rate, self.rate);
        let real = x.len();
        x.resize(real.div_ceil(self.pad_multiple) * self.pad_multiple, 0.0);
        let padded: Vec<f32> = std::iter::repeat_n(0.0, self.frame / 2).chain(x).collect();
        let frames: Vec<f32> = (0..t)
            .flat_map(|i| &padded[i * self.hop..i * self.hop + self.frame])
            .copied()
            .collect();
        Ok((frames, mask))
    }

    /// `ln(energy + floor)` over the mel energies `[t, mel]`, invalid frames zeroed.
    fn log_mel(&self, mel: &mut [f32], mask: &[bool]) {
        for (row, &ok) in mel.chunks_exact_mut(self.mel).zip(mask) {
            row.iter_mut()
                .for_each(|v| *v = if ok { (*v + self.floor).ln() } else { 0.0 });
        }
    }

    /// Log-mel frames `[t, mel]` (invalid frames zeroed) and each frame's validity (host DFT).
    pub fn extract(&self, wav: &[u8]) -> Result<(Vec<f32>, Vec<bool>), String> {
        crate::hardware::cpu_compute::enter("gemma4::audio::features");
        let (frames, mask) = self.frames(wav)?;
        let t = mask.len();
        let (b, f) = (self.bins, self.frame);
        let mut spec = vec![0.0f32; t * 2 * b];
        nn::matmul_nt(&frames, &self.dft, None, t, f, 2 * b, &mut spec);
        let mag: Vec<f32> = spec
            .chunks_exact(2 * b)
            .flat_map(|r| (0..b).map(move |k| (r[k] * r[k] + r[b + k] * r[b + k]).sqrt()))
            .collect();
        let mut mel = vec![0.0f32; t * self.mel];
        nn::matmul_nt(&mag, &self.filters, None, t, b, self.mel, &mut mel);
        self.log_mel(&mut mel, &mask);
        Ok((mel, mask))
    }

    /// The spectrogram front end as a graph: `frames [t, frame]` → mel energies `[t, mel]`
    /// (windowed DFT as one Linear, magnitude, triangular mel filters). The `ln` and the validity
    /// zeroing are elementwise host steps over the `[t, mel]` result ([`Self::log_mel`]).
    fn front_graph(&self, t: usize) -> (Graph, GWeights) {
        let (b, f, mel) = (self.bins, self.frame, self.mel);
        let mut g = Gb::new();
        let x = g.input("frames", vec![t, f], DType::F32);
        let dft = g.f32w("audio.dft.weight", vec![2 * b, f], self.dft.clone());
        let spec = g.linear("audio.dft", x, dft, None, t, 2 * b);
        let slice = |g: &mut Gb, name: &str, lo: usize| {
            let op = Op::Slice { axis: -1, start: lo as i64, end: (lo + b) as i64 };
            g.op(name, op, &[spec], vec![t, b])
        };
        let (re, im) = (slice(&mut g, "audio.dft.re", 0), slice(&mut g, "audio.dft.im", b));
        let re2 = g.op("audio.dft.re2", Op::Binary { kind: crate::graph::BinaryKind::Mul }, &[re, re], vec![t, b]);
        let im2 = g.op("audio.dft.im2", Op::Binary { kind: crate::graph::BinaryKind::Mul }, &[im, im], vec![t, b]);
        let sum = g.add("audio.dft.pow", re2, im2, vec![t, b]);
        let mag = g.unary("audio.dft.mag", UnaryKind::Sqrt, sum, vec![t, b]);
        let fw = g.f32w("audio.mel.weight", vec![mel, b], self.filters.clone());
        let out = g.linear("audio.mel", mag, fw, None, t, mel);
        g.finish(&[out])
    }
}

// ---------------------------------------------------------------------------
// Conformer
// ---------------------------------------------------------------------------

struct FeedForward {
    pre: Vec<f32>,
    w1: Clipped,
    w2: Clipped,
    post: Vec<f32>,
}

struct Layer {
    ff1: FeedForward,
    ff2: FeedForward,
    norm_pre_attn: Vec<f32>,
    norm_post_attn: Vec<f32>,
    norm_out: Vec<f32>,
    q: Clipped,
    k: Clipped,
    v: Clipped,
    post: Clipped,
    rel_k: Mat,
    /// `q_scale * softplus(per_dim_scale)` per head dim.
    q_dim_scale: Vec<f32>,
    lc_pre: Vec<f32>,
    lc_start: Clipped,
    lc_dw: Vec<f32>,
    lc_norm: Vec<f32>,
    lc_end: Clipped,
}

pub struct Audio {
    features: Features,
    hidden: usize,
    heads: usize,
    left: usize,
    softcap: f32,
    eps: f32,
    residual_weight: f32,
    kernel: usize,
    conv0: (Vec<f32>, Vec<f32>, usize),
    conv1: (Vec<f32>, Vec<f32>, usize),
    in_proj: Mat,
    /// Sinusoidal relative-position rows for distances `left..=0` (descending).
    rel_pos: Vec<f32>,
    layers: Vec<Layer>,
    out_proj: Mat,
    out_bias: Vec<f32>,
    embed: Mat,
    /// Metal programs (set by [`Self::on_gpu`]); `None` = the host forward.
    gpu: Option<ProgramCache>,
}

/// Biased-variance LayerNorm with weight only (`bias=False`).
fn layer_norm(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter_mut()
        .zip(w)
        .for_each(|(v, g)| *v = (*v - mean) * inv * g);
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl Audio {
    pub fn load(w: &Weights, ac: &Value, processor: &Value) -> Result<Self, String> {
        let u = |k: &str| {
            ac[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("audio_config.{k} missing"))
        };
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let hd = hidden / heads;
        let eps = ac["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let left = u("attention_context_left")?.saturating_sub(1);
        if ac["attention_context_right"].as_u64().unwrap_or(0) != 0 {
            return Err("audio attention with right context is not supported".into());
        }
        if ac["hidden_act"].as_str().unwrap_or("silu") != "silu" {
            return Err("audio_config.hidden_act must be silu".into());
        }
        let p = "model.audio_tower";
        let conv = |i: usize| -> Result<(Vec<f32>, Vec<f32>, usize), String> {
            let wn = format!("{p}.subsample_conv_projection.layer{i}.conv.weight");
            let shape = w.shape(&wn)?.to_vec();
            Ok((
                w.vec(&wn)?,
                w.vec_n(
                    &format!("{p}.subsample_conv_projection.layer{i}.norm.weight"),
                    shape[0],
                )?,
                shape[0],
            ))
        };
        let q_scale = (hd as f32).powf(-0.5) / std::f32::consts::LN_2;
        let n = u("num_hidden_layers")?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let lp = format!("{p}.layers.{i}");
            let c = |s: &str| Clipped::load(w, &format!("{lp}.{s}"));
            let v = |s: &str| w.vec_n(&format!("{lp}.{s}.weight"), hidden);
            let ff = |s: &str| -> Result<FeedForward, String> {
                Ok(FeedForward {
                    pre: v(&format!("{s}.pre_layer_norm"))?,
                    w1: c(&format!("{s}.ffw_layer_1"))?,
                    w2: c(&format!("{s}.ffw_layer_2"))?,
                    post: v(&format!("{s}.post_layer_norm"))?,
                })
            };
            let pds = w.vec_n(&format!("{lp}.self_attn.per_dim_scale"), hd)?;
            layers.push(Layer {
                ff1: ff("feed_forward1")?,
                ff2: ff("feed_forward2")?,
                norm_pre_attn: v("norm_pre_attn")?,
                norm_post_attn: v("norm_post_attn")?,
                norm_out: v("norm_out")?,
                q: c("self_attn.q_proj")?,
                k: c("self_attn.k_proj")?,
                v: c("self_attn.v_proj")?,
                post: c("self_attn.post")?,
                rel_k: w.mat(&format!("{lp}.self_attn.relative_k_proj.weight"))?,
                q_dim_scale: pds
                    .iter()
                    .map(|&s| q_scale * (1.0 + s.exp()).ln())
                    .collect(),
                lc_pre: v("lconv1d.pre_layer_norm")?,
                lc_start: c("lconv1d.linear_start")?,
                lc_dw: w.vec(&format!("{lp}.lconv1d.depthwise_conv1d.weight"))?,
                lc_norm: v("lconv1d.conv_norm")?,
                lc_end: c("lconv1d.linear_end")?,
            });
        }
        // Relative positions `left, left-1, ..., 0`: [sin(p * inv), cos(p * inv)].
        let half = hidden / 2;
        let inc = (10000f64).ln() / (half.max(2) - 1) as f64;
        let rel_pos = (0..=left)
            .rev()
            .flat_map(|p| {
                let sin =
                    (0..half).map(move |i| (p as f64 * (-(i as f64) * inc).exp()).sin() as f32);
                let cos =
                    (0..half).map(move |i| (p as f64 * (-(i as f64) * inc).exp()).cos() as f32);
                sin.chain(cos)
            })
            .collect();
        let out_proj = w.mat(&format!("{p}.output_proj.weight"))?;
        let kernel = u("conv_kernel_size")?;
        Ok(Self {
            features: Features::new(&processor["feature_extractor"])?,
            hidden,
            heads,
            left,
            softcap: ac["attention_logit_cap"].as_f64().unwrap_or(50.0) as f32,
            eps,
            residual_weight: ac["residual_weight"].as_f64().unwrap_or(0.5) as f32,
            kernel,
            conv0: conv(0)?,
            conv1: conv(1)?,
            in_proj: w.mat(&format!(
                "{p}.subsample_conv_projection.input_proj_linear.weight"
            ))?,
            rel_pos,
            layers,
            out_bias: w.vec_n(&format!("{p}.output_proj.bias"), out_proj.rows)?,
            out_proj,
            embed: w.mat("model.embed_audio.embedding_projection.weight")?,
            gpu: None,
        })
    }

    /// Runs the front end and conformer as Metal graph programs on `mode`.
    pub fn on_gpu(mut self, mode: crate::hardware::PlacementMode) -> Self {
        self.gpu = Some(ProgramCache::new(mode));
        self
    }

    /// One 3x3 stride-2 conv block: masked input `[c, t, f]` -> conv -> channel LayerNorm -> ReLU.
    fn sscp(
        &self,
        x: &[f32],
        c_in: usize,
        t: usize,
        f: usize,
        mask: &[bool],
        layer: &(Vec<f32>, Vec<f32>, usize),
    ) -> (Vec<f32>, usize, usize) {
        let mut x = x.to_vec();
        for c in 0..c_in {
            for (ti, &ok) in mask.iter().enumerate() {
                if !ok {
                    x[(c * t + ti) * f..][..f].fill(0.0);
                }
            }
        }
        let (w, norm, c_out) = layer;
        let (y, oh, ow) = nn::conv2d(&x, c_in, t, f, w, None, *c_out, 3, 2, 1);
        let plane = oh * ow;
        let mut out = vec![0.0f32; c_out * plane];
        let mut v = vec![0.0f32; *c_out];
        for s in 0..plane {
            (0..*c_out).for_each(|c| v[c] = y[c * plane + s]);
            layer_norm(&mut v, norm, self.eps);
            (0..*c_out).for_each(|c| out[c * plane + s] = v[c].max(0.0));
        }
        (out, oh, ow)
    }

    fn feed_forward(&self, ff: &FeedForward, x: &mut [f32], t: usize) {
        let mut a = x.to_vec();
        rms_norm(&mut a, self.hidden, Some(&ff.pre), self.eps);
        let mut a = ff.w1.forward(&a, t);
        a.iter_mut().for_each(|v| *v = silu(*v));
        let mut a = ff.w2.forward(&a, t);
        rms_norm(&mut a, self.hidden, Some(&ff.post), self.eps);
        x.iter_mut()
            .zip(&a)
            .for_each(|(x, a)| *x += self.residual_weight * a);
    }

    /// Chunked local self-attention: each query sees the `left`-wide causal window of valid keys
    /// (distances `0..left`, as HF's `sliding_window_mask_function((left, 0))`; the relative table
    /// also has a distance-`left` row that the mask never selects), scores = q.k + q.rel[distance],
    /// softcapped.
    fn attention(&self, l: &Layer, a: &[f32], t: usize, valid: &[bool]) -> Vec<f32> {
        let (d, nh) = (self.hidden, self.heads);
        let hd = d / nh;
        let mut q = l.q.forward(a, t);
        let mut k = l.k.forward(a, t);
        let v = l.v.forward(a, t);
        for row in q.chunks_exact_mut(hd) {
            row.iter_mut()
                .zip(&l.q_dim_scale)
                .for_each(|(x, s)| *x *= s);
        }
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        k.iter_mut().for_each(|x| *x *= k_scale);
        // rel[c] for distance left - c.
        let rel = linear(&self.rel_pos, self.left + 1, &l.rel_k);
        let mut out = vec![0.0f32; t * d];
        let o = OutPtr::new(&mut out);
        let cap = self.softcap;
        par_for(t * nh, &|task| {
            let (qi, h) = (task / nh, task % nh);
            let qv = &q[qi * d + h * hd..][..hd];
            let lo = (qi + 1).saturating_sub(self.left);
            let keys: Vec<usize> = (lo..=qi).filter(|&j| valid[j]).collect();
            let mut s: Vec<f32> = keys
                .iter()
                .map(|&j| {
                    let kv = &k[j * d + h * hd..][..hd];
                    let rv = &rel[(self.left - (qi - j)) * d + h * hd..][..hd];
                    let logit: f32 = qv
                        .iter()
                        .zip(kv)
                        .zip(rv)
                        .map(|((q, k), r)| q * (k + r))
                        .sum();
                    (logit / cap).tanh() * cap
                })
                .collect();
            // SAFETY: (query, head) slot belongs to this task.
            let dst = unsafe { o.slice(qi * d + h * hd, hd) };
            if s.is_empty() {
                return; // padding query: its output is discarded
            }
            crate::model::simd::vector_softmax(&mut s);
            for (&j, &p) in keys.iter().zip(&s) {
                crate::model::simd::vector_fused_scale_add(dst, &v[j * d + h * hd..][..hd], p);
            }
        });
        l.post.forward(&out, t)
    }

    fn light_conv(&self, l: &Layer, x: &mut [f32], t: usize) {
        let d = self.hidden;
        let mut a = x.to_vec();
        rms_norm(&mut a, d, Some(&l.lc_pre), self.eps);
        let g = l.lc_start.forward(&a, t);
        let glu: Vec<f32> = g
            .chunks_exact(2 * d)
            .flat_map(|r| (0..d).map(move |c| r[c] / (1.0 + (-r[d + c]).exp())))
            .collect();
        let kz = self.kernel;
        let mut y = vec![0.0f32; t * d];
        for ti in 0..t {
            for j in 0..kz {
                let Some(src) = (ti + j).checked_sub(kz - 1) else {
                    continue;
                };
                for c in 0..d {
                    y[ti * d + c] += l.lc_dw[c * kz + j] * glu[src * d + c];
                }
            }
        }
        rms_norm(&mut y, d, Some(&l.lc_norm), self.eps);
        y.iter_mut().for_each(|v| *v = silu(*v));
        let e = l.lc_end.forward(&y, t);
        x.iter_mut().zip(&e).for_each(|(x, e)| *x += e);
    }

    /// The `n` [`Self::encode`] returns: frames still valid after the two stride-2 convolutions.
    pub fn count(&self, wav: &[u8]) -> Result<usize, String> {
        let (pcm, rate) = read_wav(wav)?;
        let mask = self.features.mask(pcm.len(), rate)?;
        match mask.iter().step_by(4).filter(|&&ok| ok).count() {
            0 => Err("audio clip is too short".into()),
            n => Ok(n),
        }
    }

    /// Encodes one WAV clip to `[n, text_hidden]` soft tokens; returns `(rows, n)`.
    pub fn encode(&self, wav: &[u8]) -> Result<(Vec<f32>, usize), String> {
        match &self.gpu {
            Some(g) => self.encode_gpu(g, wav),
            None => self.encode_cpu(wav),
        }
    }

    /// Output frames of a stride-2, 3x3, pad-1 convolution over `n` rows.
    fn sub(n: usize) -> usize {
        (n - 1) / 2 + 1
    }

    /// The Metal path: log-mel front end program, then the conformer program (compiled once per
    /// clip length).
    fn encode_gpu(&self, gpu: &ProgramCache, wav: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (frames, mask) = self.features.frames(wav)?;
        let (t0, f) = (mask.len(), self.features.mel);
        let front = gpu.get(&[0, t0], || Ok(self.features.front_graph(t0)))?;
        let energies = front.lock().run(vec![Tensor::F32(frames)])?;
        let mut mel = energies.into_iter().next().ok_or("audio front end returned nothing")?.into_f32()?;
        self.features.log_mel(&mut mel, &mask);
        let (t1, f1) = (Self::sub(t0), Self::sub(f));
        let t = Self::sub(t1);
        let mask1: Vec<bool> = mask.iter().step_by(2).copied().take(t1).collect();
        let valid: Vec<bool> = mask1.iter().step_by(2).copied().take(t).collect();
        let n = valid.iter().filter(|&&v| v).count();
        if n == 0 {
            return Err("audio clip is too short".into());
        }
        if !valid[..n].iter().all(|&v| v) {
            return Err("audio frame mask is not a prefix".into());
        }
        let (nh, left) = (self.heads, self.left);
        let mask_plane: Vec<f32> = mask1.iter().flat_map(|&ok| std::iter::repeat_n(ok as u8 as f32, f1)).collect();
        // Query `i` sees keys `i-left < j <= i` that are valid; the relative-position score of the
        // pair sits at column `left - (i - j)` of the `[t, left + 1]` table `q · rel`.
        let mut bias = vec![-1e30f32; t * t];
        let mut idx = vec![0u32; nh * t * t];
        for i in 0..t {
            for j in (i + 1).saturating_sub(left)..=i {
                if valid[j] {
                    bias[i * t + j] = 0.0;
                    for h in 0..nh {
                        idx[(h * t + i) * t + j] = ((h * t + i) * (left + 1) + left - (i - j)) as u32;
                    }
                }
            }
        }
        let prog = gpu.get(&[1, t0], || self.conformer_graph(t0))?;
        let out = prog.lock().run(vec![
            Tensor::F32(mel),
            Tensor::F32(mask_plane),
            Tensor::F32(bias),
            Tensor::U32(idx),
        ])?;
        let mut rows = out.into_iter().next().ok_or("audio program returned nothing")?.into_f32()?;
        rows.truncate(n * self.embed.rows);
        Ok((rows, n))
    }

    /// One subsampling block as nodes: `x [c_in, t, f]` → conv (3x3, stride 2, pad 1) → channel
    /// LayerNorm → ReLU; returns the rows `[t', f' * c_out]`-ordered `[t' * f', c_out]` and `(t', f')`.
    fn sscp_graph(&self, b: &mut Gb, name: &str, x: ValueId, c_in: usize, tf: (usize, usize), layer: &(Vec<f32>, Vec<f32>, usize)) -> (ValueId, usize, usize) {
        let (w, norm, c_out) = layer;
        let (t, f) = (Self::sub(tf.0), Self::sub(tf.1));
        let cw = b.f32w(&format!("{name}.weight"), vec![*c_out, c_in, 3, 3], w.clone());
        let conv = Op::Conv2d { stride: [2, 2], padding: [1; 4], dilation: [1, 1], groups: 1 };
        let y = b.op(name, conv, &[x, cw], vec![*c_out, t, f]);
        let y = b.reshape(&format!("{name}.flat"), y, vec![*c_out, t * f]);
        let y = b.transpose(&format!("{name}.rows"), y, &[*c_out, t * f], &[1, 0]);
        let gm = b.f32w(&format!("{name}.norm.weight"), vec![*c_out], norm.clone());
        let bt = b.f32w(&format!("{name}.norm.bias"), vec![*c_out], vec![0.0; *c_out]);
        let y = b.op(&format!("{name}.norm"), Op::LayerNorm { eps: self.eps }, &[y, gm, bt], vec![t * f, *c_out]);
        let y = b.unary(&format!("{name}.relu"), UnaryKind::Relu, y, vec![t * f, *c_out]);
        (y, t, f)
    }

    /// The conformer + output projection as a graph for a clip of `t0` mel frames. Inputs: `mel
    /// [1, t0, mel]` (log-mel, invalid frames zeroed), `mask1 [1, t1 * f1]` (layer-0 output
    /// validity), `attn_bias [t, t]` (0 / -1e30 local-window and padding mask) and `rel_idx
    /// [heads * t * t]` (U32: where each (query, key) pair reads its relative-position score).
    /// Output: `[t, text_hidden]` (rows past the valid prefix are padding).
    fn conformer_graph(&self, t0: usize) -> Result<(Graph, GWeights), String> {
        let (d, nh, eps, left) = (self.hidden, self.heads, self.eps, self.left);
        let hd = d / nh;
        let f = self.features.mel;
        let (t1, f1) = (Self::sub(t0), Self::sub(f));
        let t = Self::sub(t1);
        let mut b = Gb::new();
        let mel = b.input("mel", vec![1, t0, f], DType::F32);
        let m1 = b.input("mask1", vec![1, t1 * f1], DType::F32);
        let bias = b.input("attn_bias", vec![t, t], DType::F32);
        let idx = b.input("rel_idx", vec![nh * t * t], DType::U32);
        let (y0, t1_, f1_) = self.sscp_graph(&mut b, "audio.sscp0", mel, 1, (t0, f), &self.conv0);
        debug_assert_eq!((t1_, f1_), (t1, f1));
        let c0 = self.conv0.2;
        let y0 = b.transpose("audio.sscp0.t", y0, &[t1 * f1, c0], &[1, 0]);
        let y0 = b.binary("audio.sscp0.mask", crate::graph::BinaryKind::Mul, y0, m1, vec![c0, t1 * f1]);
        let y0 = b.reshape("audio.sscp0.grid", y0, vec![c0, t1, f1]);
        let (y1, t2, f2) = self.sscp_graph(&mut b, "audio.sscp1", y0, c0, (t1, f1), &self.conv1);
        debug_assert_eq!(t2, t);
        let c1 = self.conv1.2;
        let flat = b.reshape("audio.sscp.flat", y1, vec![t, f2 * c1]);
        let w = b.mat("audio.in_proj.weight", &self.in_proj);
        let mut h = b.linear("audio.in_proj", flat, w, None, t, d);
        let rel_pos = b.f32w("audio.rel_pos", vec![left + 1, d], self.rel_pos.clone());
        let rw = self.residual_weight;
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        let ff = |b: &mut Gb, p: &str, h: ValueId, ff: &FeedForward| {
            let a = b.rms(&format!("{p}.pre"), h, ff.pre.clone(), eps, vec![t, d]);
            let a = ff.w1.add_to(b, &format!("{p}.w1"), a, t);
            let inner = ff.w1.w.rows;
            let a = b.unary(&format!("{p}.silu"), UnaryKind::Silu, a, vec![t, inner]);
            let a = ff.w2.add_to(b, &format!("{p}.w2"), a, t);
            let a = b.rms(&format!("{p}.post"), a, ff.post.clone(), eps, vec![t, d]);
            let a = b.unary(&format!("{p}.res_scale"), UnaryKind::Scale(rw), a, vec![t, d]);
            b.add(&format!("{p}.res"), h, a, vec![t, d])
        };
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("audio.layers.{i}");
            h = ff(&mut b, &format!("{p}.ff1"), h, &l.ff1);
            // Chunked local attention with relative positions: scores = softcap(q·k + q·rel[dist]).
            let a = b.rms(&format!("{p}.attn_pre"), h, l.norm_pre_attn.clone(), eps, vec![t, d]);
            let q = l.q.add_to(&mut b, &format!("{p}.q"), a, t);
            let k = l.k.add_to(&mut b, &format!("{p}.k"), a, t);
            let v = l.v.add_to(&mut b, &format!("{p}.v"), a, t);
            let qs = b.f32w(&format!("{p}.q_scale"), vec![d], l.q_dim_scale.repeat(nh));
            let q = b.binary(&format!("{p}.q_scaled"), crate::graph::BinaryKind::Mul, q, qs, vec![t, d]);
            let k = b.unary(&format!("{p}.k_scaled"), UnaryKind::Scale(k_scale), k, vec![t, d]);
            let rw_ = b.mat(&format!("{p}.rel_k.weight"), &l.rel_k);
            let rel = b.linear(&format!("{p}.rel"), rel_pos, rw_, None, left + 1, d);
            let heads = |b: &mut Gb, name: &str, x: ValueId, rows: usize| {
                let x = b.reshape(&format!("{name}.split"), x, vec![rows, nh, hd]);
                b.transpose(&format!("{name}.heads"), x, &[rows, nh, hd], &[1, 0, 2])
            };
            let (qh, kh, vh) = (heads(&mut b, &format!("{p}.qh"), q, t), heads(&mut b, &format!("{p}.kh"), k, t), heads(&mut b, &format!("{p}.vh"), v, t));
            let rh = heads(&mut b, &format!("{p}.rh"), rel, left + 1);
            let mm = |tb: bool| Op::MatMul { transpose_a: false, transpose_b: tb };
            let s1 = b.op(&format!("{p}.qk"), mm(true), &[qh, kh], vec![nh, t, t]);
            let s2 = b.op(&format!("{p}.qr"), mm(true), &[qh, rh], vec![nh, t, left + 1]);
            let s2 = b.reshape(&format!("{p}.qr.flat"), s2, vec![nh * t * (left + 1)]);
            let s2 = b.op(&format!("{p}.qr.pick"), Op::Gather { axis: 0 }, &[s2, idx], vec![nh * t * t]);
            let s2 = b.reshape(&format!("{p}.qr.grid"), s2, vec![nh, t, t]);
            let s = b.add(&format!("{p}.scores"), s1, s2, vec![nh, t, t]);
            let s = b.unary(&format!("{p}.softcap"), UnaryKind::SoftCap(self.softcap), s, vec![nh, t, t]);
            let s = b.add(&format!("{p}.masked"), s, bias, vec![nh, t, t]);
            let pr = b.op(&format!("{p}.softmax"), Op::Softmax { axis: -1 }, &[s], vec![nh, t, t]);
            let o = b.op(&format!("{p}.pv"), mm(false), &[pr, vh], vec![nh, t, hd]);
            let o = b.transpose(&format!("{p}.o_t"), o, &[nh, t, hd], &[1, 0, 2]);
            let o = b.reshape(&format!("{p}.o_flat"), o, vec![t, d]);
            let o = l.post.add_to(&mut b, &format!("{p}.post"), o, t);
            let o = b.rms(&format!("{p}.attn_post"), o, l.norm_post_attn.clone(), eps, vec![t, d]);
            h = b.add(&format!("{p}.attn_res"), h, o, vec![t, d]);
            // Light conv: GLU, causal depthwise conv over time, norm, SiLU, projection.
            let a = b.rms(&format!("{p}.lc_pre"), h, l.lc_pre.clone(), eps, vec![t, d]);
            let g = l.lc_start.add_to(&mut b, &format!("{p}.lc_start"), a, t);
            let half = |b: &mut Gb, tag: &str, lo: usize| {
                let op = Op::Slice { axis: -1, start: lo as i64, end: (lo + d) as i64 };
                b.op(&format!("{p}.lc_{tag}"), op, &[g], vec![t, d])
            };
            let (ga, gb) = (half(&mut b, "a", 0), half(&mut b, "b", d));
            let gb = b.unary(&format!("{p}.lc_gate"), UnaryKind::Sigmoid, gb, vec![t, d]);
            let glu = b.op(&format!("{p}.lc_glu"), Op::Binary { kind: crate::graph::BinaryKind::Mul }, &[ga, gb], vec![t, d]);
            let gt = b.transpose(&format!("{p}.lc_glu.t"), glu, &[t, d], &[1, 0]);
            let kz = self.kernel;
            let dw = b.f32w(&format!("{p}.lc_dw.weight"), vec![d, 1, kz], l.lc_dw.clone());
            let conv = Op::Conv1d { stride: 1, padding: [kz - 1, 0], dilation: 1, groups: d };
            let y = b.op(&format!("{p}.lc_dw"), conv, &[gt, dw], vec![d, t]);
            let y = b.transpose(&format!("{p}.lc_dw.t"), y, &[d, t], &[1, 0]);
            let y = b.rms(&format!("{p}.lc_norm"), y, l.lc_norm.clone(), eps, vec![t, d]);
            let y = b.unary(&format!("{p}.lc_silu"), UnaryKind::Silu, y, vec![t, d]);
            let e = l.lc_end.add_to(&mut b, &format!("{p}.lc_end"), y, t);
            h = b.add(&format!("{p}.lc_res"), h, e, vec![t, d]);
            h = ff(&mut b, &format!("{p}.ff2"), h, &l.ff2);
            h = b.rms(&format!("{p}.out_norm"), h, l.norm_out.clone(), eps, vec![t, d]);
        }
        let od = self.out_proj.rows;
        let ow = b.mat("audio.out_proj.weight", &self.out_proj);
        let ob = b.f32w("audio.out_proj.bias", vec![od], self.out_bias.clone());
        let o = b.linear("audio.out_proj", h, ow, Some(ob), t, od);
        let o = b.rms("audio.out_norm", o, vec![1.0; od], eps, vec![t, od]);
        let ew = b.mat("audio.embed.weight", &self.embed);
        let out = b.linear("audio.embed", o, ew, None, t, self.embed.rows);
        Ok(b.finish(&[out]))
    }

    /// The host forward (`--backend cpu`).
    // ponytail: the conformer's 1e10 gradient-clipping clamps are omitted (no-ops on finite f32 activations).
    fn encode_cpu(&self, wav: &[u8]) -> Result<(Vec<f32>, usize), String> {
        crate::hardware::cpu_compute::enter("gemma4::audio::forward");
        let (mel, mask) = self.features.extract(wav)?;
        let f = self.features.mel;
        let t0 = mask.len();
        let (x, t1, f1) = self.sscp(&mel, 1, t0, f, &mask, &self.conv0);
        let mask1: Vec<bool> = mask.iter().step_by(2).copied().take(t1).collect();
        let (x, t, f2) = self.sscp(&x, self.conv0.2, t1, f1, &mask1, &self.conv1);
        let valid: Vec<bool> = mask1.iter().step_by(2).copied().take(t).collect();
        let c2 = self.conv1.2;
        // [c, t, f] -> rows t of (f, c).
        let mut flat = vec![0.0f32; t * f2 * c2];
        for c in 0..c2 {
            for ti in 0..t {
                for fi in 0..f2 {
                    flat[ti * f2 * c2 + fi * c2 + c] = x[(c * t + ti) * f2 + fi];
                }
            }
        }
        let mut h = linear(&flat, t, &self.in_proj);
        let d = self.hidden;
        for l in &self.layers {
            self.feed_forward(&l.ff1, &mut h, t);
            let mut a = h.clone();
            rms_norm(&mut a, d, Some(&l.norm_pre_attn), self.eps);
            let mut a = self.attention(l, &a, t, &valid);
            rms_norm(&mut a, d, Some(&l.norm_post_attn), self.eps);
            h.iter_mut().zip(&a).for_each(|(h, a)| *h += a);
            self.light_conv(l, &mut h, t);
            self.feed_forward(&l.ff2, &mut h, t);
            rms_norm(&mut h, d, Some(&l.norm_out), self.eps);
        }
        let kept: Vec<f32> = h
            .chunks_exact(d)
            .zip(&valid)
            .filter(|(_, &ok)| ok)
            .flat_map(|(r, _)| r)
            .copied()
            .collect();
        let n = kept.len() / d;
        if n == 0 {
            return Err("audio clip is too short".into());
        }
        let od = self.out_proj.rows;
        let mut o = linear(&kept, n, &self.out_proj);
        for row in o.chunks_exact_mut(od) {
            row.iter_mut()
                .zip(&self.out_bias)
                .for_each(|(v, b)| *v += b);
        }
        rms_norm(&mut o, od, None, self.eps);
        Ok((linear(&o, n, &self.embed), n))
    }
}

/// Deterministic synthetic tiny conformer (hidden 16, 2 layers, 16 mel bins, text width 24): the
/// fixture of `tests/cpu_free_matrix.rs` and the CPU-vs-Metal parity tests.
pub mod tiny {
    use super::*;
    use crate::model::siglip::tiny::values;

    pub const TEXT: usize = 24;
    const D: usize = 16;
    const MEL: usize = 16;

    fn specs() -> Vec<(String, Vec<usize>)> {
        let p = "model.audio_tower";
        let mut s: Vec<(String, Vec<usize>)> = vec![
            (format!("{p}.subsample_conv_projection.layer0.conv.weight"), vec![4, 1, 3, 3]),
            (format!("{p}.subsample_conv_projection.layer0.norm.weight"), vec![4]),
            (format!("{p}.subsample_conv_projection.layer1.conv.weight"), vec![8, 4, 3, 3]),
            (format!("{p}.subsample_conv_projection.layer1.norm.weight"), vec![8]),
            (format!("{p}.subsample_conv_projection.input_proj_linear.weight"), vec![D, 32]),
            (format!("{p}.output_proj.weight"), vec![12, D]),
            (format!("{p}.output_proj.bias"), vec![12]),
            ("model.embed_audio.embedding_projection.weight".into(), vec![TEXT, 12]),
        ];
        for i in 0..2 {
            let l = format!("{p}.layers.{i}");
            for ff in ["feed_forward1", "feed_forward2"] {
                s.push((format!("{l}.{ff}.pre_layer_norm.weight"), vec![D]));
                s.push((format!("{l}.{ff}.post_layer_norm.weight"), vec![D]));
                s.push((format!("{l}.{ff}.ffw_layer_1.linear.weight"), vec![32, D]));
                s.push((format!("{l}.{ff}.ffw_layer_2.linear.weight"), vec![D, 32]));
            }
            for n in ["norm_pre_attn", "norm_post_attn", "norm_out", "lconv1d.pre_layer_norm", "lconv1d.conv_norm"] {
                s.push((format!("{l}.{n}.weight"), vec![D]));
            }
            for n in ["q_proj", "k_proj", "v_proj", "post"] {
                s.push((format!("{l}.self_attn.{n}.linear.weight"), vec![D, D]));
            }
            s.push((format!("{l}.self_attn.relative_k_proj.weight"), vec![D, D]));
            s.push((format!("{l}.self_attn.per_dim_scale"), vec![D / 2]));
            s.push((format!("{l}.lconv1d.linear_start.linear.weight"), vec![2 * D, D]));
            s.push((format!("{l}.lconv1d.depthwise_conv1d.weight"), vec![D, 1, 5]));
            s.push((format!("{l}.lconv1d.linear_end.linear.weight"), vec![D, D]));
            // `ClippableLinear` bounds on the first feed-forward's first layer.
            for b in ["input_min", "input_max", "output_min", "output_max"] {
                s.push((format!("{l}.feed_forward1.ffw_layer_1.{b}"), vec![1]));
            }
        }
        s
    }

    pub fn weights() -> Weights {
        let specs = specs();
        let shapes: std::collections::HashMap<_, _> = specs.iter().cloned().collect();
        Weights::from_f32(specs, |name| {
            let n: usize = shapes[name].iter().product();
            Ok(match name {
                s if s.ends_with("input_min") => vec![-2.0],
                s if s.ends_with("input_max") => vec![2.0],
                s if s.ends_with("output_min") => vec![-3.0],
                s if s.ends_with("output_max") => vec![3.0],
                s if shapes[s].len() == 1 && s.ends_with("weight") => {
                    values(s, n).into_iter().map(|v| 1.0 + 0.1 * v).collect()
                }
                s => values(s, n).into_iter().map(|v| 0.2 * v).collect(),
            })
        })
        .expect("tiny audio weights")
    }

    pub fn audio() -> Audio {
        let ac = serde_json::json!({
            "hidden_size": D, "num_attention_heads": 2, "num_hidden_layers": 2,
            "attention_context_left": 5, "attention_context_right": 0, "conv_kernel_size": 5,
            "rms_norm_eps": 1e-6, "attention_logit_cap": 50.0, "residual_weight": 0.5,
        });
        let processor = serde_json::json!({"feature_extractor": {
            "sampling_rate": 16000, "frame_length": 32, "fft_length": 64, "feature_size": MEL,
            "hop_length": 16, "mel_floor": 1e-3,
        }});
        Audio::load(&weights(), &ac, &processor).expect("tiny audio loads")
    }

    /// A 0.25 s 16 kHz mono 16-bit WAV: two tones.
    pub fn wav() -> Vec<u8> {
        let pcm: Vec<u8> = (0..4000)
            .flat_map(|i| {
                let x = i as f32 / 16000.0;
                let v = 0.4 * (2.0 * std::f32::consts::PI * 440.0 * x).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1750.0 * x).sin();
                ((v * 32767.0) as i16).to_le_bytes()
            })
            .collect();
        let mut b = Vec::new();
        b.extend(b"RIFF");
        b.extend((36 + pcm.len() as u32).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend([1, 0, 1, 0]);
        b.extend(16000u32.to_le_bytes());
        b.extend(32000u32.to_le_bytes());
        b.extend([2, 0, 16, 0]);
        b.extend(b"data");
        b.extend((pcm.len() as u32).to_le_bytes());
        b.extend(pcm);
        b
    }

    /// The conformer graph for a clip of `t0` mel frames.
    pub fn graph(t0: usize) -> (Graph, GWeights) {
        audio().conformer_graph(t0).expect("tiny audio graph")
    }

    /// The conformer's `(mel frames, graph)` for [`wav`] (the fixture's real shape).
    pub fn wav_graph() -> (Graph, GWeights) {
        let a = audio();
        let (_, mask) = a.features.frames(&wav()).expect("tiny wav frames");
        a.conformer_graph(mask.len()).expect("tiny audio graph")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `graph::shape::infer` reproduces every node of the tiny conformer graphs.
    #[test]
    fn conformer_graph_shapes_match_inference() {
        crate::graph::shape::assert_shapes(&tiny::wav_graph().0);
        for t0 in [8, 37] {
            crate::graph::shape::assert_shapes(&tiny::graph(t0).0);
        }
    }

    fn wav(format: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        crate::model::audio::wav_bytes(format, bits, channels, rate, data)
    }

    #[test]
    fn over_long_audio_is_rejected_before_resampling() {
        let fe = Features::new(&serde_json::json!({})).unwrap();
        // 31 s at 8 kHz: the 16 kHz output would exceed the 30 s cap.
        let pcm = vec![0u8; 31 * 8000 * 2];
        let err = fe.extract(&wav(1, 16, 1, 8000, &pcm)).unwrap_err();
        assert!(err.contains("at most 30 s"), "{err}");
    }

    #[test]
    fn log_mel_peaks_at_the_tone_and_masks_padding() {
        let fe = Features::new(&serde_json::json!({})).unwrap();
        // 1 kHz tone, 0.5 s: 8000 samples -> padded to 8064.
        let pcm: Vec<u8> = (0..8000)
            .flat_map(|i| {
                (((2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin() * 16000.0)
                    as i16)
                    .to_le_bytes()
            })
            .collect();
        let (mel, mask) = fe.extract(&wav(1, 16, 1, 16000, &pcm)).unwrap();
        assert_eq!(mask.len(), (8064 + 160 - 321) / 160 + 1);
        assert!(mask[..48].iter().all(|&m| m) && !mask[mask.len() - 1]);
        let row = &mel[20 * 128..21 * 128];
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let peak_hz =
            700.0 * (10f64.powf(hz_to_mel(8000.0) * (peak + 1) as f64 / 129.0 / 2595.0) - 1.0);
        assert!(
            (peak_hz - 1000.0).abs() < 60.0,
            "peak filter centre {peak_hz} Hz"
        );
    }

    /// The Metal front end + conformer match the host forward on a real clip end to end.
    #[test]
    fn metal_conformer_matches_the_host_forward() {
        use crate::hardware::{available_devices, cpu_compute, Device, PlacementMode};
        if !available_devices().contains(&Device::Metal) {
            println!("SKIP: no Metal device");
            return;
        }
        let _allow = cpu_compute::allow();
        let cpu = super::tiny::audio();
        let gpu = super::tiny::audio().on_gpu(PlacementMode::Pin(Device::Metal));
        let wav = super::tiny::wav();
        let (want, n) = cpu.encode(&wav).unwrap();
        let (got, m) = gpu.encode(&wav).unwrap();
        assert_eq!((n, got.len()), (m, want.len()));
        assert!(n > 4, "clip too short to exercise the window: {n}");
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let err = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
        println!("gemma4 audio {n} tokens: rel err {err:e}");
        assert!(err < 1e-3, "{err}");
        let abs = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(abs < 1e-3, "max |d| {abs}");
        let (_, mask) = gpu.features.frames(&wav).unwrap();
        let cache = gpu.gpu.as_ref().unwrap();
        for key in [[0, mask.len()], [1, mask.len()]] {
            let p = cache.get(&key, || unreachable!("compiled by encode")).unwrap();
            assert_eq!(p.lock().cpu_node_count(), 0, "nodes run on the CPU");
        }
    }
}
