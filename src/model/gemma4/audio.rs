//! Gemma 4 audio input: WAV decoding (PCM 8/16/24/32-bit, IEEE float), resampling to 16 kHz,
//! the `Gemma4AudioFeatureExtractor` log-mel spectrogram, and the `gemma4_audio` conformer
//! (2-D conv subsampling, chunked local attention with relative positions, light conv) followed by
//! `embed_audio` into the text width.

use super::ops::{linear, rms_norm, Clipped};
use crate::model::nn;
use crate::model::par::{par_for, OutPtr};
use crate::model::weights::{Mat, Weights};
use serde_json::Value;

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

/// Mono samples in [-1, 1] and the sample rate of a RIFF/WAVE file; other containers or codecs
/// are an Err.
pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("audio is not a RIFF/WAVE file (only WAV input is supported)".into());
    }
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut fmt, mut data) = (None, None);
    let mut o = 12;
    while o + 8 <= b.len() {
        let size = u32_at(o + 4) as usize;
        let body = &b[o + 8..(o + 8).saturating_add(size).min(b.len())];
        match &b[o..o + 4] {
            b"fmt " if body.len() >= 16 => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        o += 8 + size + (size & 1);
    }
    let fmt = fmt.ok_or("WAV has no fmt chunk")?;
    let data = data.ok_or("WAV has no data chunk")?;
    let f = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
    let mut format = f(0);
    let channels = f(2) as usize;
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits = f(14) as usize;
    if format == 0xFFFE && fmt.len() >= 26 {
        format = f(24); // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the format tag
    }
    let bytes = bits / 8;
    if channels == 0 || bytes == 0 || !bits.is_multiple_of(8) {
        return Err(format!(
            "WAV fmt is invalid ({channels} channels, {rate} Hz, {bits} bits)"
        ));
    }
    if !(1_000..=768_000).contains(&rate) {
        return Err(format!(
            "WAV sample rate {rate} Hz is outside 1 kHz..768 kHz"
        ));
    }
    let sample = |s: &[u8]| -> f32 {
        match (format, bits) {
            (1, 8) => (s[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            (3, 64) => f64::from_le_bytes(s[..8].try_into().unwrap()) as f32,
            _ => f32::NAN,
        }
    };
    if sample(&[0; 8]).is_nan() {
        return Err(format!(
            "WAV codec {format} at {bits} bits is not supported (PCM or IEEE float only)"
        ));
    }
    let frame = bytes * channels;
    let mono = data
        .chunks_exact(frame)
        .map(|fr| fr.chunks_exact(bytes).map(sample).sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, rate))
}

/// Samples [`resample`] returns for `len` input samples.
pub fn resampled_len(len: usize, from: u32, to: u32) -> usize {
    if from == to {
        return len;
    }
    (len as f64 / (from as f64 / to as f64)).round() as usize
}

/// Band-limited resampling (Hann-windowed sinc, 16 zero crossings).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    const ZEROS: f64 = 16.0;
    let ratio = from as f64 / to as f64;
    let cutoff = (1.0 / ratio).min(1.0);
    let half = ZEROS / cutoff;
    let n_out = resampled_len(x.len(), from, to);
    (0..n_out)
        .map(|i| {
            let t = i as f64 * ratio;
            let (lo, hi) = (
                (t - half).ceil().max(0.0) as usize,
                ((t + half).floor() as usize).min(x.len().saturating_sub(1)),
            );
            let mut acc = 0.0;
            for (k, &v) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let d = t - k as f64;
                let a = std::f64::consts::PI * cutoff * d;
                let sinc = if d.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                let win = 0.5 + 0.5 * (std::f64::consts::PI * d / half).cos();
                acc += v as f64 * cutoff * sinc * win;
            }
            acc as f32
        })
        .collect()
}

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
    dft: Vec<f32>,
    /// `[mel, bins]` HTK triangular filters, unnormalised.
    filters: Vec<f32>,
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

    /// Log-mel frames `[t, mel]` (invalid frames zeroed) and each frame's validity.
    pub fn extract(&self, wav: &[u8]) -> Result<(Vec<f32>, Vec<bool>), String> {
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
        let (b, f) = (self.bins, self.frame);
        let mut spec = vec![0.0f32; t * 2 * b];
        nn::matmul_nt(&frames, &self.dft, None, t, f, 2 * b, &mut spec);
        let mag: Vec<f32> = spec
            .chunks_exact(2 * b)
            .flat_map(|r| (0..b).map(move |k| (r[k] * r[k] + r[b + k] * r[b + k]).sqrt()))
            .collect();
        let mut mel = vec![0.0f32; t * self.mel];
        nn::matmul_nt(&mag, &self.filters, None, t, b, self.mel, &mut mel);
        for (row, &ok) in mel.chunks_exact_mut(self.mel).zip(&mask) {
            row.iter_mut()
                .for_each(|v| *v = if ok { (*v + self.floor).ln() } else { 0.0 });
        }
        Ok((mel, mask))
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
        })
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
    // ponytail: the conformer's 1e10 gradient-clipping clamps are omitted (no-ops on finite f32 activations).
    pub fn encode(&self, wav: &[u8]) -> Result<(Vec<f32>, usize), String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend(16u32.to_le_bytes());
        b.extend(format.to_le_bytes());
        b.extend(channels.to_le_bytes());
        b.extend(rate.to_le_bytes());
        b.extend((rate * (bits / 8 * channels) as u32).to_le_bytes());
        b.extend((bits / 8 * channels).to_le_bytes());
        b.extend(bits.to_le_bytes());
        b.extend(b"LIST\x02\0\0\0xy"); // unknown chunk, skipped
        b.extend(b"data");
        b.extend((data.len() as u32).to_le_bytes());
        b.extend(data);
        b
    }

    #[test]
    fn wav_decodes_pcm_float_and_downmixes() {
        let pcm: Vec<u8> = [16384i16, -16384, 32767, 0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let (x, rate) = read_wav(&wav(1, 16, 2, 22050, &pcm)).unwrap();
        assert_eq!(rate, 22050);
        assert_eq!(x.len(), 2);
        assert!(x[0].abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-3);
        let f: Vec<u8> = [0.25f32, -1.0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(
            read_wav(&wav(3, 32, 1, 16000, &f)).unwrap().0,
            vec![0.25, -1.0]
        );
        assert!(read_wav(&wav(2, 4, 1, 16000, &[0; 4])).is_err()); // ADPCM
        assert!(read_wav(b"OggS....").is_err());
        // Absurd declared rates are rejected before anything is sized from them.
        assert!(read_wav(&wav(1, 16, 1, 1, &pcm)).is_err());
        assert!(read_wav(&wav(1, 16, 1, 10_000_000, &pcm)).is_err());
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
    fn resample_keeps_a_tone() {
        let tone: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let y = resample(&tone, 48000, 16000);
        assert_eq!(y.len(), 16000);
        for (i, &v) in y.iter().enumerate().skip(100).take(15800) {
            let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
            assert!((v - want).abs() < 2e-2, "sample {i}: {v} vs {want}");
        }
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
}
