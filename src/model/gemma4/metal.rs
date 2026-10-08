//! Apple Metal GPU acceleration for the Gemma 4 text decoder.

use super::text::TextModel;
use crate::backend::metal::{
    dispatch_1d, new_shared_buffer, set_bytes, AttnParams, Gpu, GpuLinear, MetalKvCache, Uploaded,
    TG,
};
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use parking_lot::Mutex;
use std::ffi::c_void;
use std::sync::Arc;

#[repr(C)]
struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    pairs: u32,
    pair_stride: u32,
    pair_gap: u32,
    k_offset: u32,
}

struct GpuLayer {
    sliding: bool,
    head_dim: usize,
    kv_heads: usize,
    kv_src: usize,
    q_proj: GpuLinear,
    q_norm: Buffer,
    k_proj: Option<GpuLinear>,
    v_proj: Option<GpuLinear>,
    k_norm: Option<Buffer>,
    o_proj: GpuLinear,
    gate_proj: GpuLinear,
    up_proj: GpuLinear,
    down_proj: GpuLinear,
    input_ln: Buffer,
    post_attn_ln: Buffer,
    pre_ff_ln: Buffer,
    post_ff_ln: Buffer,
    ple: Option<(GpuLinear, GpuLinear, Buffer)>,
    scalar: f32,
}

struct Scratch {
    hidden: Buffer,
    norm_x: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    proj: Buffer,
    gate: Buffer,
    up: Buffer,
    mlp: Buffer,
    down: Buffer,
    ple_row: Buffer,
    ple_g: Buffer,
    ple_p: Buffer,
    logits: Buffer,
}

pub struct MetalGemma4Decoder {
    gpu: Gpu,
    device_name: String,
    text: Arc<TextModel>,
    final_norm: Buffer,
    lm_head: GpuLinear,
    layers: Vec<GpuLayer>,
    scratch: Mutex<Scratch>,
    pub hidden: usize,
    pub vocab: usize,
    pub heads: usize,
    pub max_head_dim: usize,
    pub max_kv_heads: usize,
    pub window: usize,
    pub eps: f32,
    pub softcap: Option<f32>,
    pub inv_freq_sliding: Vec<f32>,
    pub inv_freq_full: Vec<f32>,
    pub first_shared: usize,
    pub max_pos: usize,
    pub ple_dim: usize,
}

impl MetalGemma4Decoder {
    pub fn new(text: &Arc<TextModel>) -> Result<Self, String> {
        let gpu = Gpu::new()?;
        let mut up = Uploaded::default();

        let hidden = text.hidden;
        let vocab = text.vocab;
        let heads = text.heads;
        let max_pos = text.max_pos;
        let eps = text.eps;

        let mut max_head_dim = 0;
        let mut max_kv_heads = 0;
        let mut max_inter = 0;

        let mut gpu_layers = Vec::with_capacity(text.layers.len());
        for l in &text.layers {
            max_head_dim = max_head_dim.max(l.head_dim);
            max_kv_heads = max_kv_heads.max(l.kv_heads);
            max_inter = max_inter.max(l.gate.rows);

            let q_proj = gpu.linear_from_mat(&l.attn.q, &mut up)?;
            let q_norm = gpu.upload(&l.attn.q_norm, &mut up)?;
            let o_proj = gpu.linear_from_mat(&l.attn.o, &mut up)?;

            let (k_proj, v_proj, k_norm) = if let Some((kw, vw, kn)) = &l.attn.kv {
                let kp = Some(gpu.linear_from_mat(kw, &mut up)?);
                let vp = match vw {
                    Some(vw) => Some(gpu.linear_from_mat(vw, &mut up)?),
                    None => None,
                };
                let kn_buf = Some(gpu.upload(kn, &mut up)?);
                (kp, vp, kn_buf)
            } else {
                (None, None, None)
            };

            let gate_proj = gpu.linear_from_mat(&l.gate, &mut up)?;
            let up_proj = gpu.linear_from_mat(&l.up, &mut up)?;
            let down_proj = gpu.linear_from_mat(&l.down, &mut up)?;

            let input_ln = gpu.upload(&l.input_ln, &mut up)?;
            let post_attn_ln = gpu.upload(&l.post_attn_ln, &mut up)?;
            let pre_ff_ln = gpu.upload(&l.pre_ff_ln, &mut up)?;
            let post_ff_ln = gpu.upload(&l.post_ff_ln, &mut up)?;

            let ple = if let Some((gate, proj, norm)) = &l.ple {
                Some((
                    gpu.linear_from_mat(gate, &mut up)?,
                    gpu.linear_from_mat(proj, &mut up)?,
                    gpu.upload(norm, &mut up)?,
                ))
            } else {
                None
            };

            gpu_layers.push(GpuLayer {
                sliding: l.sliding,
                head_dim: l.head_dim,
                kv_heads: l.kv_heads,
                kv_src: l.kv_src,
                q_proj,
                q_norm,
                k_proj,
                v_proj,
                k_norm,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                input_ln,
                post_attn_ln,
                pre_ff_ln,
                post_ff_ln,
                ple,
                scalar: l.scalar,
            });
        }

        // Tied LM head uses unscaled embed
        let lm_head = gpu.linear_from_mat(&text.embed, &mut up)?;
        let final_norm = gpu.upload(&text.norm, &mut up)?;

        let ple_dim = text.ple.as_ref().map_or(0, |p| p.dim);

        let f = |n: usize| new_shared_buffer(&gpu.device, n * 4);
        let scratch = Scratch {
            hidden: f(hidden)?,
            norm_x: f(hidden)?,
            q: f(heads * max_head_dim)?,
            k: f(max_kv_heads * max_head_dim)?,
            v: f(max_kv_heads * max_head_dim)?,
            attn: f(heads * max_head_dim)?,
            proj: f(hidden)?,
            gate: f(max_inter)?,
            up: f(max_inter)?,
            mlp: f(max_inter)?,
            down: f(hidden)?,
            ple_row: f((text.layers.len() * ple_dim).max(1))?,
            ple_g: f(ple_dim.max(1))?,
            ple_p: f(hidden)?,
            logits: f(vocab)?,
        };

        Ok(Self {
            device_name: gpu.device_name().to_string(),
            gpu,
            text: text.clone(),
            final_norm,
            lm_head,
            layers: gpu_layers,
            scratch: Mutex::new(scratch),
            hidden,
            vocab,
            heads,
            max_head_dim,
            max_kv_heads,
            window: text.window,
            eps,
            softcap: text.softcap,
            inv_freq_sliding: text.inv_freq_sliding.clone(),
            inv_freq_full: text.inv_freq_full.clone(),
            first_shared: text.first_shared,
            max_pos,
            ple_dim,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn new_kv_cache(&self) -> Result<MetalKvCache, String> {
        self.gpu.new_kv_cache()
    }

    pub fn reserve(&self, kv: &mut MetalKvCache, need: usize) -> Result<(), String> {
        let width = self.max_kv_heads * self.max_head_dim;
        self.gpu
            .reserve_kv(kv, need, self.first_shared, width, self.max_pos)
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv: &mut MetalKvCache,
    ) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("forward_tokens: no tokens".into());
        }
        let need = start_pos + tokens.len();
        self.reserve(kv, need)?;

        let s = self.scratch.lock();
        let h = self.hidden;

        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let is_last = i + 1 == tokens.len();

            // 1. Embed directly into s.hidden shared memory from CPU (avoids 1.6GB table wrap)
            let hidden_slice =
                unsafe { std::slice::from_raw_parts_mut(s.hidden.contents() as *mut f32, h) };
            self.text.embed_row(tok, hidden_slice);

            // 2. Compute per-layer inputs (PLE) on CPU into s.ple_row shared memory
            if self.ple_dim > 0 {
                let ple_vals = self.text.per_layer_inputs(hidden_slice, &[tok]);
                let ple_slice = unsafe {
                    std::slice::from_raw_parts_mut(s.ple_row.contents() as *mut f32, ple_vals.len())
                };
                ple_slice.copy_from_slice(&ple_vals);
            }

            // 3. Dispatch GPU command encoder for this token step
            self.gpu.run(|enc| {
                self.encode_token(enc, &s, kv, pos, is_last);
            })?;
        }
        kv.len = need;

        let logits =
            unsafe { std::slice::from_raw_parts(s.logits.contents() as *const f32, self.vocab) }
                .to_vec();
        Ok(logits)
    }

    fn encode_token(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &Scratch,
        kv: &MetalKvCache,
        pos: usize,
        want_logits: bool,
    ) {
        let g = &self.gpu;
        let h = self.hidden;
        let eps = self.eps;
        let pd = self.ple_dim;

        // Precompute RoPE tables for this position
        let mut cs_s = [0.0f32; 512];
        let pairs_s = self.inv_freq_sliding.len();
        for d in 0..pairs_s {
            let theta = pos as f32 * self.inv_freq_sliding[d];
            cs_s[d] = theta.cos();
            cs_s[pairs_s + d] = theta.sin();
        }

        let mut cs_f = [0.0f32; 512];
        let pairs_f = self.inv_freq_full.len();
        for d in 0..pairs_f {
            let theta = pos as f32 * self.inv_freq_full[d];
            cs_f[d] = theta.cos();
            cs_f[pairs_f + d] = theta.sin();
        }

        let kv_width = self.max_kv_heads * self.max_head_dim;
        let kv_layer_bytes = (kv.cap * kv_width * 4) as u64;

        // Decoder layers
        for (li, l) in self.layers.iter().enumerate() {
            let (pairs, cs) = if l.sliding {
                (pairs_s, &cs_s)
            } else {
                (pairs_f, &cs_f)
            };

            // Input RMSNorm
            g.norm(
                enc,
                &s.hidden,
                None,
                &l.input_ln,
                None,
                &s.norm_x,
                false,
                h,
                eps,
            );

            // Q projection
            g.gemv(enc, &l.q_proj, &s.norm_x, &s.q);

            // Q per-head RMSNorm
            enc.set_compute_pipeline_state(&g.rms_norm_heads);
            enc.set_buffer(0, Some(&s.q), 0);
            enc.set_buffer(1, Some(&l.q_norm), 0);
            set_bytes(enc, 2, &(l.head_dim as u32));
            set_bytes(enc, 3, &eps);
            set_bytes(enc, 4, &1u32);
            enc.dispatch_thread_groups(
                MTLSize::new(self.heads as u64, 1, 1),
                MTLSize::new(l.head_dim.min(256) as u64, 1, 1),
            );

            // Q RoPE
            enc.set_compute_pipeline_state(&g.rope_single);
            enc.set_buffer(0, Some(&s.q), 0);
            enc.set_bytes(1, (2 * pairs * 4) as u64, cs.as_ptr() as *const c_void);
            set_bytes(
                enc,
                2,
                &RopeParams {
                    n_heads: self.heads as u32,
                    head_dim: l.head_dim as u32,
                    pairs: pairs as u32,
                    pair_stride: 1,
                    pair_gap: pairs as u32,
                    k_offset: 0,
                },
            );
            dispatch_1d(enc, self.heads * pairs);

            // KV projection & cache append (unless KV-shared)
            if li < self.first_shared {
                if let Some(kp) = &l.k_proj {
                    g.gemv(enc, kp, &s.norm_x, &s.k);
                    enc.set_compute_pipeline_state(&g.rms_norm_heads);
                    enc.set_buffer(0, Some(&s.k), 0);
                    enc.set_buffer(1, Some(l.k_norm.as_ref().unwrap()), 0);
                    set_bytes(enc, 2, &(l.head_dim as u32));
                    set_bytes(enc, 3, &eps);
                    set_bytes(enc, 4, &1u32);
                    enc.dispatch_thread_groups(
                        MTLSize::new(l.kv_heads as u64, 1, 1),
                        MTLSize::new(l.head_dim.min(256) as u64, 1, 1),
                    );

                    enc.set_compute_pipeline_state(&g.rope_single);
                    enc.set_buffer(0, Some(&s.k), 0);
                    enc.set_bytes(1, (2 * pairs * 4) as u64, cs.as_ptr() as *const c_void);
                    set_bytes(
                        enc,
                        2,
                        &RopeParams {
                            n_heads: l.kv_heads as u32,
                            head_dim: l.head_dim as u32,
                            pairs: pairs as u32,
                            pair_stride: 1,
                            pair_gap: pairs as u32,
                            k_offset: 0,
                        },
                    );
                    dispatch_1d(enc, l.kv_heads * pairs);
                }

                let v_proj_to_use = l.v_proj.as_ref().or(l.k_proj.as_ref());
                if let Some(vp) = v_proj_to_use {
                    g.gemv(enc, vp, &s.norm_x, &s.v);
                    enc.set_compute_pipeline_state(&g.rms_norm_heads);
                    enc.set_buffer(0, Some(&s.v), 0);
                    enc.set_buffer(1, Some(&g.dummy), 0);
                    set_bytes(enc, 2, &(l.head_dim as u32));
                    set_bytes(enc, 3, &eps);
                    set_bytes(enc, 4, &0u32);
                    enc.dispatch_thread_groups(
                        MTLSize::new(l.kv_heads as u64, 1, 1),
                        MTLSize::new(l.head_dim.min(256) as u64, 1, 1),
                    );
                }

                // Append to this layer's cache slot
                let off = li as u64 * kv_layer_bytes;
                let cur_width = (l.kv_heads * l.head_dim) as u32;
                enc.set_compute_pipeline_state(&g.kv_append_heads);
                enc.set_buffer(0, Some(&s.k), 0);
                enc.set_buffer(1, Some(&s.v), 0);
                enc.set_buffer(2, Some(&kv.k), off);
                enc.set_buffer(3, Some(&kv.v), off);
                set_bytes(enc, 4, &cur_width);
                set_bytes(enc, 5, &(pos as u32));
                dispatch_1d(enc, cur_width as usize);
            }

            // Attention
            let kv_src = l.kv_src;
            let off = kv_src as u64 * kv_layer_bytes;
            let cur_width = (l.kv_heads * l.head_dim) as u32;
            let group = (self.heads / l.kv_heads) as u32;
            let window = if l.sliding { self.window as u32 } else { 0u32 };

            enc.set_compute_pipeline_state(&g.attn);
            enc.set_buffer(0, Some(&s.q), 0);
            enc.set_buffer(1, Some(&kv.k), off);
            enc.set_buffer(2, Some(&kv.v), off);
            enc.set_buffer(3, Some(&s.attn), 0);
            set_bytes(
                enc,
                4,
                &AttnParams {
                    n_heads: self.heads as u32,
                    head_dim: l.head_dim as u32,
                    stride: cur_width,
                    len: (pos + 1) as u32,
                    group,
                    scale: 1.0,
                    window,
                    softcap: 0.0,
                },
            );
            enc.dispatch_thread_groups(
                MTLSize::new(self.heads as u64, 1, 1),
                MTLSize::new(TG, 1, 1),
            );

            // Out projection
            g.gemv(enc, &l.o_proj, &s.attn, &s.proj);
            g.norm(
                enc,
                &s.proj,
                None,
                &l.post_attn_ln,
                None,
                &s.proj,
                false,
                h,
                eps,
            );
            g.add(enc, &s.hidden, &s.proj, h);

            // Feed-forward block
            g.norm(
                enc,
                &s.hidden,
                None,
                &l.pre_ff_ln,
                None,
                &s.norm_x,
                false,
                h,
                eps,
            );
            g.gemv(enc, &l.gate_proj, &s.norm_x, &s.gate);
            g.gemv(enc, &l.up_proj, &s.norm_x, &s.up);

            let inter = l.down_proj.cols;
            enc.set_compute_pipeline_state(&g.geglu);
            enc.set_buffer(0, Some(&s.gate), 0);
            enc.set_buffer(1, Some(&s.up), 0);
            enc.set_buffer(2, Some(&s.mlp), 0);
            set_bytes(enc, 3, &(inter as u32));
            dispatch_1d(enc, inter);

            g.gemv(enc, &l.down_proj, &s.mlp, &s.down);
            g.norm(
                enc,
                &s.down,
                None,
                &l.post_ff_ln,
                None,
                &s.down,
                false,
                h,
                eps,
            );
            g.add(enc, &s.hidden, &s.down, h);

            // Per-layer embedding projection (PLE)
            if let Some((gate, proj, norm)) = &l.ple {
                g.gemv(enc, gate, &s.hidden, &s.ple_g);
                enc.set_compute_pipeline_state(&g.geglu);
                enc.set_buffer(0, Some(&s.ple_g), 0);
                let ple_off = (li * pd * 4) as u64;
                enc.set_buffer(1, Some(&s.ple_row), ple_off);
                enc.set_buffer(2, Some(&s.ple_g), 0);
                set_bytes(enc, 3, &(pd as u32));
                dispatch_1d(enc, pd);

                g.gemv(enc, proj, &s.ple_g, &s.ple_p);
                g.norm(enc, &s.ple_p, None, norm, None, &s.ple_p, false, h, eps);
                g.add(enc, &s.hidden, &s.ple_p, h);
            }

            // Layer scalar
            if (l.scalar - 1.0).abs() > 1e-6 {
                enc.set_compute_pipeline_state(&g.scale);
                enc.set_buffer(0, Some(&s.hidden), 0);
                set_bytes(enc, 1, &l.scalar);
                set_bytes(enc, 2, &(h as u32));
                dispatch_1d(enc, h);
            }
        }

        // 3. Final norm & LM head
        if want_logits {
            g.norm(
                enc,
                &s.hidden,
                None,
                &self.final_norm,
                None,
                &s.norm_x,
                false,
                h,
                eps,
            );
            g.gemv(enc, &self.lm_head, &s.norm_x, &s.logits);

            if let Some(cap) = self.softcap {
                enc.set_compute_pipeline_state(&g.softcap);
                enc.set_buffer(0, Some(&s.logits), 0);
                set_bytes(enc, 1, &cap);
                set_bytes(enc, 2, &(self.vocab as u32));
                dispatch_1d(enc, self.vocab);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::weights::Weights;

    #[test]
    fn test_gemma4_metal_parity() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let hidden = 64;
        let vocab = 128;
        let layers = 2;
        let heads = 4;
        let hd = 16;
        let kv_heads = 2;

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "sliding_window": 32,
            "max_position_embeddings": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
            }
        });

        let mut specs = Vec::new();
        specs.push((
            "model.language_model.embed_tokens.weight".into(),
            vec![vocab, hidden],
        ));
        specs.push(("model.language_model.norm.weight".into(), vec![hidden]));
        for i in 0..layers {
            let p = format!("model.language_model.layers.{i}");
            specs.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
            specs.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
            specs.push((
                format!("{p}.pre_feedforward_layernorm.weight"),
                vec![hidden],
            ));
            specs.push((
                format!("{p}.post_feedforward_layernorm.weight"),
                vec![hidden],
            ));
            specs.push((
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden, heads * hd],
            ));
            specs.push((format!("{p}.self_attn.q_norm.weight"), vec![hd]));
            specs.push((format!("{p}.self_attn.k_norm.weight"), vec![hd]));
            specs.push((
                format!("{p}.mlp.gate_proj.weight"),
                vec![hidden * 2, hidden],
            ));
            specs.push((format!("{p}.mlp.up_proj.weight"), vec![hidden * 2, hidden]));
            specs.push((
                format!("{p}.mlp.down_proj.weight"),
                vec![hidden, hidden * 2],
            ));
            specs.push((format!("{p}.layer_scalar"), vec![1]));
        }

        let weights = Weights::from_f32(specs, |name| {
            let n: usize = name.len();
            Ok((0..10000)
                .map(|i| (((i * 13 + n * 7) % 31) as f32 - 15.0) / 20.0)
                .take(if name.contains("scalar") {
                    1
                } else if name.contains("norm.weight")
                    && (name.contains("q_norm") || name.contains("k_norm"))
                {
                    hd
                } else if name.contains("norm.weight") || name.contains("layernorm") {
                    hidden
                } else if name.contains("embed") {
                    vocab * hidden
                } else if name.contains("q_proj") {
                    heads * hd * hidden
                } else if name.contains("k_proj") || name.contains("v_proj") {
                    kv_heads * hd * hidden
                } else if name.contains("o_proj") {
                    hidden * heads * hd
                } else if name.contains("down_proj") {
                    hidden * hidden * 2
                } else {
                    hidden * 2 * hidden
                })
                .collect())
        })
        .unwrap();

        let text_model = Arc::new(TextModel::load(&weights, &cfg).unwrap());
        let metal_decoder = MetalGemma4Decoder::new(&text_model).unwrap();

        let mut cpu_cache = text_model.new_cache();
        let mut metal_cache = metal_decoder.new_kv_cache().unwrap();

        let tokens = vec![5u32, 12, 42];
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut x = vec![0.0f32; hidden];
            text_model.embed_row(tok, &mut x);
            let cpu_logits = text_model.forward(x, &[tok], pos, &mut cpu_cache, &[]);
            let metal_logits = metal_decoder
                .forward_tokens(&[tok], pos, &mut metal_cache)
                .unwrap();

            assert_eq!(cpu_logits.len(), metal_logits.len());
            for (c, m) in cpu_logits.iter().zip(&metal_logits) {
                assert!(
                    (c - m).abs() < 5e-2,
                    "CPU {c} vs Metal {m} diverged at pos {pos}"
                );
            }
        }
    }
}
