//! Apple Metal GPU acceleration for Qwen3 and HunYuan dense text decoders.

use super::text::{Ffn, Mixer, TextModel};
use super::Linear;
use crate::backend::metal::{
    dispatch_1d, new_shared_buffer, set_bytes, AttnParams, Gpu, GpuLinear, MetalKvCache, Uploaded,
    SHARED, TG,
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
    qkv: GpuLinear,
    q_norm: Buffer,
    k_norm: Buffer,
    o: GpuLinear,
    in_norm: Buffer,
    post_norm: Buffer,
    gate_up: GpuLinear,
    down: GpuLinear,
    inter: usize,
}

struct Scratch {
    hidden: Buffer,
    norm_x: Buffer,
    qkv: Buffer,
    attn: Buffer,
    proj: Buffer,
    gate_up: Buffer,
    mlp: Buffer,
    down: Buffer,
    logits: Buffer,
}

pub struct MetalQwenDecoder {
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
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub norm_after_rope: bool,
    pub inv_freq: Vec<f32>,
    pub max_pos: usize,
}

fn gpu_linear(gpu: &Gpu, lin: &Linear, up: &mut Uploaded) -> Result<GpuLinear, String> {
    if lin.parts.len() == 1 && lin.bias.is_none() {
        return gpu.linear_from_mat(&lin.parts[0], up);
    }
    if lin.parts.iter().all(|m| m.packed().is_none()) {
        let mut total = Vec::with_capacity(lin.rows * lin.cols);
        for p in &lin.parts {
            total.extend_from_slice(p.data());
        }
        let bytes = total.len() * 2;
        let buf =
            gpu.device
                .new_buffer_with_data(total.as_ptr() as *const c_void, bytes as u64, SHARED);
        up.copied += bytes;
        let bias_buf = match &lin.bias {
            Some(b) => Some(gpu.upload(b, up)?),
            None => None,
        };
        Ok(GpuLinear {
            w: buf,
            bias: bias_buf,
            rows: lin.rows,
            cols: lin.cols,
            format: crate::backend::metal::GpuWeightFormat::BF16,
        })
    } else {
        let mut total = Vec::with_capacity(lin.rows * lin.cols);
        for p in &lin.parts {
            total.extend(p.to_f32());
        }
        let bytes = total.len() * 4;
        let buf =
            gpu.device
                .new_buffer_with_data(total.as_ptr() as *const c_void, bytes as u64, SHARED);
        up.copied += bytes;
        let bias_buf = match &lin.bias {
            Some(b) => Some(gpu.upload(b, up)?),
            None => None,
        };
        Ok(GpuLinear {
            w: buf,
            bias: bias_buf,
            rows: lin.rows,
            cols: lin.cols,
            format: crate::backend::metal::GpuWeightFormat::F32,
        })
    }
}

impl MetalQwenDecoder {
    pub fn new(text: &Arc<TextModel>) -> Result<Self, String> {
        let gpu = Gpu::new()?;
        let mut up = Uploaded::default();

        let hidden = text.hidden;
        let vocab = text.vocab;
        let heads = text.heads;
        let kv_heads = text.kv_heads;
        let head_dim = text.head_dim;
        let eps = text.eps;
        let max_pos = text.max_pos;
        let norm_after_rope = text.norm_after_rope;

        let mut max_inter = 0;
        let mut gpu_layers = Vec::with_capacity(text.layers.len());

        for (i, l) in text.layers.iter().enumerate() {
            let Mixer::Full(attn) = &l.mixer else {
                return Err(format!(
                    "layer {i}: non-attention mixer is not supported on Metal"
                ));
            };
            let Ffn::Dense { gate_up, down } = &l.ffn else {
                return Err(format!(
                    "layer {i}: MoE is not supported on Metal Dense decoder"
                ));
            };

            let inter = down.cols;
            max_inter = max_inter.max(inter);

            let qkv = gpu_linear(&gpu, &attn.qkv, &mut up)?;
            let q_norm = gpu.upload(&attn.q_norm, &mut up)?;
            let k_norm = gpu.upload(&attn.k_norm, &mut up)?;
            let o = gpu_linear(&gpu, &attn.o, &mut up)?;

            let in_norm = gpu.upload(&l.in_norm, &mut up)?;
            let post_norm = gpu.upload(&l.post_norm, &mut up)?;

            let gate_up_lin = gpu_linear(&gpu, gate_up, &mut up)?;
            let down_lin = gpu_linear(&gpu, down, &mut up)?;

            gpu_layers.push(GpuLayer {
                qkv,
                q_norm,
                k_norm,
                o,
                in_norm,
                post_norm,
                gate_up: gate_up_lin,
                down: down_lin,
                inter,
            });
        }

        let lm_head_lin = text.lm_head.as_ref().unwrap_or(&text.embed);
        let lm_head = gpu_linear(&gpu, lm_head_lin, &mut up)?;
        let final_norm = gpu.upload(&text.norm, &mut up)?;

        let qkv_total_dim = (heads + 2 * kv_heads) * head_dim;

        let f = |n: usize| new_shared_buffer(&gpu.device, n * 4);
        let scratch = Scratch {
            hidden: f(hidden)?,
            norm_x: f(hidden)?,
            qkv: f(qkv_total_dim)?,
            attn: f(heads * head_dim)?,
            proj: f(hidden)?,
            gate_up: f(max_inter * 2)?,
            mlp: f(max_inter)?,
            down: f(hidden)?,
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
            kv_heads,
            head_dim,
            eps,
            norm_after_rope,
            inv_freq: text.inv_freq.clone(),
            max_pos,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn new_kv_cache(&self) -> Result<MetalKvCache, String> {
        self.gpu.new_kv_cache()
    }

    pub fn reserve(&self, kv: &mut MetalKvCache, need: usize) -> Result<(), String> {
        let width = self.kv_heads * self.head_dim;
        self.gpu
            .reserve_kv(kv, need, self.layers.len(), width, self.max_pos)
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

            let hidden_slice =
                unsafe { std::slice::from_raw_parts_mut(s.hidden.contents() as *mut f32, h) };
            self.text.embed(tok, hidden_slice)?;

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
        let hd = self.head_dim;
        let nh = self.heads;
        let kvh = self.kv_heads;

        let pairs = self.inv_freq.len();
        let mut cs = [0.0f32; 512];
        for d in 0..pairs {
            let theta = pos as f32 * self.inv_freq[d];
            cs[d] = theta.cos();
            cs[pairs + d] = theta.sin();
        }

        let q_len = nh * hd;
        let k_len = kvh * hd;
        let kv_width = kvh * hd;
        let kv_layer_bytes = (kv.cap * kv_width * 4) as u64;

        for (li, l) in self.layers.iter().enumerate() {
            // 1. in_norm
            g.norm(
                enc, &s.hidden, None, &l.in_norm, None, &s.norm_x, false, h, eps,
            );

            // 2. QKV projection
            g.gemv(enc, &l.qkv, &s.norm_x, &s.qkv);

            let q_off = 0u64;
            let k_off = (q_len * 4) as u64;
            let v_off = ((q_len + k_len) * 4) as u64;

            // 3. Per-head RMSNorm before RoPE (Qwen3) or after RoPE (HunYuan)
            if !self.norm_after_rope {
                enc.set_compute_pipeline_state(&g.rms_norm_heads);
                enc.set_buffer(0, Some(&s.qkv), q_off);
                enc.set_buffer(1, Some(&l.q_norm), 0);
                set_bytes(enc, 2, &(hd as u32));
                set_bytes(enc, 3, &eps);
                set_bytes(enc, 4, &1u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(nh as u64, 1, 1),
                    MTLSize::new(hd.min(256) as u64, 1, 1),
                );

                enc.set_compute_pipeline_state(&g.rms_norm_heads);
                enc.set_buffer(0, Some(&s.qkv), k_off);
                enc.set_buffer(1, Some(&l.k_norm), 0);
                set_bytes(enc, 2, &(hd as u32));
                set_bytes(enc, 3, &eps);
                set_bytes(enc, 4, &1u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(kvh as u64, 1, 1),
                    MTLSize::new(hd.min(256) as u64, 1, 1),
                );
            }

            // 4. RoPE
            enc.set_compute_pipeline_state(&g.rope_single);
            enc.set_buffer(0, Some(&s.qkv), q_off);
            enc.set_bytes(1, (2 * pairs * 4) as u64, cs.as_ptr() as *const c_void);
            set_bytes(
                enc,
                2,
                &RopeParams {
                    n_heads: nh as u32,
                    head_dim: hd as u32,
                    pairs: pairs as u32,
                    pair_stride: 1,
                    pair_gap: pairs as u32,
                    k_offset: 0,
                },
            );
            dispatch_1d(enc, nh * pairs);

            enc.set_compute_pipeline_state(&g.rope_single);
            enc.set_buffer(0, Some(&s.qkv), k_off);
            enc.set_bytes(1, (2 * pairs * 4) as u64, cs.as_ptr() as *const c_void);
            set_bytes(
                enc,
                2,
                &RopeParams {
                    n_heads: kvh as u32,
                    head_dim: hd as u32,
                    pairs: pairs as u32,
                    pair_stride: 1,
                    pair_gap: pairs as u32,
                    k_offset: 0,
                },
            );
            dispatch_1d(enc, kvh * pairs);

            if self.norm_after_rope {
                enc.set_compute_pipeline_state(&g.rms_norm_heads);
                enc.set_buffer(0, Some(&s.qkv), q_off);
                enc.set_buffer(1, Some(&l.q_norm), 0);
                set_bytes(enc, 2, &(hd as u32));
                set_bytes(enc, 3, &eps);
                set_bytes(enc, 4, &1u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(nh as u64, 1, 1),
                    MTLSize::new(hd.min(256) as u64, 1, 1),
                );

                enc.set_compute_pipeline_state(&g.rms_norm_heads);
                enc.set_buffer(0, Some(&s.qkv), k_off);
                enc.set_buffer(1, Some(&l.k_norm), 0);
                set_bytes(enc, 2, &(hd as u32));
                set_bytes(enc, 3, &eps);
                set_bytes(enc, 4, &1u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(kvh as u64, 1, 1),
                    MTLSize::new(hd.min(256) as u64, 1, 1),
                );
            }

            // 5. Append K and V to cache
            let off = li as u64 * kv_layer_bytes;
            let cur_width = kv_width as u32;
            enc.set_compute_pipeline_state(&g.kv_append_heads);
            enc.set_buffer(0, Some(&s.qkv), k_off);
            enc.set_buffer(1, Some(&s.qkv), v_off);
            enc.set_buffer(2, Some(&kv.k), off);
            enc.set_buffer(3, Some(&kv.v), off);
            set_bytes(enc, 4, &cur_width);
            set_bytes(enc, 5, &(pos as u32));
            dispatch_1d(enc, kv_width);

            // 6. GQA attention
            let group = (nh / kvh) as u32;
            enc.set_compute_pipeline_state(&g.attn);
            enc.set_buffer(0, Some(&s.qkv), q_off);
            enc.set_buffer(1, Some(&kv.k), off);
            enc.set_buffer(2, Some(&kv.v), off);
            enc.set_buffer(3, Some(&s.attn), 0);
            set_bytes(
                enc,
                4,
                &AttnParams {
                    n_heads: nh as u32,
                    head_dim: hd as u32,
                    stride: cur_width,
                    len: (pos + 1) as u32,
                    group,
                    scale: 1.0 / (hd as f32).sqrt(),
                    window: 0,
                    softcap: 0.0,
                },
            );
            enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(TG, 1, 1));

            // 7. Output projection & residual
            g.gemv(enc, &l.o, &s.attn, &s.proj);
            g.add(enc, &s.hidden, &s.proj, h);

            // 8. Post norm & FFN
            g.norm(
                enc,
                &s.hidden,
                None,
                &l.post_norm,
                None,
                &s.norm_x,
                false,
                h,
                eps,
            );
            g.gemv(enc, &l.gate_up, &s.norm_x, &s.gate_up);

            let inter = l.inter;
            enc.set_compute_pipeline_state(&g.swiglu);
            enc.set_buffer(0, Some(&s.gate_up), 0);
            enc.set_buffer(1, Some(&s.mlp), 0);
            set_bytes(enc, 2, &(inter as u32));
            dispatch_1d(enc, inter);

            g.gemv(enc, &l.down, &s.mlp, &s.down);
            g.add(enc, &s.hidden, &s.down, h);
        }

        // 9. Final norm & LM head
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen3_5::text::Arch;
    use crate::model::weights::Weights;

    #[test]
    fn test_qwen3_metal_parity() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let hidden = 64;
        let vocab = 128;
        let layers = 2;
        let heads = 4;
        let hd = 16;
        let kv_heads = 2;
        let inter = 128;

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "intermediate_size": inter,
            "max_position_embeddings": 64,
            "rope_scaling": {
                "rope_theta": 10000.0
            }
        });

        let mut specs = Vec::new();
        specs.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
        specs.push(("model.norm.weight".into(), vec![hidden]));
        specs.push(("lm_head.weight".into(), vec![vocab, hidden]));
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            specs.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
            specs.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
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
            specs.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]));
        }

        let shapes: std::collections::HashMap<String, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product::<usize>()))
            .collect();

        let weights = Weights::from_f32(specs, |name| {
            let n: usize = name
                .bytes()
                .enumerate()
                .map(|(i, b)| (i + 1) * b as usize)
                .sum();
            let count = shapes[name];
            Ok((0..count)
                .map(|i| (((i * 17 + n * 11) % 37) as f32 - 18.0) / 25.0)
                .collect())
        })
        .unwrap();

        let text_model = Arc::new(TextModel::load(&weights, &cfg, Arch::Qwen3).unwrap());
        let metal_decoder = MetalQwenDecoder::new(&text_model).unwrap();

        let mut cpu_state = text_model.new_state();
        let mut metal_cache = metal_decoder.new_kv_cache().unwrap();

        let tokens = vec![7u32, 23, 55];
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; hidden];
            text_model.embed(tok, &mut h).unwrap();
            let pos_arr = [[pos as u32; 3]];
            let cpu_hidden = text_model.forward_rows(
                h,
                &pos_arr,
                pos,
                &mut cpu_state,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let cpu_logits = text_model.logits(&cpu_hidden, 1);
            let metal_logits = metal_decoder
                .forward_tokens(&[tok], pos, &mut metal_cache)
                .unwrap();

            assert_eq!(cpu_logits.len(), metal_logits.len());
            for (c, m) in cpu_logits.iter().zip(&metal_logits) {
                assert!(
                    (c - m).abs() < 5e-2,
                    "Qwen3 CPU {c} vs Metal {m} diverged at pos {pos}"
                );
            }
        }
    }

    #[test]
    fn test_hunyuan_metal_parity() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let hidden = 64;
        let vocab = 128;
        let layers = 2;
        let heads = 4;
        let hd = 16;
        let kv_heads = 2;
        let inter = 128;

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "intermediate_size": inter,
            "max_position_embeddings": 64,
            "model_type": "hunyuan_v1_dense",
            "rope_scaling": {
                "rope_theta": 10000.0
            }
        });

        let mut specs = Vec::new();
        specs.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
        specs.push(("model.norm.weight".into(), vec![hidden]));
        specs.push(("lm_head.weight".into(), vec![vocab, hidden]));
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            specs.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
            specs.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
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
            specs.push((format!("{p}.self_attn.query_layernorm.weight"), vec![hd]));
            specs.push((format!("{p}.self_attn.key_layernorm.weight"), vec![hd]));
            specs.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]));
        }

        let shapes: std::collections::HashMap<String, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product::<usize>()))
            .collect();

        let weights = Weights::from_f32(specs, |name| {
            let n: usize = name
                .bytes()
                .enumerate()
                .map(|(i, b)| (i + 1) * b as usize)
                .sum();
            let count = shapes[name];
            Ok((0..count)
                .map(|i| (((i * 19 + n * 13) % 41) as f32 - 20.0) / 25.0)
                .collect())
        })
        .unwrap();

        let text_model = Arc::new(TextModel::load(&weights, &cfg, Arch::HunYuan).unwrap());
        let metal_decoder = MetalQwenDecoder::new(&text_model).unwrap();

        let mut cpu_state = text_model.new_state();
        let mut metal_cache = metal_decoder.new_kv_cache().unwrap();

        let tokens = vec![9u32, 31, 67];
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; hidden];
            text_model.embed(tok, &mut h).unwrap();
            let pos_arr = [[pos as u32; 3]];
            let cpu_hidden = text_model.forward_rows(
                h,
                &pos_arr,
                pos,
                &mut cpu_state,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let cpu_logits = text_model.logits(&cpu_hidden, 1);
            let metal_logits = metal_decoder
                .forward_tokens(&[tok], pos, &mut metal_cache)
                .unwrap();

            assert_eq!(cpu_logits.len(), metal_logits.len());
            for (c, m) in cpu_logits.iter().zip(&metal_logits) {
                assert!(
                    (c - m).abs() < 5e-2,
                    "HunYuan CPU {c} vs Metal {m} diverged at pos {pos}"
                );
            }
        }
    }
}
