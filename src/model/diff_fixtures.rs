//! Tiny synthetic diffusion checkpoints (LDM UNet / VAE names, LTX) and GPU-only fixtures, shared
//! by the lib tests, the goldens and `tests/cpu_free_matrix.rs`.

/// Deterministic uniform [-1, 1) stream.
pub(crate) struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

type Specs = Vec<(String, Vec<usize>)>;

fn lin(s: &mut Specs, p: &str, i: usize, o: usize, bias: bool) {
    s.push((format!("{p}.weight"), vec![o, i]));
    if bias {
        s.push((format!("{p}.bias"), vec![o]));
    }
}
fn conv(s: &mut Specs, p: &str, i: usize, o: usize, k: usize) {
    s.push((format!("{p}.weight"), vec![o, i, k, k]));
    s.push((format!("{p}.bias"), vec![o]));
}
fn nrm(s: &mut Specs, p: &str, c: usize) {
    s.push((format!("{p}.weight"), vec![c]));
    s.push((format!("{p}.bias"), vec![c]));
}
pub fn res(s: &mut Specs, p: &str, i: usize, o: usize, emb: usize) {
    nrm(s, &format!("{p}.in_layers.0"), i);
    conv(s, &format!("{p}.in_layers.2"), i, o, 3);
    lin(s, &format!("{p}.emb_layers.1"), emb, o, true);
    nrm(s, &format!("{p}.out_layers.0"), o);
    conv(s, &format!("{p}.out_layers.3"), o, o, 3);
    if i != o {
        conv(s, &format!("{p}.skip_connection"), i, o, 1);
    }
}
pub fn trans(s: &mut Specs, p: &str, c: usize, ctx: usize) {
    nrm(s, &format!("{p}.norm"), c);
    lin(s, &format!("{p}.proj_in"), c, c, true);
    let b = format!("{p}.transformer_blocks.0");
    for n in ["norm1", "norm2", "norm3"] {
        nrm(s, &format!("{b}.{n}"), c);
    }
    for (a, kv) in [("attn1", c), ("attn2", ctx)] {
        lin(s, &format!("{b}.{a}.to_q"), c, c, false);
        lin(s, &format!("{b}.{a}.to_k"), kv, c, false);
        lin(s, &format!("{b}.{a}.to_v"), kv, c, false);
        lin(s, &format!("{b}.{a}.to_out.0"), c, c, true);
    }
    lin(s, &format!("{b}.ff.net.0.proj"), c, 2 * 2 * c, true);
    lin(s, &format!("{b}.ff.net.2"), 2 * c, c, true);
    lin(s, &format!("{p}.proj_out"), c, c, true);
}

/// A 2-level SD VAE in LDM (`first_stage_model.`) names with `latent` channels.
pub fn vae_specs(latent: usize) -> Vec<(String, Vec<usize>)> {
    let mut s: Vec<(String, Vec<usize>)> = Vec::new();
    let conv = |s: &mut Vec<_>, p: &str, i: usize, o: usize, k: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![o, i, k, k]));
        s.push((format!("first_stage_model.{p}.bias"), vec![o]));
    };
    let norm = |s: &mut Vec<_>, p: &str, c: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![c]));
        s.push((format!("first_stage_model.{p}.bias"), vec![c]));
    };
    fn resnet(s: &mut Vec<(String, Vec<usize>)>, p: &str, i: usize, o: usize) {
        for (n, c) in [("norm1", i), ("norm2", o)] {
            s.push((format!("first_stage_model.{p}.{n}.weight"), vec![c]));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![c]));
        }
        for (n, ci, co, k) in [("conv1", i, o, 3), ("conv2", o, o, 3)] {
            s.push((format!("first_stage_model.{p}.{n}.weight"), vec![co, ci, k, k]));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![co]));
        }
        if i != o {
            s.push((format!("first_stage_model.{p}.nin_shortcut.weight"), vec![o, i, 1, 1]));
            s.push((format!("first_stage_model.{p}.nin_shortcut.bias"), vec![o]));
        }
    }
    let mid = |s: &mut Vec<(String, Vec<usize>)>, p: &str, c: usize| {
        resnet(s, &format!("{p}.mid.block_1"), c, c);
        resnet(s, &format!("{p}.mid.block_2"), c, c);
        s.push((format!("first_stage_model.{p}.mid.attn_1.norm.weight"), vec![c]));
        s.push((format!("first_stage_model.{p}.mid.attn_1.norm.bias"), vec![c]));
        for n in ["q", "k", "v", "proj_out"] {
            s.push((format!("first_stage_model.{p}.mid.attn_1.{n}.weight"), vec![c, c, 1, 1]));
            s.push((format!("first_stage_model.{p}.mid.attn_1.{n}.bias"), vec![c]));
        }
    };
    // encoder: 32 -> (down) 64; decoder mirrors it with 2 resnets per level.
    conv(&mut s, "encoder.conv_in", 3, 32, 3);
    resnet(&mut s, "encoder.down.0.block.0", 32, 32);
    conv(&mut s, "encoder.down.0.downsample.conv", 32, 32, 3);
    resnet(&mut s, "encoder.down.1.block.0", 32, 64);
    mid(&mut s, "encoder", 64);
    norm(&mut s, "encoder.norm_out", 64);
    conv(&mut s, "encoder.conv_out", 64, 2 * latent, 3);
    conv(&mut s, "quant_conv", 2 * latent, 2 * latent, 1);
    conv(&mut s, "post_quant_conv", latent, latent, 1);
    conv(&mut s, "decoder.conv_in", latent, 64, 3);
    mid(&mut s, "decoder", 64);
    resnet(&mut s, "decoder.up.1.block.0", 64, 64);
    resnet(&mut s, "decoder.up.1.block.1", 64, 64);
    conv(&mut s, "decoder.up.1.upsample.conv", 64, 64, 3);
    resnet(&mut s, "decoder.up.0.block.0", 64, 32);
    resnet(&mut s, "decoder.up.0.block.1", 32, 32);
    norm(&mut s, "decoder.norm_out", 32);
    conv(&mut s, "decoder.conv_out", 32, 3, 3);
    s
}


/// The SD VAE decoded as a Metal program from a tiny synthetic checkpoint; the caller arms the CPU
/// tripwire, so any CPU compute panics.
pub fn tiny_vae_on_gpu() -> Result<(), String> {
    use crate::arch::expr::Val;
    use crate::graph::Tensor;
    use crate::hardware::{Device, PlacementMode};
    let d = std::env::temp_dir().join(format!("gibbs-tiny-vae-{}", std::process::id()));
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    write_f32_safetensors(&d.join("diffusion_pytorch_model.safetensors"), &values(&vae_specs(4), 7))?;
    let de = crate::runtime::diffusion::Net::open(&d, "sd_vae", "decoder", PlacementMode::Pin(Device::Metal))?;
    let (h, w) = (6, 5);
    let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
    let img = de.run_one(&vars, &[("z", Tensor::F32(Lcg(3).vec(4 * h * w, 1.0)))], &[], None)?;
    std::fs::remove_dir_all(&d).ok();
    if img.len() != 3 * 2 * h * 2 * w {
        return Err(format!("sd vae: {} values", img.len()));
    }
    Ok(())
}

/// LDM UNet specs from a config: `mc` model channels, per-level multipliers, `nres` ResBlocks per level,
/// attention on the levels flagged in `attn`, cross-attention width `ctx`, feed-forward `ff_mult * c`.
/// (SD 1.5: `mc` 320, mult [1,2,4,4], nres 2, attn [t,t,t,f], ctx 768, ff_mult 4.)
pub fn ldm_unet_specs(mc: usize, mult: &[usize], nres: usize, attn: &[bool], ctx: usize, ff_mult: usize) -> Specs {
    let emb = 4 * mc;
    let mut s = Specs::new();
    let trans = |s: &mut Specs, p: &str, c: usize| {
        trans(s, p, c, ctx);
        // `trans` builds a 2c inner width; widen to ff_mult * c.
        let b = format!("{p}.transformer_blocks.0.ff.net");
        s.retain(|(n, _)| !n.starts_with(&b));
        lin(s, &format!("{b}.0.proj"), c, 2 * ff_mult * c, true);
        lin(s, &format!("{b}.2"), ff_mult * c, c, true);
    };
    conv(&mut s, "input_blocks.0.0", 4, mc, 3);
    let (mut ib, mut ch, mut skips) = (1, mc, vec![mc]);
    for (l, &m) in mult.iter().enumerate() {
        for _ in 0..nres {
            res(&mut s, &format!("input_blocks.{ib}.0"), ch, mc * m, emb);
            ch = mc * m;
            if attn[l] {
                trans(&mut s, &format!("input_blocks.{ib}.1"), ch);
            }
            skips.push(ch);
            ib += 1;
        }
        if l + 1 < mult.len() {
            conv(&mut s, &format!("input_blocks.{ib}.0.op"), ch, ch, 3);
            skips.push(ch);
            ib += 1;
        }
    }
    res(&mut s, "middle_block.0", ch, ch, emb);
    trans(&mut s, "middle_block.1", ch);
    res(&mut s, "middle_block.2", ch, ch, emb);
    let mut ob = 0;
    for (l, &m) in mult.iter().enumerate().rev() {
        for i in 0..=nres {
            let skip = skips.pop().unwrap();
            res(&mut s, &format!("output_blocks.{ob}.0"), ch + skip, mc * m, emb);
            ch = mc * m;
            let mut j = 1;
            if attn[l] {
                trans(&mut s, &format!("output_blocks.{ob}.1"), ch);
                j = 2;
            }
            if l > 0 && i == nres {
                conv(&mut s, &format!("output_blocks.{ob}.{j}.conv"), ch, ch, 3);
            }
            ob += 1;
        }
    }
    nrm(&mut s, "out.0", mc);
    conv(&mut s, "out.2", mc, 4, 3);
    lin(&mut s, "time_embed.0", mc, emb, true);
    lin(&mut s, "time_embed.2", emb, emb, true);
    s
}

/// SD AutoencoderKL (LDM names): `ch` base channels, per-level multipliers, 2 encoder / 3 decoder
/// ResBlocks per level, 4 latent channels.
pub fn ldm_vae_specs(ch: usize, mult: &[usize]) -> Specs {
    let p = |n: &str| format!("first_stage_model.{n}");
    let mut s = Specs::new();
    let mut resnet = |s: &mut Specs, n: &str, i: usize, o: usize| {
        for (k, c) in [("norm1", i), ("norm2", o)] {
            nrm(s, &p(&format!("{n}.{k}")), c);
        }
        conv(s, &p(&format!("{n}.conv1")), i, o, 3);
        conv(s, &p(&format!("{n}.conv2")), o, o, 3);
        if i != o {
            conv(s, &p(&format!("{n}.nin_shortcut")), i, o, 1);
        }
    };
    let top = ch * mult[mult.len() - 1];
    let mid = |s: &mut Specs, resnet: &mut dyn FnMut(&mut Specs, &str, usize, usize), side: &str| {
        resnet(s, &format!("{side}.mid.block_1"), top, top);
        resnet(s, &format!("{side}.mid.block_2"), top, top);
        nrm(s, &p(&format!("{side}.mid.attn_1.norm")), top);
        for n in ["q", "k", "v", "proj_out"] {
            conv(s, &p(&format!("{side}.mid.attn_1.{n}")), top, top, 1);
        }
    };
    conv(&mut s, &p("encoder.conv_in"), 3, ch, 3);
    let mut c = ch;
    for (l, &m) in mult.iter().enumerate() {
        for b in 0..2 {
            resnet(&mut s, &format!("encoder.down.{l}.block.{b}"), c, ch * m);
            c = ch * m;
        }
        if l + 1 < mult.len() {
            conv(&mut s, &p(&format!("encoder.down.{l}.downsample.conv")), c, c, 3);
        }
    }
    mid(&mut s, &mut resnet, "encoder");
    nrm(&mut s, &p("encoder.norm_out"), top);
    conv(&mut s, &p("encoder.conv_out"), top, 8, 3);
    conv(&mut s, &p("quant_conv"), 8, 8, 1);
    conv(&mut s, &p("post_quant_conv"), 4, 4, 1);
    conv(&mut s, &p("decoder.conv_in"), 4, top, 3);
    mid(&mut s, &mut resnet, "decoder");
    let mut c = top;
    for (l, &m) in mult.iter().enumerate().rev() {
        for b in 0..3 {
            resnet(&mut s, &format!("decoder.up.{l}.block.{b}"), c, ch * m);
            c = ch * m;
        }
        if l > 0 {
            conv(&mut s, &p(&format!("decoder.up.{l}.upsample.conv")), c, c, 3);
        }
    }
    nrm(&mut s, &p("decoder.norm_out"), c);
    conv(&mut s, &p("decoder.conv_out"), c, 3, 3);
    s
}

#[cfg(test)]
mod vae_fixture_tests {
    /// The VAE fixture runs (CPU decode here: the tripwire is not armed in the lib tests). The armed
    /// run lives in `tests/cpu_free_matrix.rs`.
    #[test]
    fn tiny_vae_fixtures_run() {
        super::tiny_vae_on_gpu().unwrap();
    }
}

/// `(name, shape, values)` tensors as an F32 safetensors file.
pub(crate) fn write_f32_safetensors(path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<(), String> {
    let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
    for (name, shape, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// Random values for `specs`: norm gains near 1, other rank-1 tensors near 0, matrices by fan-in.
pub(crate) fn values(specs: &Specs, seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut r = Lcg(seed);
    specs
        .iter()
        .map(|(n, s)| {
            let len: usize = s.iter().product();
            let fan: usize = s[1..].iter().product::<usize>().max(1);
            let v = match s.len() {
                1 if n.ends_with("weight") => (0..len).map(|_| 1.0 + 0.1 * r.next()).collect(),
                1 => (0..len).map(|_| 0.1 * r.next()).collect(),
                _ => (0..len).map(|_| r.next() / (fan as f32).sqrt()).collect(),
            };
            (n.clone(), s.clone(), v)
        })
        .collect()
}

/// A tiny LTX-Video transformer `dir/diffusion_pytorch_model.safetensors` (1 block, d 64, 128
/// latent channels, caption width `ti`).
pub(crate) fn write_tiny_ltx(dir: &std::path::Path, ti: usize) -> Result<(), String> {
    let d = 64;
    let mut s = Specs::new();
    for (p, o, i) in [("proj_in", d, 128), ("time_embed.emb.timestep_embedder.linear_1", d, 256),
        ("time_embed.emb.timestep_embedder.linear_2", d, d), ("time_embed.linear", 6 * d, d),
        ("caption_projection.linear_1", d, ti), ("caption_projection.linear_2", d, d), ("proj_out", 128, d)] {
        lin(&mut s, p, i, o, true);
    }
    let b = "transformer_blocks.0";
    for a in ["attn1", "attn2"] {
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            lin(&mut s, &format!("{b}.{a}.{n}"), d, d, true);
        }
        s.push((format!("{b}.{a}.norm_q.weight"), vec![d]));
        s.push((format!("{b}.{a}.norm_k.weight"), vec![d]));
    }
    lin(&mut s, &format!("{b}.ff.net.0.proj"), d, 4 * d, true);
    lin(&mut s, &format!("{b}.ff.net.2"), 4 * d, d, true);
    s.push((format!("{b}.scale_shift_table"), vec![6, d]));
    s.push(("scale_shift_table".into(), vec![2, d]));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(&dir.join("diffusion_pytorch_model.safetensors"), &values(&s, 3))?;
    Ok(())
}

/// A tiny LTX-Video VAE `dir/` (widths 8, one spatio-temporal stage, patch 2, `lat` latent
/// channels; decoder and encoder, diffusers names, `config.json`).
pub(crate) fn write_tiny_ltx_vae(dir: &std::path::Path, lat: usize) -> Result<(), String> {
    let (ch, p) = (8, 2);
    let mut s = Specs::new();
    let c3 = |s: &mut Specs, n: &str, ci: usize, co: usize, k: usize| {
        s.push((format!("{n}.weight"), vec![co, ci, k, k, k]));
        s.push((format!("{n}.bias"), vec![co]));
    };
    let res = |s: &mut Specs, b: &str| {
        c3(s, &format!("{b}.conv1.conv"), ch, ch, 3);
        c3(s, &format!("{b}.conv2.conv"), ch, ch, 3);
    };
    c3(&mut s, "decoder.conv_in.conv", lat, ch, 3);
    res(&mut s, "decoder.mid_block.resnets.0");
    for i in 0..4 {
        if i == 3 {
            c3(&mut s, "decoder.up_blocks.3.upsamplers.0.conv.conv", ch, 8 * ch, 3);
        }
        res(&mut s, &format!("decoder.up_blocks.{i}.resnets.0"));
    }
    c3(&mut s, "decoder.conv_out.conv", ch, 3 * p * p, 3);
    c3(&mut s, "encoder.conv_in.conv", 3 * p * p, ch, 3);
    for i in 0..4 {
        res(&mut s, &format!("encoder.down_blocks.{i}.resnets.0"));
        if i == 0 {
            c3(&mut s, "encoder.down_blocks.0.downsamplers.0.conv", ch, ch, 3);
        }
    }
    res(&mut s, "encoder.mid_block.resnets.0");
    c3(&mut s, "encoder.conv_out.conv", ch, lat + 1, 3);
    s.push(("latents_mean".into(), vec![lat]));
    s.push(("latents_std".into(), vec![lat]));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(&dir.join("diffusion_pytorch_model.safetensors"), &values(&s, 9))?;
    let cfg = serde_json::json!({"_class_name": "AutoencoderKLLTXVideo", "block_out_channels": [ch, ch, ch, ch],
        "layers_per_block": [1, 1, 1, 1, 1], "spatio_temporal_scaling": [true, false, false, false], "patch_size": p, "latent_channels": lat});
    std::fs::write(dir.join("config.json"), cfg.to_string()).map_err(|e| e.to_string())?;
    Ok(())
}

/// Tiny LTX-Video transformer (1 block, d 64), LTX VAE (decoder + encoder) and ModelScope UNet3D (widths [32, 32]) checkpoints,
/// each run once through its family spec's `denoiser` as a Metal program (two frames, a clean
/// first-frame span for LTX); the caller arms the CPU tripwire.
pub fn tiny_video_on_gpu() -> Result<(), String> {
    use crate::arch::expr::Val;
    use crate::graph::Tensor;
    use crate::hardware::{Device, PlacementMode};
    use crate::runtime::diffusion::Net;
    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let num = |v: usize| Val::Num(v as f64);

    let ti = 32;
    let dir = tmp.path().join("ltx");
    write_tiny_ltx(&dir, ti)?;
    let net = Net::open(&dir, "ltx", "denoiser", PlacementMode::Pin(Device::Metal))?;
    let (f, h, w, nt) = (2, 2, 2, 3);
    let vars = [("frames", num(f)), ("h", num(h)), ("w", num(w)), ("nt", num(nt)), ("clean", num(h * w))];
    let given = [("x", Tensor::F32(Lcg(4).vec(f * h * w * 128, 1.0))), ("context", Tensor::F32(Lcg(5).vec(nt * ti, 1.0)))];
    let v = net.run_one(&vars, &given, &[("t", 500.0)], None)?;
    if v.iter().any(|x| !x.is_finite()) {
        return Err("non-finite LTX output".into());
    }

    // LTX causal-3D VAE: widths [8, 8, 8, 8], one up/down stage, patch 2, 8 latents; decode a
    // 2-frame clip, encode one image.
    let (lat, p) = (8, 2);
    let dir = tmp.path().join("ltx_vae");
    write_tiny_ltx_vae(&dir, lat)?;
    let mode = PlacementMode::Pin(Device::Metal);
    let dec = Net::open(&dir, "ltx_vae", "decoder", mode)?;
    let (f, h, w) = (2, 2, 3);
    let vars = [("frames", num(f)), ("h", num(h)), ("w", num(w))];
    let rgb = dec.run_one(&vars, &[("z", Tensor::F32(Lcg(10).vec(f * lat * h * w, 1.0)))], &[], None)?;
    if rgb.len() != 3 * 3 * (2 * p * h) * (2 * p * w) || rgb.iter().any(|x| !x.is_finite()) {
        return Err(format!("LTX VAE decode: {} values", rgb.len()));
    }
    let enc = Net::open(&dir, "ltx_vae", "encoder", mode)?;
    let img = Lcg(11).vec(3 * (2 * p * h) * (2 * p * w), 1.0);
    let z = enc.run_one(&[("h", num(h)), ("w", num(w))], &[("image", Tensor::F32(img))], &[], None)?;
    if z.len() != lat * h * w || z.iter().any(|x| !x.is_finite()) {
        return Err(format!("LTX VAE encode: {} values", z.len()));
    }

    // UNet3D: CrossAttnDownBlock3D, DownBlock3D, mid, UpBlock3D, CrossAttnUpBlock3D.
    let (c, ctx) = (32, 24);
    let mut s = Specs::new();
    let temporal = |s: &mut Specs, p: &str, c: usize, kv: usize| {
        nrm(s, &format!("{p}.norm"), c);
        lin(s, &format!("{p}.proj_in"), c, c, true);
        lin(s, &format!("{p}.proj_out"), c, c, true);
        let t = format!("{p}.transformer_blocks.0");
        for n in ["norm1", "norm2", "norm3"] {
            nrm(s, &format!("{t}.{n}"), c);
        }
        for (a, kv) in [("attn1", c), ("attn2", kv)] {
            lin(s, &format!("{t}.{a}.to_q"), c, c, false);
            lin(s, &format!("{t}.{a}.to_k"), kv, c, false);
            lin(s, &format!("{t}.{a}.to_v"), kv, c, false);
            lin(s, &format!("{t}.{a}.to_out.0"), c, c, true);
        }
        lin(s, &format!("{t}.ff.net.0.proj"), c, 8 * c, true);
        lin(s, &format!("{t}.ff.net.2"), 4 * c, c, true);
    };
    let layer = |s: &mut Specs, base: &str, j: usize, ci: usize, attn: bool| {
        let r = format!("{base}.resnets.{j}");
        nrm(s, &format!("{r}.norm1"), ci);
        conv(s, &format!("{r}.conv1"), ci, c, 3);
        lin(s, &format!("{r}.time_emb_proj"), 4 * c, c, true);
        nrm(s, &format!("{r}.norm2"), c);
        conv(s, &format!("{r}.conv2"), c, c, 3);
        if ci != c {
            conv(s, &format!("{r}.conv_shortcut"), ci, c, 1);
        }
        for i in 1..=4 {
            nrm(s, &format!("{base}.temp_convs.{j}.conv{i}.0"), c);
            let cw = format!("{base}.temp_convs.{j}.conv{i}.{}", if i == 1 { 2 } else { 3 });
            s.push((format!("{cw}.weight"), vec![c, c, 3, 1, 1]));
            s.push((format!("{cw}.bias"), vec![c]));
        }
        if attn {
            temporal(s, &format!("{base}.attentions.{j}"), c, ctx);
            temporal(s, &format!("{base}.temp_attentions.{j}"), c, c);
        }
    };
    lin(&mut s, "time_embedding.linear_1", c, 4 * c, true);
    lin(&mut s, "time_embedding.linear_2", 4 * c, 4 * c, true);
    conv(&mut s, "conv_in", 4, c, 3);
    temporal(&mut s, "transformer_in", c, c);
    for (i, attn) in [(0, true), (1, false)] {
        for j in 0..2 {
            layer(&mut s, &format!("down_blocks.{i}"), j, c, attn);
        }
    }
    conv(&mut s, "down_blocks.0.downsamplers.0.conv", c, c, 3);
    layer(&mut s, "mid_block", 0, c, false);
    temporal(&mut s, "mid_block.attentions.0", c, ctx);
    temporal(&mut s, "mid_block.temp_attentions.0", c, c);
    layer(&mut s, "mid_block", 1, c, false);
    for (i, attn) in [(0, false), (1, true)] {
        for j in 0..3 {
            layer(&mut s, &format!("up_blocks.{i}"), j, 2 * c, attn);
        }
    }
    conv(&mut s, "up_blocks.0.upsamplers.0.conv", c, c, 3);
    nrm(&mut s, "conv_norm_out", c);
    conv(&mut s, "conv_out", c, 4, 3);
    let dir = tmp.path().join("unet3d");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(&dir.join("diffusion_pytorch_model.safetensors"), &values(&s, 6))?;
    let cfg = serde_json::json!({"_class_name": "UNet3DConditionModel", "in_channels": 4, "block_out_channels": [c, c],
        "cross_attention_dim": ctx, "attention_head_dim": 32, "layers_per_block": 2, "norm_num_groups": 32});
    std::fs::write(dir.join("config.json"), cfg.to_string()).map_err(|e| e.to_string())?;
    let net = Net::open(&dir, "modelscope_t2v", "denoiser", PlacementMode::Pin(Device::Metal))?;
    let (f, h, w, nt) = (2, 8, 8, 3);
    let vars = [("frames", num(f)), ("h", num(h)), ("w", num(w)), ("nt", num(nt))];
    let given = [("x", Tensor::F32(Lcg(7).vec(f * 4 * h * w, 1.0))), ("context", Tensor::F32(Lcg(8).vec(nt * ctx, 1.0)))];
    let e = net.run_one(&vars, &given, &[("t", 900.0)], None)?;
    if e.len() != f * 4 * h * w || e.iter().any(|x| !x.is_finite()) {
        return Err("bad UNet3D output".into());
    }
    Ok(())
}
