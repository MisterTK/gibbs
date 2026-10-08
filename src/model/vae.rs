//! Stable Diffusion AutoencoderKL on real weights: LDM `first_stage_model.*` names (SD GGUF) or
//! diffusers AutoencoderKL names (ModelScope safetensors), translated by [`diffusers_name`].

use crate::model::clip::Tensors;
use crate::model::image::ImageBuffer;
use crate::model::nn;

const SCALE: f32 = 0.18215;
const GN_GROUPS: usize = 32;
const GN_EPS: f32 = 1e-6;

/// Tensor names the VAE asks for are LDM names without the `first_stage_model.` prefix.
struct Src<'a> {
    t: &'a Tensors<'a>,
    /// `None`: LDM checkpoint. `Some(n)`: diffusers checkpoint with `n` decoder up_blocks.
    diffusers_up_levels: Option<usize>,
}

impl Src<'_> {
    fn name(&self, ldm: &str) -> Option<String> {
        let levels = match self.diffusers_up_levels {
            None => return Some(format!("first_stage_model.{ldm}")),
            Some(levels) => levels,
        };
        let name = diffusers_name(ldm, levels)?;
        if self.t.shape(&name).is_some() {
            return Some(name);
        }
        // diffusers >= 0.15 saves the mid-block attention as to_q/to_k/to_v/to_out.0.
        let new = [(".query.", ".to_q."), (".key.", ".to_k."), (".value.", ".to_v."), (".proj_attn.", ".to_out.0.")]
            .iter()
            .fold(name.clone(), |n, (old, new)| n.replace(old, new));
        Some(if self.t.shape(&new).is_some() { new } else { name })
    }

    fn has(&self, ldm: &str) -> bool {
        self.name(ldm).is_some_and(|n| self.t.shape(&n).is_some())
    }

    fn load(&self, ldm: &str) -> Result<Vec<f32>, String> {
        let name = self.name(ldm).ok_or_else(|| format!("VAE: no diffusers tensor for LDM name {ldm}"))?;
        self.t.load(&name).map_err(|e| format!("VAE: {e}"))
    }
}

/// LDM AutoencoderKL name (no `first_stage_model.` prefix) -> diffusers AutoencoderKL name, as stored in
/// ModelScope's `vae/diffusion_pytorch_model.fp16.safetensors` (mid-block attention uses the
/// `group_norm/query/key/value/proj_attn` names; its q/k/v/proj are `[C,C]` linear, LDM's `[C,C,1,1]`).
/// Decoder levels are reversed: LDM `up.i` is diffusers `up_blocks.(levels-1-i)`. `None` past the last level.
fn diffusers_name(ldm: &str, up_levels: usize) -> Option<String> {
    let mut out = Vec::new();
    let mut parts = ldm.split('.');
    while let Some(p) = parts.next() {
        out.push(match p {
            "up" => format!("up_blocks.{}", up_levels.checked_sub(parts.next()?.parse::<usize>().ok()? + 1)?),
            "down" => format!("down_blocks.{}", parts.next()?),
            "block" => format!("resnets.{}", parts.next()?),
            "mid" => "mid_block".into(),
            "block_1" => "resnets.0".into(),
            "block_2" => "resnets.1".into(),
            "attn_1" => "attentions.0".into(),
            "norm" => "group_norm".into(),
            "q" => "query".into(),
            "k" => "key".into(),
            "v" => "value".into(),
            "proj_out" => "proj_attn".into(),
            "upsample" => "upsamplers.0".into(),
            "downsample" => "downsamplers.0".into(),
            "nin_shortcut" => "conv_shortcut".into(),
            "norm_out" => "conv_norm_out".into(),
            other => other.into(),
        });
    }
    Some(out.join("."))
}

struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
}

impl Conv {
    fn load(r: &Src, p: &str, k: usize) -> Result<Self, String> {
        let w = r.load(&format!("{p}.weight"))?;
        let b = r.load(&format!("{p}.bias"))?;
        let c_out = b.len();
        if c_out == 0 || w.len() % (c_out * k * k) != 0 {
            return Err(format!("VAE: {p}.weight has {} elems, not [{c_out}, _, {k}, {k}]", w.len()));
        }
        Ok(Self { c_in: w.len() / (c_out * k * k), w, b, c_out, k })
    }

    /// Symmetric "same" padding conv, stride 1.
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        nn::conv2d(x, self.c_in, h, w, &self.w, Some(&self.b), self.c_out, self.k, 1, self.k / 2).0
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(r: &Src, p: &str) -> Result<Self, String> {
        Ok(Self { g: r.load(&format!("{p}.weight"))?, b: r.load(&format!("{p}.bias"))? })
    }
    fn fwd(&self, x: &[f32], hw: usize) -> Vec<f32> {
        let mut y = x.to_vec();
        nn::group_norm(&mut y, self.g.len(), hw, GN_GROUPS, GN_EPS, &self.g, &self.b);
        y
    }
}

struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    nin: Option<Conv>,
}

impl Resnet {
    fn load(r: &Src, p: &str) -> Result<Self, String> {
        let nin_name = format!("{p}.nin_shortcut");
        Ok(Self {
            norm1: Norm::load(r, &format!("{p}.norm1"))?,
            conv1: Conv::load(r, &format!("{p}.conv1"), 3)?,
            norm2: Norm::load(r, &format!("{p}.norm2"))?,
            conv2: Conv::load(r, &format!("{p}.conv2"), 3)?,
            nin: if r.has(&format!("{nin_name}.weight")) { Some(Conv::load(r, &nin_name, 1)?) } else { None },
        })
    }
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut t = self.norm1.fwd(x, h * w);
        nn::silu(&mut t);
        let t = self.conv1.fwd(&t, h, w);
        let mut t = self.norm2.fwd(&t, h * w);
        nn::silu(&mut t);
        let mut t = self.conv2.fwd(&t, h, w);
        let skip = match &self.nin {
            Some(c) => c.fwd(x, h, w),
            None => x.to_vec(),
        };
        t.iter_mut().zip(&skip).for_each(|(a, s)| *a += s);
        t
    }
}

struct Attn {
    norm: Norm,
    q: Conv,
    k: Conv,
    v: Conv,
    proj: Conv,
}

impl Attn {
    fn load(r: &Src, p: &str) -> Result<Self, String> {
        Ok(Self {
            norm: Norm::load(r, &format!("{p}.norm"))?,
            q: Conv::load(r, &format!("{p}.q"), 1)?,
            k: Conv::load(r, &format!("{p}.k"), 1)?,
            v: Conv::load(r, &format!("{p}.v"), 1)?,
            proj: Conv::load(r, &format!("{p}.proj_out"), 1)?,
        })
    }
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let hw = h * w;
        let c = self.q.c_out;
        let hn = self.norm.fwd(x, hw);
        // 1x1 conv == linear over channels; tokens are [hw, c].
        let tok = nn::transpose(&hn, c, hw);
        let lin = |cv: &Conv, t: &[f32]| nn::linear(t, hw, &cv.w, Some(&cv.b), cv.c_in, cv.c_out);
        let (q, k, v) = (lin(&self.q, &tok), lin(&self.k, &tok), lin(&self.v, &tok));
        let a = nn::attention(&q, &k, &v, hw, hw, 1, c, false);
        let o = nn::transpose(&lin(&self.proj, &a), hw, c);
        x.iter().zip(&o).map(|(a, b)| a + b).collect()
    }
}

struct Mid {
    b1: Resnet,
    attn: Attn,
    b2: Resnet,
}

impl Mid {
    fn load(r: &Src, p: &str) -> Result<Self, String> {
        Ok(Self {
            b1: Resnet::load(r, &format!("{p}.block_1"))?,
            attn: Attn::load(r, &format!("{p}.attn_1"))?,
            b2: Resnet::load(r, &format!("{p}.block_2"))?,
        })
    }
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let x = self.b1.fwd(x, h, w);
        let x = self.attn.fwd(&x, h, w);
        self.b2.fwd(&x, h, w)
    }
}

struct Level {
    blocks: Vec<Resnet>,
    resample: Option<Conv>,
}

fn load_levels(r: &Src, p: &str, resample: &str) -> Result<Vec<Level>, String> {
    let mut levels = Vec::new();
    for i in 0.. {
        let mut blocks = Vec::new();
        while r.has(&format!("{p}.{i}.block.{}.conv1.weight", blocks.len())) {
            blocks.push(Resnet::load(r, &format!("{p}.{i}.block.{}", blocks.len()))?);
        }
        if blocks.is_empty() {
            break;
        }
        let rs = format!("{p}.{i}.{resample}.conv");
        let resample = if r.has(&format!("{rs}.weight")) { Some(Conv::load(r, &rs, 3)?) } else { None };
        levels.push(Level { blocks, resample });
    }
    if levels.is_empty() {
        let first = format!("{p}.0.block.0.conv1.weight");
        return Err(format!("VAE: missing tensor {}", r.name(&first).unwrap_or(first)));
    }
    Ok(levels)
}

struct Tail {
    norm_out: Norm,
    conv_out: Conv,
}

impl Tail {
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut t = self.norm_out.fwd(x, h * w);
        nn::silu(&mut t);
        self.conv_out.fwd(&t, h, w)
    }
}

pub struct SdVae {
    /// Absent in SD3/FLUX VAEs (`use_quant_conv: false`).
    post_quant_conv: Option<Conv>,
    dec_in: Conv,
    dec_mid: Mid,
    dec_up: Vec<Level>,
    dec_tail: Tail,
    quant_conv: Option<Conv>,
    enc_in: Conv,
    enc_down: Vec<Level>,
    enc_mid: Mid,
    enc_tail: Tail,
    /// Model latents are `(z - shift) * scale`.
    scale: f32,
    shift: f32,
}

impl SdVae {
    /// SD GGUF: LDM names under `first_stage_model.`.
    pub fn from_ldm(t: &Tensors) -> Result<Self, String> {
        Self::load(&Src { t, diffusers_up_levels: None })
    }

    /// diffusers AutoencoderKL names (e.g. ModelScope `vae/diffusion_pytorch_model.fp16.safetensors`).
    pub fn from_diffusers(t: &Tensors) -> Result<Self, String> {
        let levels = t
            .names()
            .filter_map(|n| n.strip_prefix("decoder.up_blocks.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map(|i| i + 1)
            .ok_or("VAE: no decoder.up_blocks.* tensors")?;
        Self::load(&Src { t, diffusers_up_levels: Some(levels) })
    }

    fn load(r: &Src) -> Result<Self, String> {
        let opt_conv = |p: &str| -> Result<Option<Conv>, String> {
            if r.has(&format!("{p}.weight")) {
                Conv::load(r, p, 1).map(Some)
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            post_quant_conv: opt_conv("post_quant_conv")?,
            dec_in: Conv::load(r, "decoder.conv_in", 3)?,
            dec_mid: Mid::load(r, "decoder.mid")?,
            dec_up: load_levels(r, "decoder.up", "upsample")?,
            dec_tail: Tail {
                norm_out: Norm::load(r, "decoder.norm_out")?,
                conv_out: Conv::load(r, "decoder.conv_out", 3)?,
            },
            quant_conv: opt_conv("quant_conv")?,
            enc_in: Conv::load(r, "encoder.conv_in", 3)?,
            enc_down: load_levels(r, "encoder.down", "downsample")?,
            enc_mid: Mid::load(r, "encoder.mid")?,
            enc_tail: Tail {
                norm_out: Norm::load(r, "encoder.norm_out")?,
                conv_out: Conv::load(r, "encoder.conv_out", 3)?,
            },
            scale: SCALE,
            shift: 0.0,
        })
    }

    /// Latent normalisation of the checkpoint (`vae/config.json` `scaling_factor`, `shift_factor`);
    /// the default is SD 1.x/2.x: 0.18215, 0.
    pub fn with_scaling(mut self, scale: f32, shift: f32) -> Self {
        self.scale = scale;
        self.shift = shift;
        self
    }

    /// Latent channels (4 for SD 1.x/2.x, 16 for SD3/FLUX).
    pub fn latent_channels(&self) -> usize {
        self.dec_in.c_in
    }

    /// Scaled latents [C,h,w] -> RGB image of 8h x 8w.
    pub fn decode(&self, latents: &[f32], h: usize, w: usize) -> Result<ImageBuffer, String> {
        let c = self.latent_channels();
        if latents.len() != c * h * w {
            return Err(format!("VAE decode: latents len {} != {c}x{h}x{w}", latents.len()));
        }
        let z: Vec<f32> = latents.iter().map(|v| v / self.scale + self.shift).collect();
        let x = match &self.post_quant_conv {
            Some(p) => p.fwd(&z, h, w),
            None => z,
        };
        let mut x = self.dec_in.fwd(&x, h, w);
        let (mut h, mut w) = (h, w);
        x = self.dec_mid.fwd(&x, h, w);
        for level in self.dec_up.iter().rev() {
            for b in &level.blocks {
                x = b.fwd(&x, h, w);
            }
            if let Some(up) = &level.resample {
                x = nn::upsample_nearest2x(&x, up.c_in, h, w);
                h *= 2;
                w *= 2;
                x = up.fwd(&x, h, w);
            }
        }
        let x = self.dec_tail.fwd(&x, h, w);
        Ok(to_image(&x, h, w))
    }

    /// RGB image -> (scaled latents [C,H/8,W/8] of the posterior mean, H/8, W/8).
    pub fn encode(&self, image: &ImageBuffer) -> Result<(Vec<f32>, usize, usize), String> {
        let mut x = from_image(image)?;
        let (mut h, mut w) = (image.height, image.width);
        x = self.enc_in.fwd(&x, h, w);
        for level in &self.enc_down {
            for b in &level.blocks {
                x = b.fwd(&x, h, w);
            }
            if let Some(down) = &level.resample {
                let (p, ph, pw) = pad_right_bottom(&x, down.c_in, h, w);
                let (y, oh, ow) = nn::conv2d(&p, down.c_in, ph, pw, &down.w, Some(&down.b), down.c_out, 3, 2, 0);
                x = y;
                h = oh;
                w = ow;
            }
        }
        x = self.enc_mid.fwd(&x, h, w);
        let x = self.enc_tail.fwd(&x, h, w);
        let moments = match &self.quant_conv {
            Some(q) => q.fwd(&x, h, w),
            None => x,
        };
        let c = self.latent_channels();
        let mean = moments[..c * h * w].iter().map(|v| (v - self.shift) * self.scale).collect();
        Ok((mean, h, w))
    }
}

fn pad_right_bottom(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h + 1, w + 1);
    let mut out = vec![0.0; c * ph * pw];
    for ch in 0..c {
        for y in 0..h {
            let src = (ch * h + y) * w;
            let dst = (ch * ph + y) * pw;
            out[dst..dst + w].copy_from_slice(&x[src..src + w]);
        }
    }
    (out, ph, pw)
}

fn to_pixel(v: f32) -> u8 {
    (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8
}

fn to_image(x: &[f32], h: usize, w: usize) -> ImageBuffer {
    let mut img = ImageBuffer::new(w, h);
    let hw = h * w;
    for y in 0..h {
        for xi in 0..w {
            let i = y * w + xi;
            img.set_pixel(xi, y, to_pixel(x[i]), to_pixel(x[hw + i]), to_pixel(x[2 * hw + i]));
        }
    }
    img
}

fn from_image(image: &ImageBuffer) -> Result<Vec<f32>, String> {
    let (h, w) = (image.height, image.width);
    if h == 0 || w == 0 || h % 8 != 0 || w % 8 != 0 {
        return Err(format!("VAE encode: image {w}x{h} must be non-empty multiples of 8"));
    }
    let hw = h * w;
    let mut x = vec![0.0; 3 * hw];
    for y in 0..h {
        for xi in 0..w {
            let (r, g, b) = image.get_pixel(xi, y);
            let i = y * w + xi;
            x[i] = r as f32 / 127.5 - 1.0;
            x[hw + i] = g as f32 / 127.5 - 1.0;
            x[2 * hw + i] = b as f32 / 127.5 - 1.0;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_input_rejects_non_multiple_of_8() {
        assert!(from_image(&ImageBuffer::new(12, 16)).is_err());
        assert!(from_image(&ImageBuffer::new(16, 20)).is_err());
        assert_eq!(from_image(&ImageBuffer::new(16, 8)).unwrap().len(), 3 * 128);
    }

    #[test]
    fn pixel_mapping_clamps() {
        assert_eq!(to_pixel(-5.0), 0);
        assert_eq!(to_pixel(-1.0), 0);
        assert_eq!(to_pixel(0.0), 128);
        assert_eq!(to_pixel(1.0), 255);
        assert_eq!(to_pixel(7.0), 255);
    }

    #[test]
    fn pad_right_bottom_zero_fills() {
        let (p, ph, pw) = pad_right_bottom(&[1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        assert_eq!((ph, pw), (3, 3));
        assert_eq!(p, vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn ldm_to_diffusers_names() {
        let d = |n| diffusers_name(n, 4);
        assert_eq!(d("decoder.up.3.block.0.conv1.weight").unwrap(), "decoder.up_blocks.0.resnets.0.conv1.weight");
        assert_eq!(d("decoder.up.0.block.0.nin_shortcut.bias").unwrap(), "decoder.up_blocks.3.resnets.0.conv_shortcut.bias");
        assert_eq!(d("decoder.up.2.upsample.conv.weight").unwrap(), "decoder.up_blocks.1.upsamplers.0.conv.weight");
        assert_eq!(d("decoder.mid.block_2.norm1.weight").unwrap(), "decoder.mid_block.resnets.1.norm1.weight");
        assert_eq!(d("decoder.mid.attn_1.norm.bias").unwrap(), "decoder.mid_block.attentions.0.group_norm.bias");
        assert_eq!(d("decoder.mid.attn_1.proj_out.weight").unwrap(), "decoder.mid_block.attentions.0.proj_attn.weight");
        assert_eq!(d("decoder.mid.attn_1.q.weight").unwrap(), "decoder.mid_block.attentions.0.query.weight");
        assert_eq!(d("decoder.norm_out.weight").unwrap(), "decoder.conv_norm_out.weight");
        assert_eq!(d("encoder.down.1.downsample.conv.bias").unwrap(), "encoder.down_blocks.1.downsamplers.0.conv.bias");
        assert_eq!(d("post_quant_conv.weight").unwrap(), "post_quant_conv.weight");
        assert_eq!(d("decoder.up.4.block.0.conv1.weight"), None);
    }
}
