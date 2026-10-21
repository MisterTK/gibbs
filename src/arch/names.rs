//! Pattern rename rules: map checkpoint tensor names (GGUF, LDM, BFL, ...) to the canonical
//! (HF/diffusers) names specs use. A rule is `[from, to]`. In `from`, `{v}` captures a decimal
//! integer and `{v:*}` captures any non-empty text; `to` interpolates expressions over the captures
//! (`{i}`, `{i*3+1}`, `{rest}`). The first matching rule wins; unmatched names are already canonical.

use super::expr::{interp, walk, Env, Val};

#[derive(Debug, Clone)]
pub struct Rule {
    from: Vec<Part>,
    to: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Part {
    Lit(String),
    Int(String),
    Any(String),
}

/// Captures of a match, as an [`Env`] for the `to` template.
struct Caps<'a>(&'a [(String, Val)]);

impl Env for Caps<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        self.0
            .iter()
            .find(|(k, _)| k == root)
            .and_then(|(_, v)| walk(v.clone(), path))
    }
}

impl Rule {
    pub fn new(from: &str, to: &str) -> Result<Self, String> {
        let mut parts = Vec::new();
        let mut rest = from;
        while let Some(s) = rest.find('{') {
            if s > 0 {
                parts.push(Part::Lit(rest[..s].to_string()));
            }
            let e = rest[s..]
                .find('}')
                .ok_or_else(|| format!("rename `{from}`: unclosed `{{`"))?
                + s;
            let cap = &rest[s + 1..e];
            parts.push(match cap.strip_suffix(":*") {
                Some(n) => Part::Any(n.to_string()),
                None if cap.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') => {
                    Part::Int(cap.to_string())
                }
                None => {
                    return Err(format!(
                        "rename `{from}`: capture `{cap}` must be a name or `name:*`"
                    ))
                }
            });
            rest = &rest[e + 1..];
        }
        if !rest.is_empty() {
            parts.push(Part::Lit(rest.to_string()));
        }
        Ok(Self {
            from: parts,
            to: to.to_string(),
        })
    }

    /// The canonical name for `name`, when this rule matches it.
    pub fn apply(&self, name: &str) -> Option<String> {
        let mut caps = Vec::new();
        if !matches(&self.from, name, &mut caps) {
            return None;
        }
        interp(&self.to, &Caps(&caps)).ok()
    }

    /// The captures of `name` under this rule's `from` pattern; `None` when it does not match.
    pub fn captures(&self, name: &str) -> Option<Vec<(String, Val)>> {
        let mut caps = Vec::new();
        matches(&self.from, name, &mut caps).then_some(caps)
    }
}

fn matches(parts: &[Part], s: &str, caps: &mut Vec<(String, Val)>) -> bool {
    let Some((first, rest)) = parts.split_first() else {
        return s.is_empty();
    };
    match first {
        Part::Lit(l) => s
            .strip_prefix(l.as_str())
            .is_some_and(|t| matches(rest, t, caps)),
        Part::Int(n) => {
            let d = s.bytes().take_while(u8::is_ascii_digit).count();
            // Shortest digit run that lets the rest match (digits may be followed by digits in a literal).
            (1..=d).any(|k| {
                caps.push((n.clone(), Val::Num(s[..k].parse::<f64>().unwrap())));
                if matches(rest, &s[k..], caps) {
                    return true;
                }
                caps.pop();
                false
            })
        }
        Part::Any(n) => (1..=s.len()).filter(|&k| s.is_char_boundary(k)).any(|k| {
            caps.push((n.clone(), Val::Str(s[..k].to_string())));
            if matches(rest, &s[k..], caps) {
                return true;
            }
            caps.pop();
            false
        }),
    }
}

/// An ordered rule list.
#[derive(Debug, Clone, Default)]
pub struct Renamer(pub Vec<Rule>);

impl Renamer {
    pub fn from_pairs(pairs: &[(String, String)]) -> Result<Self, String> {
        Ok(Self(
            pairs
                .iter()
                .map(|(f, t)| Rule::new(f, t))
                .collect::<Result<_, _>>()?,
        ))
    }

    /// Canonical name of `name`: the first matching rule's output, else `name` unchanged.
    pub fn canonical(&self, name: &str) -> String {
        self.0
            .iter()
            .find_map(|r| r.apply(name))
            .unwrap_or_else(|| name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(pairs: &[(&str, &str)]) -> Renamer {
        Renamer::from_pairs(
            &pairs
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn captures_and_arithmetic() {
        let rn = r(&[
            (
                "blk.{i}.attn_q.weight",
                "model.layers.{i}.self_attn.q_proj.weight",
            ),
            ("down.{b}.res.{j}.w", "down_blocks.{b}.resnets.{j*3+1}.w"),
            ("model.text_model.{rest:*}", "model.{rest}"),
            ("token_embd.weight", "model.embed_tokens.weight"),
        ]);
        assert_eq!(
            rn.canonical("blk.12.attn_q.weight"),
            "model.layers.12.self_attn.q_proj.weight"
        );
        assert_eq!(rn.canonical("down.2.res.1.w"), "down_blocks.2.resnets.4.w");
        assert_eq!(
            rn.canonical("model.text_model.layers.0.mlp.up_proj.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            rn.canonical("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(rn.canonical("lm_head.weight"), "lm_head.weight");
        assert_eq!(rn.canonical("blk.x.attn_q.weight"), "blk.x.attn_q.weight");
    }

    #[test]
    fn round_trip_through_inverse_rules() {
        let fwd = r(&[(
            "blk.{i}.ffn_up.weight",
            "model.layers.{i}.mlp.up_proj.weight",
        )]);
        let back = r(&[(
            "model.layers.{i}.mlp.up_proj.weight",
            "blk.{i}.ffn_up.weight",
        )]);
        for n in ["blk.0.ffn_up.weight", "blk.31.ffn_up.weight"] {
            assert_eq!(back.canonical(&fwd.canonical(n)), n);
        }
    }

    #[test]
    fn digit_capture_before_digit_literal() {
        let rn = r(&[("conv{i}1.w", "c.{i}.w")]);
        assert_eq!(rn.canonical("conv121.w"), "c.12.w");
    }

    fn set(name: &str) -> Renamer {
        let reg = crate::arch::Registry::load(None).unwrap();
        Renamer::from_pairs(&reg.name_set(name).unwrap()).unwrap()
    }

    /// `ldm` inverts `model::unet::ldm_unet_name` (layers_per_block 2, SD 1.x up_blocks.0 without
    /// attentions and SDXL-style attention up blocks).
    #[test]
    fn ldm_set_inverts_unet_aliases() {
        let rn = set("ldm");
        for d in [
            "conv_in.weight",
            "time_embedding.linear_1.bias",
            "time_embedding.cond_proj.weight",
            "add_embedding.linear_2.weight",
            "down_blocks.0.resnets.1.conv1.weight",
            "down_blocks.1.resnets.0.conv_shortcut.bias",
            "down_blocks.1.attentions.1.proj_in.weight",
            "down_blocks.2.downsamplers.0.conv.weight",
            "mid_block.resnets.0.norm1.weight",
            "mid_block.resnets.1.time_emb_proj.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "up_blocks.0.resnets.2.norm2.bias",
            "up_blocks.0.upsamplers.0.conv.weight",
            "up_blocks.1.upsamplers.0.conv.bias",
            "up_blocks.3.attentions.2.proj_out.weight",
            "up_blocks.3.resnets.2.conv2.weight",
            "conv_norm_out.weight",
            "conv_out.bias",
        ] {
            let ldm = crate::model::unet::ldm_unet_name(d, 2, &|i| i > 0).unwrap();
            assert_eq!(rn.canonical(&format!("model.diffusion_model.{ldm}")), d, "{ldm}");
        }
    }

    #[test]
    fn openclip_set_maps_to_hf_clip() {
        let rn = set("openclip");
        for (o, h) in [
            ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
            ("ln_final.bias", "text_model.final_layer_norm.bias"),
            ("text_projection", "text_projection.weight"),
            ("transformer.resblocks.11.attn.in_proj_weight", "text_model.encoder.layers.11.self_attn.in_proj.weight"),
            ("transformer.resblocks.3.mlp.c_proj.bias", "text_model.encoder.layers.3.mlp.fc2.bias"),
        ] {
            assert_eq!(rn.canonical(o), h);
        }
    }
}
