//! Draft models for the scheduler's speculative decoding ([`crate::scheduler::spec`]): block
//! drafters conditioned on target hidden states (DFlash, DFlash2, DSpark; see
//! `qwen3_5/draft.rs`) and plain causal LMs sharing the target's tokenizer.

use crate::hardware::{Device, PlacementMode, QuantMode};
use crate::model::graph_lm::{GraphLm, LayerKVCache};
use crate::model::hf_loader::{load_causal_lm, load_json_tokenizer};
use crate::model::hub::ModelArtifacts;
use crate::model::loaders::gguf;
use crate::model::native::{load_native, NativeLm, SeqState};
use crate::model::weights::Weights;
use crate::scheduler::spec::{DraftState, Drafter};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

/// Config `architectures` of hidden-state-conditioned drafters (they need a matching target).
pub use crate::model::qwen3_5::DRAFT_ARCHITECTURES as BLOCK_ARCHITECTURES;

fn config(a: &ModelArtifacts) -> Result<Value, String> {
    let p = a
        .config_path
        .as_deref()
        .ok_or_else(|| format!("{}: no config.json", a.model_id))?;
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.display()))
}

/// The target a drafter repo pairs with (`None` when `cfg` is no hidden-state drafter and the
/// repo is no llama.cpp `dflash` GGUF sidecar, `gguf_arch`): the verifier its config names, a
/// `base_model:` hub tag, else the repo name minus its drafter suffix
/// (`z-lab/Qwen3-4B-DFlash-b16` -> `Qwen3-4B`).
pub fn draft_target(
    id: &str,
    cfg: &Value,
    tags: &[&str],
    gguf_arch: Option<&str>,
) -> Option<String> {
    if gguf_arch != Some("dflash")
        && !BLOCK_ARCHITECTURES.contains(&cfg["architectures"][0].as_str()?)
    {
        return None;
    }
    if let Some(v) = cfg["speculators_config"]["verifier"]["name_or_path"].as_str() {
        return Some(v.to_string());
    }
    let tagged = tags
        .iter()
        .find_map(|t| t.strip_prefix("base_model:").filter(|r| !r.contains(':')));
    if let Some(t) = tagged {
        return Some(t.to_string());
    }
    let name = id.rsplit('/').next().unwrap_or(id);
    let lower = name.to_ascii_lowercase();
    let cut = ["-dflash", "-dspark", "-speculator", ".dflash", ".dspark"]
        .iter()
        .filter_map(|s| lower.find(s))
        .min()
        .unwrap_or(name.len());
    Some(name[..cut].to_string())
}

/// Whether drafter target `target` (from [`draft_target`]) is the model `active`: repo names
/// compared without org (verifier paths may be local directories) or a `-GGUF` suffix, and with
/// an explicit weight file (`org/repo/x-Q4_K_M.gguf`) reduced to its repo, case-insensitively.
pub fn is_draft_for(target: &str, active: &str) -> bool {
    let name = |s: &str| {
        let repo = crate::model::hub::split_weight_file(s).0;
        let n = repo
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(repo);
        let n = n.split('@').next().unwrap_or(n).to_ascii_lowercase();
        n.trim_end_matches("-gguf")
            .trim_end_matches("_gguf")
            .to_string()
    };
    name(target) == name(active)
}

/// Loads `draft` as a drafter for the loaded target `target` (same artifacts it was loaded from),
/// computing on the target's backend `mode`.
pub fn load_drafter(
    draft: &ModelArtifacts,
    target: &ModelArtifacts,
    mode: PlacementMode,
) -> Result<Arc<dyn Drafter>, String> {
    let dw = draft
        .weights_path
        .as_deref()
        .ok_or_else(|| format!("{}: no weights downloaded", draft.model_id))?;
    let tw = target
        .weights_path
        .as_deref()
        .ok_or_else(|| format!("{}: no weights", target.model_id))?;
    let is_gguf = |p: &Path| p.extension().is_some_and(|e| e == "gguf");
    // llama.cpp `dflash` sidecars carry their config in GGUF metadata and no config.json.
    let mut dck = None;
    let sidecar = is_gguf(dw).then(|| gguf::architecture(dw)).flatten();
    let dcfg = match sidecar.filter(|a| a.contains("dflash")) {
        Some(_) => {
            let (m, data) = gguf::open(dw)?;
            let (ck, cfg) = crate::model::qwen3_5::gguf_drafter(&m, &data)
                .map_err(|e| format!("{}: {e}", dw.display()))?;
            dck = Some(ck);
            cfg
        }
        None => config(draft)?,
    };
    let arch = dcfg["architectures"][0].as_str().unwrap_or("");
    let block = BLOCK_ARCHITECTURES.contains(&arch);
    // A native GGUF target brings its config and shared embeddings from the file (hidden-state
    // drafters only; plain drafters only need the config.json).
    let (tcfg, mut tck) = match is_gguf(tw) && block {
        true => {
            let (cfg, ck) = crate::model::qwen3_5::gguf_target(tw)?;
            (cfg, Some(ck))
        }
        false => (config(target)?, None),
    };
    let tc = if tcfg["text_config"].is_object() {
        &tcfg["text_config"]
    } else {
        &tcfg
    };
    if block {
        crate::graph::avenues::family_gate(
            crate::graph::avenues::Family::DraftersBlock,
            crate::graph::avenues::resolved_backend(mode),
        )?;
        // Native models with taps: the qwen3_5 decoder family; GraphLm targets always have them.
        let mt = tcfg["model_type"].as_str().unwrap_or("");
        if crate::model::native::NATIVE_MODEL_TYPES.contains(&mt)
            && !matches!(mt, "qwen3_5" | "qwen3_vl" | "qwen3")
        {
            return Err(format!(
                "{arch} reads target hidden states; '{mt}' targets do not expose them yet"
            ));
        }
        let dck = match dck {
            Some(ck) => ck,
            None => Weights::open(
                &dw.parent()
                    .unwrap_or(Path::new("."))
                    .join("model.safetensors"),
            )?,
        };
        let tck = match tck.take() {
            Some(ck) => ck,
            None => Weights::open(tw)?,
        };
        return crate::model::qwen3_5::load_drafter(&draft.model_id, &dck, &dcfg, &tck, &tcfg);
    }
    let (lm, tok) = match load_native(draft, mode) {
        Some(m) => {
            let m = m?;
            let tok = m.tokenizer().clone();
            (Lm::Native(m), tok)
        }
        None => {
            let m = load_causal_lm(draft, QuantMode::Uncompressed)?;
            let tok = m.tokenizer.clone();
            if matches!(mode, PlacementMode::Pin(Device::Cpu)) {
                (Lm::GraphLm(m), tok)
            } else {
                let (g, w) = m.to_decode_graph();
                let prog = crate::graph::program::GraphProgram::compile(g, w, mode)
                    .map_err(|e| format!("drafter {}: {e}", draft.model_id))?;
                let (vocab, ctx) = (m.config.vocab_size, m.config.max_seq_len);
                let prog = Arc::new(parking_lot::Mutex::new(prog));
                (Lm::Program { prog, vocab, ctx }, tok)
            }
        }
    };
    let ttok = load_json_tokenizer(
        target
            .tokenizer_path
            .as_deref()
            .ok_or("target has no tokenizer.json")?,
    )?;
    let probe = "Speculative decoding: 12 × 34 = 408, naïve café 🙂\n<|im_end|>";
    if tok.encode(probe) != ttok.encode(probe) {
        return Err(format!(
            "{} tokenizes differently from {}; plain drafting needs the same tokenizer",
            draft.model_id, target.model_id
        ));
    }
    let d_vocab = match &lm {
        Lm::GraphLm(m) => m.config.vocab_size,
        Lm::Program { vocab, .. } => *vocab,
        Lm::Native(_) => {
            let dc = if dcfg["text_config"].is_object() {
                &dcfg["text_config"]
            } else {
                &dcfg
            };
            dc["vocab_size"].as_u64().unwrap_or(0) as usize
        }
    };
    let t_vocab = tc["vocab_size"].as_u64().unwrap_or(0) as usize;
    if d_vocab != t_vocab {
        return Err(format!("draft vocab {d_vocab} != target vocab {t_vocab}"));
    }
    Ok(Arc::new(Plain {
        label: format!("draft {}", draft.model_id),
        lm,
    }))
}

enum Lm {
    /// Host forward (`--backend cpu`).
    GraphLm(GraphLm),
    /// The decode graph of a `GraphLm` compiled for the backend.
    Program {
        prog: Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>,
        vocab: usize,
        ctx: usize,
    },
    Native(Arc<dyn NativeLm>),
}

/// A small causal LM with the target's tokenizer, proposing tokens autoregressively.
struct Plain {
    label: String,
    lm: Lm,
}

enum Kv {
    GraphLm(Vec<LayerKVCache>),
    /// The draft program's KV buffers.
    Program(crate::graph::StateStore),
    /// Built at the first propose.
    Native(Option<SeqState>),
}

/// The draft's sequence state and the ids it has run.
struct PlainState {
    kv: Kv,
    ids: Vec<u32>,
}

impl Drafter for Plain {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn tap_layers(&self) -> Vec<usize> {
        Vec::new()
    }

    fn new_state(&self) -> DraftState {
        let kv = match &self.lm {
            Lm::GraphLm(m) => Kv::GraphLm(
                (0..m.layers.len())
                    .map(|_| LayerKVCache::new(m.config.max_seq_len.max(1), m.config.hidden_dim))
                    .collect(),
            ),
            Lm::Program { .. } => Kv::Program(crate::graph::StateStore::new()),
            Lm::Native(_) => Kv::Native(None),
        };
        Box::new(PlainState {
            kv,
            ids: Vec::new(),
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        _hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        let st = st
            .downcast_mut::<PlainState>()
            .ok_or("draft state was not built by this drafter")?;
        let n = tokens.len();
        let ctx = match &self.lm {
            Lm::GraphLm(m) => m.config.max_seq_len,
            Lm::Program { ctx, .. } => *ctx,
            Lm::Native(m) => m.context_len(),
        };
        let k = k.min(ctx.saturating_sub(n));
        if k == 0 || n == 0 {
            return Ok(Vec::new());
        }
        // Rewind to the longest prefix the draft shares with the sequence (its last drafts were
        // rejected or its last call drafted past them), then catch up on committed tokens.
        let keep = st
            .ids
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(n - 1);
        let caught_up = &tokens[keep..n - 1];
        match (&self.lm, &mut st.kv) {
            (Lm::GraphLm(m), Kv::GraphLm(cache)) => {
                for c in cache.iter_mut() {
                    c.len = keep;
                }
                for (i, &t) in caught_up.iter().enumerate() {
                    m.forward_token_with_kv(t, keep + i, cache);
                }
            }
            (Lm::Program { prog, .. }, Kv::Program(state)) => {
                // Positions index the KV rows, so rewinding to `keep` is just writing from there.
                for (i, &t) in caught_up.iter().enumerate() {
                    run_program(prog, state, t, keep + i)?;
                }
            }
            (Lm::Native(m), Kv::Native(s)) => {
                if s.is_none() {
                    *s = Some(m.prepare(Vec::new(), &[])?.1);
                }
                let s = s.as_mut().expect("built above");
                // Drafts ran as consecutive checkpointing forwards, so any `keep` since the
                // previous call's first draft rewinds exactly.
                m.truncate(s, keep)?;
                if !caught_up.is_empty() {
                    m.forward(caught_up, keep, s)?;
                }
            }
            _ => return Err("draft state does not match the draft model".into()),
        }
        st.ids.truncate(keep);
        st.ids.extend_from_slice(caught_up);
        let mut out = Vec::with_capacity(k);
        let mut last = tokens[n - 1];
        for i in 0..k {
            let logits = match (&self.lm, &mut st.kv) {
                (Lm::GraphLm(m), Kv::GraphLm(cache)) => {
                    m.forward_token_with_kv(last, n - 1 + i, cache)
                }
                (Lm::Program { prog, .. }, Kv::Program(state)) => {
                    run_program(prog, state, last, n - 1 + i)?
                }
                (Lm::Native(m), Kv::Native(Some(s))) => m.forward_all(&[last], n - 1 + i, s)?,
                _ => return Err("draft state does not match the draft model".into()),
            };
            st.ids.push(last);
            last = pick(&logits);
            out.push(last);
        }
        Ok(out)
    }
}

/// One token at `pos` through a draft program whose KV buffers are `state`; its logits.
fn run_program(
    prog: &parking_lot::Mutex<crate::graph::program::GraphProgram>,
    state: &mut crate::graph::StateStore,
    token: u32,
    pos: usize,
) -> Result<Vec<f32>, String> {
    use crate::graph::Tensor;
    let mut p = prog.lock();
    std::mem::swap(&mut p.state, state);
    let out = p.run(vec![Tensor::U32(vec![token]), Tensor::U32(vec![pos as u32])]);
    std::mem::swap(&mut p.state, state);
    match out?.into_iter().next() {
        Some(Tensor::F32(v)) => Ok(v),
        _ => Err("the draft program did not return f32 logits".into()),
    }
}
