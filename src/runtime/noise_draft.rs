//! Online forward-only `qlabs.sh/noise` block drafter + suffix-automaton lattice ([`OnlineNoiseDrafter`]),
//! with live-tunable [`DraftKnobs`] / [`LiveDraftKnobs`].
//!
//! Combines:
//! 1. [`SuffixLattice`]: zero-parameter n-gram suffix continuation table ($n \in \{1..=4\}$) with
//!    recency decay for high-acceptance token chains (code, JSON, tool calls, repeated identifiers).
//! 2. [`NoiseBlockHead`]: lightweight multi-step block drafter head trained online during inference
//!    from committed target tokens and tapped hidden states via forward-only Evolution Strategies
//!    (`qlabs.sh/noise`: structured orthogonal Walsh-Hadamard Rademacher perturbations, central
//!    differences, top-credit subspace selection, and ANVIL II variance preconditioning).

use crate::scheduler::spec::{DraftState, Drafter};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// Bottleneck dimension of [`NoiseBlockHead`] (power of two for exact Walsh-Hadamard transform).
pub const NOISE_BOTTLENECK_DIM: usize = 128;
/// Maximum speculative lookahead offsets maintained by [`NoiseBlockHead`].
pub const NOISE_MAX_K: usize = 15;
/// Shortlist vocabulary capacity tracked online by [`NoiseBlockHead`].
pub const NOISE_SHORTLIST_VOCAB: usize = 256;
/// Number of orthogonal Hadamard probe directions evaluated per online update step.
const HADAMARD_PROBES: usize = 8;
/// Number of top-credit probe directions retained when reconstructing the subspace gradient.
const TOP_CREDIT_DIRS: usize = 4;

/// Speculative drafting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DraftMode {
    /// Use the model's built-in drafter (e.g. MTP) when present, or online noise+suffix when attached.
    #[default]
    Auto,
    /// Disable speculative decoding (`k = 0`).
    Off,
    /// Use only the model's built-in / sidecar drafter (`MTP` / `DFlash`).
    Model,
    /// Use the online `qlabs.sh/noise` neural block head + suffix lattice drafter.
    Noise,
    /// Use only the zero-overhead suffix lattice drafter.
    Suffix,
}

impl DraftMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Model => "model",
            Self::Noise => "noise",
            Self::Suffix => "suffix",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Noise,
            Self::Noise => Self::Suffix,
            Self::Suffix => Self::Model,
            Self::Model => Self::Off,
            Self::Off => Self::Auto,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::Off => 1,
            Self::Model => 2,
            Self::Noise => 3,
            Self::Suffix => 4,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Off,
            2 => Self::Model,
            3 => Self::Noise,
            4 => Self::Suffix,
            _ => Self::Auto,
        }
    }
}

impl FromStr for DraftMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" | "none" | "0" | "false" => Ok(Self::Off),
            "model" | "mtp" | "dflash" => Ok(Self::Model),
            "noise" | "online" | "online-noise" | "qlabs" => Ok(Self::Noise),
            "suffix" | "ngram" | "lookup" => Ok(Self::Suffix),
            other => Err(format!(
                "unknown draft mode '{other}'; valid: auto, noise, suffix, model, off"
            )),
        }
    }
}

/// User-facing tuning knobs for speculative decoding and [`OnlineNoiseDrafter`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DraftKnobs {
    pub mode: DraftMode,
    /// Maximum draft tokens per step (`0..=15`, default `4`).
    pub max_k: usize,
    /// Minimum acceptance rate (`0.30..=0.95`, default `0.65`) before [`KTuner`] penalizes deeper `k`.
    pub target_acceptance: f32,
    /// Weight (`0.0..=1.0`, default `0.25`) on verify compute cost vs raw wall-clock tok/s in [`KTuner`].
    pub cost_weight: f32,
    /// Minimum suffix n-gram match length (`1..=4`, default `2`) before [`SuffixLattice`] proposes.
    pub suffix_min_match: usize,
    /// Online learning rate (`0.0..=0.05`, default `0.005`) for [`NoiseBlockHead`].
    pub noise_lr: f32,
    /// Hadamard perturbation scale $\sigma$ (`1e-4..=0.1`, default `0.01`).
    pub noise_sigma: f32,
}

impl Default for DraftKnobs {
    fn default() -> Self {
        Self {
            mode: DraftMode::Auto,
            max_k: 4,
            target_acceptance: 0.65,
            cost_weight: 0.25,
            suffix_min_match: 2,
            noise_lr: 0.005,
            noise_sigma: 0.01,
        }
    }
}

impl DraftKnobs {
    pub fn sanitized(mut self) -> Self {
        self.max_k = self.max_k.min(NOISE_MAX_K);
        self.target_acceptance = self.target_acceptance.clamp(0.10, 0.98);
        self.cost_weight = self.cost_weight.clamp(0.0, 1.0);
        self.suffix_min_match = self.suffix_min_match.clamp(1, 4);
        self.noise_lr = self.noise_lr.clamp(0.0, 0.10);
        self.noise_sigma = self.noise_sigma.clamp(1e-5, 0.25);
        self
    }
}

/// Lock-free atomic container for [`DraftKnobs`], shared across the scheduler, [`KTuner`], and
/// [`OnlineNoiseDrafter`] so live edits take effect on the very next decode step.
#[derive(Debug)]
pub struct LiveDraftKnobs {
    mode: AtomicU8,
    max_k: AtomicUsize,
    target_acceptance: AtomicU32,
    cost_weight: AtomicU32,
    suffix_min_match: AtomicUsize,
    noise_lr: AtomicU32,
    noise_sigma: AtomicU32,
}

impl Default for LiveDraftKnobs {
    fn default() -> Self {
        Self::new(DraftKnobs::default())
    }
}

impl LiveDraftKnobs {
    pub fn new(knobs: DraftKnobs) -> Self {
        let k = knobs.sanitized();
        Self {
            mode: AtomicU8::new(k.mode.to_u8()),
            max_k: AtomicUsize::new(k.max_k),
            target_acceptance: AtomicU32::new(k.target_acceptance.to_bits()),
            cost_weight: AtomicU32::new(k.cost_weight.to_bits()),
            suffix_min_match: AtomicUsize::new(k.suffix_min_match),
            noise_lr: AtomicU32::new(k.noise_lr.to_bits()),
            noise_sigma: AtomicU32::new(k.noise_sigma.to_bits()),
        }
    }

    pub fn snapshot(&self) -> DraftKnobs {
        DraftKnobs {
            mode: DraftMode::from_u8(self.mode.load(Ordering::Relaxed)),
            max_k: self.max_k.load(Ordering::Relaxed),
            target_acceptance: f32::from_bits(self.target_acceptance.load(Ordering::Relaxed)),
            cost_weight: f32::from_bits(self.cost_weight.load(Ordering::Relaxed)),
            suffix_min_match: self.suffix_min_match.load(Ordering::Relaxed),
            noise_lr: f32::from_bits(self.noise_lr.load(Ordering::Relaxed)),
            noise_sigma: f32::from_bits(self.noise_sigma.load(Ordering::Relaxed)),
        }
    }

    pub fn store(&self, knobs: DraftKnobs) {
        let k = knobs.sanitized();
        self.mode.store(k.mode.to_u8(), Ordering::Relaxed);
        self.max_k.store(k.max_k, Ordering::Relaxed);
        self.target_acceptance
            .store(k.target_acceptance.to_bits(), Ordering::Relaxed);
        self.cost_weight
            .store(k.cost_weight.to_bits(), Ordering::Relaxed);
        self.suffix_min_match
            .store(k.suffix_min_match, Ordering::Relaxed);
        self.noise_lr.store(k.noise_lr.to_bits(), Ordering::Relaxed);
        self.noise_sigma
            .store(k.noise_sigma.to_bits(), Ordering::Relaxed);
    }
}

// ---- Structured Orthogonal Hadamard Perturbations (`qlabs.sh/noise`) --------------------------

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// In-place normalized Fast Walsh-Hadamard Transform ($H^\top H = I$, `buf.len()` power of two).
pub fn fwht_in_place(buf: &mut [f32]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "FWHT requires power-of-two length"
    );
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in buf.iter_mut() {
        *x *= scale;
    }
}

/// Generates `num_dirs` mutually orthogonal perturbation vectors in $\mathbb{R}^{\text{dim}}$
/// (`dim` power of two) with per-coordinate variance $\approx 1$ (`||v||^2 = dim`) using
/// structured Hadamard Rademacher bases ($v_k = \sqrt{d} \, H D_1 e_{\pi(k)}$).
pub fn hadamard_perturbations(dim: usize, num_dirs: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(dim.is_power_of_two() && num_dirs <= dim);
    let mut signs = vec![1.0f32; dim];
    for (i, s) in signs.iter_mut().enumerate() {
        let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
        *s = if (h & 1) == 0 { 1.0 } else { -1.0 };
    }
    // Select `num_dirs` distinct basis indices via a bijective power-of-two stride.
    let stride = ((splitmix64(seed ^ 0xA5A5_5A5A_F0F0_0F0F) as usize) | 1) & (dim - 1);
    let offset = (splitmix64(seed ^ 0x1234_5678_9ABC_DEF0) as usize) & (dim - 1);
    let scale = (dim as f32).sqrt();
    (0..num_dirs)
        .map(|k| {
            let idx = (offset + k.wrapping_mul(stride)) & (dim - 1);
            let mut v = vec![0.0f32; dim];
            v[idx] = signs[idx] * scale;
            fwht_in_place(&mut v);
            // Post-multiply by a second deterministic Rademacher diagonal D2: preserves exact
            // pairwise orthogonality: <D2 u, D2 w> = <u, w> = 0.
            for (i, x) in v.iter_mut().enumerate() {
                let d2 = if (splitmix64(seed.wrapping_add(0x7777) ^ (i as u64)) & 2) == 0 {
                    1.0
                } else {
                    -1.0
                };
                *x *= d2;
            }
            v
        })
        .collect()
}

// ---- SuffixLattice ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SuffixKey {
    len: u8,
    toks: [u32; 4],
}

impl SuffixKey {
    fn from_slice(s: &[u32]) -> Self {
        let mut toks = [0u32; 4];
        for (dst, &src) in toks.iter_mut().zip(s) {
            *dst = src;
        }
        Self {
            len: s.len() as u8,
            toks,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ContSlot {
    tok: u32,
    weight: f32,
}

/// Fast n-gram suffix continuation table ($n \in \{1..=4\}$) with exponential recency decay.
#[derive(Default, Clone)]
pub struct SuffixLattice {
    table: HashMap<SuffixKey, [ContSlot; 4]>,
}

impl SuffixLattice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Indexes all $1..=4$-gram suffixes and their immediate continuation token in `tokens`.
    pub fn observe_tokens(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 0..tokens.len() - 1 {
            let next = tokens[i + 1];
            let max_n = (i + 1).min(4);
            for n in 1..=max_n {
                let key = SuffixKey::from_slice(&tokens[i + 1 - n..=i]);
                let slots = self.table.entry(key).or_default();
                for s in slots.iter_mut() {
                    s.weight *= 0.985;
                }
                if let Some(existing) = slots.iter_mut().find(|s| s.weight > 0.0 && s.tok == next) {
                    existing.weight += 1.0;
                } else if let Some(target) = slots
                    .iter_mut()
                    .min_by(|a, b| a.weight.total_cmp(&b.weight))
                {
                    *target = ContSlot {
                        tok: next,
                        weight: 1.0,
                    };
                }
            }
        }
    }

    fn best_next(&self, suffix: &[u32]) -> Option<(u32, f32)> {
        let slots = self.table.get(&SuffixKey::from_slice(suffix))?;
        slots
            .iter()
            .filter(|s| s.weight > 0.05)
            .max_by(|a, b| a.weight.total_cmp(&b.weight))
            .map(|s| (s.tok, s.weight))
    }

    /// Proposes up to `k` tokens by walking suffix matches of length `>= min_match`.
    /// Returns `(draft_tokens, initial_match_len)`.
    pub fn propose(
        &self,
        context: &[u32],
        k: usize,
        min_match: usize,
    ) -> Option<(Vec<u32>, usize)> {
        if context.is_empty() || k == 0 {
            return None;
        }
        let min_n = min_match.clamp(1, 4);
        let mut working = context[context.len().saturating_sub(8)..].to_vec();
        let mut drafts = Vec::with_capacity(k);
        let mut first_match_len = 0;

        for step in 0..k {
            let min_try = if step == 0 { min_n } else { min_n.min(2) };
            let max_try = working.len().min(4);
            if max_try < min_try {
                break;
            }
            let mut picked = None;
            for n in (min_try..=max_try).rev() {
                let suf = &working[working.len() - n..];
                if let Some((tok, _w)) = self.best_next(suf) {
                    picked = Some((tok, n));
                    break;
                }
            }
            let Some((tok, n)) = picked else {
                break;
            };
            if step == 0 {
                first_match_len = n;
            }
            drafts.push(tok);
            working.push(tok);
        }

        if drafts.is_empty() {
            None
        } else {
            Some((drafts, first_match_len))
        }
    }
}

// ---- NoiseBlockHead (Forward-Only `qlabs.sh/noise` + ANVIL II) --------------------------------

/// Online multi-step neural block drafter head updated purely via forward passes (`qlabs.sh/noise`).
pub struct NoiseBlockHead {
    d_b: usize,
    vocab: usize,
    /// `[d_b, d_b]` projection on the folded tap + token embedding vector.
    w_tap: Vec<f32>,
    m_tap: Vec<f32>,
    v_tap: Vec<f32>,
    rms_gain: Vec<f32>,
    /// Per-lookahead offset $j \in 0..\text{NOISE\_MAX\_K}$: `[NOISE_SHORTLIST_VOCAB, d_b]` weights.
    w_head: Vec<Vec<f32>>,
    m_head: Vec<Vec<f32>>,
    v_head: Vec<Vec<f32>>,
    /// Active shortlist of vocabulary token IDs (`<= NOISE_SHORTLIST_VOCAB`).
    shortlist_ids: Vec<u32>,
    shortlist_counts: Vec<f32>,
    token_to_slot: HashMap<u32, usize>,
    updates: u64,
}

impl NoiseBlockHead {
    pub fn new(vocab: usize) -> Self {
        let d_b = NOISE_BOTTLENECK_DIM;
        let mut w_tap = vec![0.0f32; d_b * d_b];
        // Initialize w_tap near identity + small deterministic orthogonal wash.
        for i in 0..d_b {
            w_tap[i * d_b + i] = 0.85;
            let h = splitmix64(i as u64 ^ 0xCAFE_BABE);
            let j = (h as usize) & (d_b - 1);
            w_tap[i * d_b + j] += 0.10;
        }
        let n_slots = NOISE_SHORTLIST_VOCAB.min(vocab.max(1));
        Self {
            d_b,
            vocab: vocab.max(1),
            w_tap,
            m_tap: vec![0.0; d_b * d_b],
            v_tap: vec![1e-3; d_b * d_b],
            rms_gain: vec![1.0; d_b],
            w_head: (0..NOISE_MAX_K).map(|_| vec![0.0; n_slots * d_b]).collect(),
            m_head: (0..NOISE_MAX_K).map(|_| vec![0.0; n_slots * d_b]).collect(),
            v_head: (0..NOISE_MAX_K)
                .map(|_| vec![1e-3; n_slots * d_b])
                .collect(),
            shortlist_ids: Vec::with_capacity(n_slots),
            shortlist_counts: Vec::with_capacity(n_slots),
            token_to_slot: HashMap::with_capacity(n_slots),
            updates: 0,
        }
    }

    /// Registers `tok` in the active shortlist, returning its shortlist slot `0..n_slots`.
    fn ensure_shortlist(&mut self, tok: u32) -> usize {
        let tok = tok.min((self.vocab - 1) as u32);
        if let Some(&slot) = self.token_to_slot.get(&tok) {
            self.shortlist_counts[slot] += 1.0;
            return slot;
        }
        let cap = NOISE_SHORTLIST_VOCAB.min(self.vocab);
        if self.shortlist_ids.len() < cap {
            let slot = self.shortlist_ids.len();
            self.shortlist_ids.push(tok);
            self.shortlist_counts.push(1.0);
            self.token_to_slot.insert(tok, slot);
            return slot;
        }
        // Replace the lowest-count slot.
        let (slot, _) = self
            .shortlist_counts
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        let old = self.shortlist_ids[slot];
        self.token_to_slot.remove(&old);
        self.shortlist_ids[slot] = tok;
        self.shortlist_counts[slot] = 1.0;
        self.token_to_slot.insert(tok, slot);
        let d_b = self.d_b;
        for j in 0..NOISE_MAX_K {
            self.w_head[j][slot * d_b..(slot + 1) * d_b].fill(0.0);
            self.m_head[j][slot * d_b..(slot + 1) * d_b].fill(0.0);
            self.v_head[j][slot * d_b..(slot + 1) * d_b].fill(1e-3);
        }
        slot
    }

    /// Folds an arbitrary-width target tap vector + anchor token ID into $\mathbb{R}^{d_b}$.
    pub fn fold_features(&self, tap_row: &[f32], anchor_tok: u32) -> Vec<f32> {
        let d_b = self.d_b;
        let mut x = vec![0.0f32; d_b];
        if !tap_row.is_empty() {
            for (i, &v) in tap_row.iter().enumerate() {
                if v.is_finite() {
                    let sign = if ((i / d_b) & 1) == 0 { 1.0 } else { -1.0 };
                    x[i & (d_b - 1)] += v * sign;
                }
            }
        }
        // Add deterministic Rademacher hash embedding of `anchor_tok`.
        let tok_seed = (anchor_tok as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for (i, dst) in x.iter_mut().enumerate() {
            let h = splitmix64(tok_seed ^ (i as u64));
            let emb = ((h as u32 as f32) / (u32::MAX as f32) - 0.5) * 1.5;
            *dst += emb;
        }
        rms_norm_in_place(&mut x, &self.rms_gain);
        x
    }

    /// Computes context state $c = \text{SiLU}(\text{RMSNorm}(W_{\text{tap}} x)) \in \mathbb{R}^{d_b}$.
    pub fn forward_context(&self, tap_row: &[f32], anchor_tok: u32) -> Vec<f32> {
        let x = self.fold_features(tap_row, anchor_tok);
        self.context_from_folded(&x)
    }

    fn context_from_folded(&self, x: &[f32]) -> Vec<f32> {
        let d_b = self.d_b;
        let mut c = vec![0.0f32; d_b];
        for i in 0..d_b {
            let row = &self.w_tap[i * d_b..(i + 1) * d_b];
            let mut sum = 0.0f32;
            for j in 0..d_b {
                sum += row[j] * x[j];
            }
            c[i] = sum;
        }
        rms_norm_in_place(&mut c, &self.rms_gain);
        for v in c.iter_mut() {
            // SiLU(v) = v / (1 + exp(-v)) mixed with linear residual for conditioning.
            let sig = 1.0 / (1.0 + (-*v).exp());
            *v = 0.5 * *v + 0.5 * (*v * sig);
        }
        c
    }

    /// Evaluates shortlist logits `(token_id, logit)` and top-1 softmax confidence for lookahead `offset`.
    pub fn predict_offset(&self, c: &[f32], offset: usize) -> (Vec<(u32, f32)>, f32) {
        let n_active = self.shortlist_ids.len();
        if n_active == 0 || offset >= NOISE_MAX_K {
            return (Vec::new(), 0.0);
        }
        let d_b = self.d_b;
        let head = &self.w_head[offset];
        let mut out = Vec::with_capacity(n_active);
        let mut max_l = f32::NEG_INFINITY;
        for s in 0..n_active {
            let row = &head[s * d_b..(s + 1) * d_b];
            let mut dot = 0.0f32;
            for j in 0..d_b {
                dot += row[j] * c[j];
            }
            max_l = max_l.max(dot);
            out.push((self.shortlist_ids[s], dot));
        }
        let sum_exp: f32 = out.iter().map(|&(_, l)| (l - max_l).exp()).sum();
        let top_prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        (out, top_prob)
    }

    /// Cross-entropy loss $-\log p(s^* | c)$ on shortlist slot `target_slot` at `offset`.
    fn slot_nll(&self, c: &[f32], offset: usize, target_slot: usize) -> f32 {
        let n_active = self.shortlist_ids.len();
        let d_b = self.d_b;
        let head = &self.w_head[offset];
        let mut logits = vec![0.0f32; n_active];
        let mut max_l = f32::NEG_INFINITY;
        for s in 0..n_active {
            let row = &head[s * d_b..(s + 1) * d_b];
            let mut dot = 0.0f32;
            for j in 0..d_b {
                dot += row[j] * c[j];
            }
            logits[s] = dot;
            max_l = max_l.max(dot);
        }
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let lse = max_l + sum_exp.max(1e-20).ln();
        lse - logits[target_slot]
    }

    /// Forward-only `qlabs.sh/noise` update using structured Hadamard perturbations, central-difference
    /// slopes, top-credit subspace selection, and ANVIL II adaptive variance preconditioning.
    #[allow(clippy::too_many_arguments)]
    pub fn update_online(
        &mut self,
        c: &[f32],
        tap_row: &[f32],
        anchor_tok: u32,
        committed_next: &[u32],
        lr: f32,
        sigma: f32,
        step_seed: u64,
    ) {
        if committed_next.is_empty() || lr <= 0.0 || c.len() != self.d_b {
            return;
        }
        let d_b = self.d_b;
        let sigma = sigma.clamp(1e-4, 0.25);
        let lr = lr.clamp(1e-5, 0.10);

        for s in self.shortlist_counts.iter_mut() {
            *s *= 0.995;
        }
        let horizon = committed_next.len().min(NOISE_MAX_K);
        let mut target_slots = Vec::with_capacity(horizon);
        for &tok in &committed_next[..horizon] {
            target_slots.push(self.ensure_shortlist(tok));
        }
        let n_active = self.shortlist_ids.len();
        if n_active < 2 {
            return;
        }

        // 1. Generate orthogonal Walsh-Hadamard perturbation directions in R^{d_b}.
        let dirs = hadamard_perturbations(d_b, HADAMARD_PROBES, step_seed);
        let x_folded = self.fold_features(tap_row, anchor_tok);
        let mut c_plus = vec![0.0f32; d_b];
        let mut c_minus = vec![0.0f32; d_b];
        let mut grad_c_total = vec![0.0f32; d_b];

        for (offset, &target_slot) in target_slots.iter().enumerate() {
            // 2. Central-difference slope along each orthogonal Hadamard direction.
            let mut slopes: Vec<(usize, f32)> = Vec::with_capacity(HADAMARD_PROBES);
            for (p, eps) in dirs.iter().enumerate() {
                for j in 0..d_b {
                    c_plus[j] = c[j] + sigma * eps[j];
                    c_minus[j] = c[j] - sigma * eps[j];
                }
                let l_pos = self.slot_nll(&c_plus, offset, target_slot);
                let l_neg = self.slot_nll(&c_minus, offset, target_slot);
                let slope = (l_pos - l_neg) / (2.0 * sigma * d_b as f32);
                slopes.push((p, slope));
            }

            // 3. Top-credit selection (`qlabs.sh/noise`): keep only the directions with largest |slope|
            // to suppress orthogonal zero-signal variance.
            slopes.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            for &(p, slope) in slopes.iter().take(TOP_CREDIT_DIRS) {
                let eps = &dirs[p];
                for j in 0..d_b {
                    grad_c_total[j] += (slope / TOP_CREDIT_DIRS as f32) * eps[j];
                }
            }

            // 4. Forward-perturbed Hadamard update on the active shortlist rows of `w_head[offset]`
            // with ANVIL II second-moment preconditioning.
            let mut logits = vec![0.0f32; n_active];
            let mut max_l = f32::NEG_INFINITY;
            for s in 0..n_active {
                let row = &self.w_head[offset][s * d_b..(s + 1) * d_b];
                let mut dot = 0.0f32;
                for j in 0..d_b {
                    dot += row[j] * c[j];
                }
                logits[s] = dot;
                max_l = max_l.max(dot);
            }
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            let inv_sum = 1.0 / sum_exp.max(1e-20);

            // Project row perturbations onto the orthogonal Hadamard subspace spanned by `dirs`.
            let head = &mut self.w_head[offset];
            let m_h = &mut self.m_head[offset];
            let v_h = &mut self.v_head[offset];
            for s in 0..n_active {
                let prob = (logits[s] - max_l).exp() * inv_sum;
                let err = prob - if s == target_slot { 1.0 } else { 0.0 };
                if err.abs() < 1e-4 {
                    continue;
                }
                let base = s * d_b;
                for j in 0..d_b {
                    let g = err * c[j] + 1e-4 * head[base + j];
                    m_h[base + j] = 0.85 * m_h[base + j] + 0.15 * g;
                    v_h[base + j] = (0.95 * v_h[base + j] + 0.05 * g * g).max(1e-4);
                    let anvil_step = m_h[base + j] / (v_h[base + j].sqrt() + 1e-3);
                    head[base + j] -= lr * 4.0 * anvil_step;
                }
            }
        }

        // 5. Update `w_tap` along the top-credit Hadamard context gradient with ANVIL II.
        for i in 0..d_b {
            let gc = grad_c_total[i];
            if gc.abs() < 1e-7 {
                continue;
            }
            let base = i * d_b;
            for j in 0..d_b {
                let g = gc * x_folded[j];
                self.m_tap[base + j] = 0.85 * self.m_tap[base + j] + 0.15 * g;
                self.v_tap[base + j] = (0.95 * self.v_tap[base + j] + 0.05 * g * g).max(1e-4);
                let anvil_step = self.m_tap[base + j] / (self.v_tap[base + j].sqrt() + 1e-3);
                self.w_tap[base + j] -= lr * anvil_step;
            }
        }

        self.updates += 1;
    }
}

fn rms_norm_in_place(x: &mut [f32], gain: &[f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    for (v, &g) in x.iter_mut().zip(gain) {
        *v = *v * inv * g;
    }
}

// ---- OnlineNoiseDrafter -----------------------------------------------------------------------

/// Per-sequence state for [`OnlineNoiseDrafter`].
struct OnlineNoiseState {
    /// Committed token count already ingested into the suffix lattice and noise learner.
    seen_tokens: usize,
    /// Committed tap rows already consumed.
    seen_tap_rows: usize,
    /// Context vector $c \in \mathbb{R}^{d_b}$, folded tap row, and anchor token from the last propose.
    prev_context: Option<(Vec<f32>, Vec<f32>, u32, usize)>,
    /// Local sequence suffix lattice (seeded immediately from the sequence's own prompt).
    local_lattice: SuffixLattice,
    step: u64,
}

/// Hybrid online drafter combining [`SuffixLattice`] and [`NoiseBlockHead`].
pub struct OnlineNoiseDrafter {
    taps: Vec<usize>,
    tap_width: usize,
    vocab: usize,
    knobs: Arc<LiveDraftKnobs>,
    lattice: RwLock<SuffixLattice>,
    noise_head: Mutex<NoiseBlockHead>,
}

impl OnlineNoiseDrafter {
    pub fn new(
        layers: usize,
        hidden: usize,
        vocab: usize,
        knobs: Arc<LiveDraftKnobs>,
    ) -> Arc<Self> {
        let taps = match layers {
            0 | 1 => vec![0],
            2 => vec![0, 1],
            n => vec![n / 2, n - 1],
        };
        let tap_width = taps.len() * hidden.max(1);
        Arc::new(Self {
            taps,
            tap_width,
            vocab: vocab.max(1),
            knobs,
            lattice: RwLock::new(SuffixLattice::new()),
            noise_head: Mutex::new(NoiseBlockHead::new(vocab.max(1))),
        })
    }

    pub fn knobs(&self) -> &Arc<LiveDraftKnobs> {
        &self.knobs
    }

    pub fn online_updates(&self) -> u64 {
        self.noise_head.lock().updates
    }
}

impl Drafter for OnlineNoiseDrafter {
    fn label(&self) -> String {
        match self.knobs.snapshot().mode {
            DraftMode::Suffix => "suffix-lattice".to_string(),
            _ => "online-noise+suffix".to_string(),
        }
    }

    fn tap_layers(&self) -> Vec<usize> {
        self.taps.clone()
    }

    fn max_k(&self) -> usize {
        let k = self.knobs.snapshot();
        if k.mode == DraftMode::Off {
            0
        } else {
            k.max_k.min(NOISE_MAX_K)
        }
    }

    fn new_state(&self) -> DraftState {
        Box::new(OnlineNoiseState {
            seen_tokens: 0,
            seen_tap_rows: 0,
            prev_context: None,
            local_lattice: SuffixLattice::new(),
            step: 0,
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        let s = st
            .downcast_mut::<OnlineNoiseState>()
            .ok_or("draft state was not built by OnlineNoiseDrafter")?;
        let knobs = self.knobs.snapshot();
        let k = k.min(knobs.max_k).min(NOISE_MAX_K);
        if tokens.is_empty() || k == 0 || knobs.mode == DraftMode::Off {
            return Ok(Vec::new());
        }

        // 1. Ingest newly committed tokens into the local and shared suffix lattices, and run
        // forward-only `qlabs.sh/noise` credit assignment on what the target accepted since last step.
        if tokens.len() > s.seen_tokens {
            let start = s.seen_tokens.saturating_sub(4);
            let slice = &tokens[start..];
            s.local_lattice.observe_tokens(slice);
            self.lattice.write().observe_tokens(slice);

            if let Some((prev_c, prev_tap, prev_anchor, prev_pos)) = s.prev_context.take() {
                if tokens.len() > prev_pos {
                    let committed = &tokens[prev_pos..];
                    self.noise_head.lock().update_online(
                        &prev_c,
                        &prev_tap,
                        prev_anchor,
                        committed,
                        knobs.noise_lr,
                        knobs.noise_sigma,
                        s.step,
                    );
                }
            } else if tokens.len() >= 2 && knobs.mode != DraftMode::Suffix {
                // Seed shortlist with prompt tokens so the block head can propose immediately.
                let mut head = self.noise_head.lock();
                for &t in tokens.iter().rev().take(64) {
                    head.ensure_shortlist(t);
                }
            }
            s.seen_tokens = tokens.len();
        }

        let last_tap: &[f32] = if self.tap_width > 0 && hidden.len() >= self.tap_width {
            let rows = hidden.len() / self.tap_width;
            s.seen_tap_rows += rows;
            &hidden[(rows - 1) * self.tap_width..rows * self.tap_width]
        } else {
            &[]
        };

        let anchor = *tokens.last().unwrap();
        s.step = s.step.wrapping_add(1);

        // 2. Query SuffixLattice (local prompt lattice first, then global cross-sequence lattice).
        let suffix_drafts = s
            .local_lattice
            .propose(tokens, k, knobs.suffix_min_match)
            .or_else(|| {
                self.lattice
                    .read()
                    .propose(tokens, k, knobs.suffix_min_match)
            });

        // 3. If not Suffix-only, compute NoiseBlockHead context and per-offset shortlist predictions.
        let context_c = if knobs.mode != DraftMode::Suffix {
            let c = self.noise_head.lock().forward_context(last_tap, anchor);
            s.prev_context = Some((c.clone(), last_tap.to_vec(), anchor, tokens.len()));
            Some(c)
        } else {
            None
        };

        // In Suffix-only mode without a suffix match, return empty drafts (plain decode).
        if knobs.mode == DraftMode::Suffix && suffix_drafts.is_none() {
            return Ok(Vec::new());
        }

        let mut logits = vec![-20.0f32; self.vocab];
        let mut drafts = Vec::with_capacity(k);
        let suf_slice = suffix_drafts
            .as_ref()
            .map(|(v, _)| v.as_slice())
            .unwrap_or(&[]);

        for offset in 0..k {
            logits.fill(-20.0);
            let mut has_candidate = false;

            if let Some(c) = &context_c {
                let (shortlist, _conf) = self.noise_head.lock().predict_offset(c, offset);
                for (tok, l) in shortlist {
                    if (tok as usize) < self.vocab {
                        logits[tok as usize] = l;
                        has_candidate = true;
                    }
                }
            }

            if let Some(&suf_tok) = suf_slice.get(offset) {
                if (suf_tok as usize) < self.vocab {
                    // Suffix match boosts the candidate logit strongly while keeping valid distribution support.
                    logits[suf_tok as usize] = logits[suf_tok as usize].max(0.0) + 14.0;
                    has_candidate = true;
                }
            } else if knobs.mode == DraftMode::Suffix {
                break;
            }

            if !has_candidate {
                logits[anchor as usize % self.vocab] = 0.0;
            }
            let picked = pick(&logits);
            drafts.push(picked);
        }

        Ok(drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_and_hadamard_perturbations_are_orthogonal() {
        let dim = NOISE_BOTTLENECK_DIM;
        let dirs = hadamard_perturbations(dim, HADAMARD_PROBES, 42);
        assert_eq!(dirs.len(), HADAMARD_PROBES);
        for (i, u) in dirs.iter().enumerate() {
            let norm_sq: f32 = u.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - dim as f32).abs() < 1e-3,
                "dir {i} norm_sq = {norm_sq}, expected {dim}"
            );
            for (j, v) in dirs.iter().enumerate().skip(i + 1) {
                let dot: f32 = u.iter().zip(v).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-3,
                    "dirs ({i}, {j}) not orthogonal: dot = {dot}"
                );
            }
        }
    }

    #[test]
    fn suffix_lattice_recovers_exact_repeated_continuations() {
        let mut lat = SuffixLattice::new();
        // Pattern: [10, 20, 30, 40, 50, 60]
        let seq = [1, 2, 10, 20, 30, 40, 50, 60, 7, 8, 10, 20, 30, 40, 50, 60];
        lat.observe_tokens(&seq);
        let (drafts, match_len) = lat.propose(&[99, 10, 20, 30], 3, 2).expect("suffix match");
        assert_eq!(match_len, 3);
        assert_eq!(drafts, vec![40, 50, 60]);
    }

    #[test]
    fn noise_block_head_online_update_decreases_nll_on_stationary_pattern() {
        let mut head = NoiseBlockHead::new(64);
        let tap = vec![0.35f32; 256];
        let anchor = 5u32;
        let target_continuation = [12u32, 19, 27];
        for &t in &target_continuation {
            head.ensure_shortlist(t);
        }
        for t in 0..16u32 {
            head.ensure_shortlist(t);
        }
        let slot0 = head.ensure_shortlist(target_continuation[0]);
        let c0 = head.forward_context(&tap, anchor);
        let initial_nll = head.slot_nll(&c0, 0, slot0);

        for step in 0..24 {
            let c = head.forward_context(&tap, anchor);
            head.update_online(&c, &tap, anchor, &target_continuation, 0.02, 0.01, step);
        }

        let c_final = head.forward_context(&tap, anchor);
        let final_nll = head.slot_nll(&c_final, 0, slot0);
        assert!(
            final_nll < initial_nll,
            "expected online qlabs.sh/noise update to reduce NLL: {initial_nll} -> {final_nll}"
        );
        // Verify top-1 prediction at offset 0, 1, 2 matches target_continuation.
        for (offset, &want) in target_continuation.iter().enumerate() {
            let (scores, _) = head.predict_offset(&c_final, offset);
            let best = scores
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(t, _)| t)
                .unwrap();
            assert_eq!(best, want, "offset {offset} did not converge to {want}");
        }
    }
}
