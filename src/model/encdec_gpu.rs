//! The Metal side of the encoder-decoder families (`t5`, `seq2seq`): one encoder program per
//! source-length bucket and one single-position decoder program, with the cross-attention K/V and
//! the decoder's self-attention K/V living in per-sequence [`StateStore`] buffers.
//!
//! The encoder program also projects its output through every decoder layer's cross K/V weights,
//! so decoding never touches the encoder again: [`EncDec::start`] runs it once, drops the
//! `xk.{l}`/`xv.{l}` projections into the sequence's state, and [`EncDec::step`] (one decoder
//! position per call) reads them as `Attention` operands. Source padding is a `-inf` key bias.
//! Beams are separate sequences: [`EncDec::fork`] shares the cross buffers and copies the
//! self-attention rows.

use crate::graph::program::GraphProgram;
use crate::graph::{DType, Graph, Op, StateStore, Tensor, UnaryKind, ValueId, Weights};
use crate::hardware::{Device, PlacementMode};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Mask value of a padded / future key (added to the attention scores).
pub const MASKED: f32 = f32::NEG_INFINITY;

/// Shapes the encoder and decoder graphs agree on.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    /// Decoder layers (one cross K/V pair each).
    pub layers: usize,
    /// K/V width (`heads * head_dim`).
    pub inner: usize,
    /// Longest source (cross-attention state rows).
    pub enc_cap: usize,
    /// Longest decoder sequence (self-attention state rows).
    pub self_cap: usize,
    /// Position-table row offset (BART: 2). `Some` adds the decoder `row` input and shifts the
    /// encoder `pos` input by it; `None` (T5) feeds raw positions.
    pub row_offset: Option<usize>,
    /// The decoder graph takes a `[self_cap]` key mask input after the position inputs (needed
    /// when self-attention is an `Attention` over the whole state, i.e. T5's biased one).
    pub self_mask: bool,
}

/// Named weights and the `(shape, dtype)` each graph declares them with.
#[derive(Default)]
pub struct Bag {
    pub tensors: Weights,
    sig: HashMap<String, (Vec<usize>, DType)>,
}

/// The declarations alone, shared by the graph builders after the tensors moved elsewhere.
#[derive(Clone, Default)]
pub struct Sig(Arc<HashMap<String, (Vec<usize>, DType)>>);

impl Bag {
    pub fn add(&mut self, name: &str, shape: Vec<usize>, dtype: DType, t: Tensor) {
        self.sig.insert(name.to_string(), (shape, dtype));
        self.tensors.insert(name, t);
    }

    pub fn sig(&self) -> Sig {
        Sig(Arc::new(self.sig.clone()))
    }
}

impl Sig {
    /// Declares weight `name` in `g`.
    pub fn w(&self, g: &mut Graph, name: &str) -> ValueId {
        let (shape, dtype) = self
            .0
            .get(name)
            .unwrap_or_else(|| panic!("encoder-decoder weight '{name}' was never registered"));
        g.weight(name, shape.clone(), *dtype)
    }
}

/// F32 node.
pub fn node(g: &mut Graph, name: impl Into<String>, op: Op, ins: &[ValueId], shape: &[usize]) -> ValueId {
    g.add(name, op, ins, shape.to_vec(), DType::F32)
}

pub fn scale(g: &mut Graph, name: impl Into<String>, x: ValueId, s: f32, shape: &[usize]) -> ValueId {
    node(g, name, Op::Unary { kind: UnaryKind::Scale(s) }, &[x], shape)
}

/// Names of the weights `g` declares.
fn weight_names(g: &Graph) -> HashSet<String> {
    g.values
        .iter()
        .filter(|v| v.kind == crate::graph::ValueKind::Weight)
        .map(|v| v.name.clone())
        .collect()
}

/// Source lengths the encoder is compiled for: the next of these at or above the source.
const ENC_BUCKETS: [usize; 2] = [64, 256];

type EncBuilder = Box<dyn Fn(usize) -> Graph + Send + Sync>;

pub struct EncDec {
    d: Dims,
    build_enc: EncBuilder,
    /// Encoder weights, cloned into each bucket's program (a program owns its device copy).
    enc_weights: Weights,
    enc: Mutex<HashMap<usize, GraphProgram>>,
    dec: Mutex<GraphProgram>,
}

/// One decoding sequence (or beam): its cross + self-attention state and decoded length.
pub struct Seq {
    state: StateStore,
    /// Decoder positions filled.
    pub len: usize,
    n_enc: usize,
    cross_mask: Vec<f32>,
}

impl EncDec {
    /// Compiles the decoder (`dec` graph + the weights it names) and keeps the encoder weights for
    /// the buckets. `bag` holds every tensor of both programs; the decoder's are moved out of it,
    /// the encoder's cloned (the token embedding is in both).
    pub fn new(
        d: Dims,
        mut bag: Bag,
        build_enc: EncBuilder,
        dec: Graph,
    ) -> Result<Self, String> {
        // ponytail: every encoder bucket re-uploads the encoder weights (a program owns its device
        // buffers); share one Session across buckets if multi-bucket memory matters.
        let probe = build_enc(ENC_BUCKETS[0].min(d.enc_cap));
        let (enc_names, dec_names) = (weight_names(&probe), weight_names(&dec));
        let mut enc_weights = Weights::new();
        for n in &enc_names {
            let t = if dec_names.contains(n) {
                bag.tensors.get(n).cloned()
            } else {
                bag.tensors.remove(n)
            };
            enc_weights.insert(n.clone(), t.ok_or_else(|| format!("weight '{n}' missing"))?);
        }
        let mut dec_weights = Weights::new();
        for n in &dec_names {
            let t = bag.tensors.remove(n).ok_or_else(|| format!("weight '{n}' missing"))?;
            dec_weights.insert(n.clone(), t);
        }
        let dec = GraphProgram::compile(dec, dec_weights, PlacementMode::Pin(Device::Metal))?;
        Ok(Self {
            d,
            build_enc,
            enc_weights,
            enc: Mutex::new(HashMap::new()),
            dec: Mutex::new(dec),
        })
    }

    pub fn backend_label(&self) -> String {
        self.dec.lock().backend_label()
    }

    pub fn cpu_node_count(&self) -> usize {
        self.dec.lock().cpu_node_count() + self.enc.lock().values().map(|p| p.cpu_node_count()).sum::<usize>()
    }

    /// Runs the encoder over `src` and returns a sequence at decoder position 0.
    pub fn start(&self, src: &[u32]) -> Result<Seq, String> {
        let (n, d) = (src.len(), &self.d);
        if n == 0 || n > d.enc_cap {
            return Err(format!("source of {n} tokens is outside 1..={}", d.enc_cap));
        }
        let bucket = ENC_BUCKETS
            .into_iter()
            .find(|&b| b >= n && b < d.enc_cap)
            .unwrap_or(d.enc_cap);
        let mut ids = src.to_vec();
        ids.resize(bucket, 0);
        let off = d.row_offset.unwrap_or(0) as u32;
        let pos: Vec<u32> = (0..bucket as u32).map(|p| p + off).collect();
        let mask: Vec<f32> = (0..bucket).map(|j| if j < n { 0.0 } else { MASKED }).collect();
        let outs = {
            let mut progs = self.enc.lock();
            if !progs.contains_key(&bucket) {
                let p = GraphProgram::compile(
                    (self.build_enc)(bucket),
                    self.enc_weights.clone(),
                    PlacementMode::Pin(Device::Metal),
                )?;
                progs.insert(bucket, p);
            }
            let p = progs.get_mut(&bucket).expect("inserted above");
            p.run(vec![Tensor::U32(ids), Tensor::U32(pos), Tensor::F32(mask)])?
        };
        if outs.len() != 2 * d.layers {
            return Err(format!("encoder returned {} tensors, expected {}", outs.len(), 2 * d.layers));
        }
        // The cross K/V reach the decoder's `State` buffers through the store (its first run
        // uploads them; rows past the source stay zero and are masked).
        let mut state = StateStore::new();
        for (i, t) in outs.into_iter().enumerate() {
            let mut kv = t.into_f32()?;
            kv.truncate(n * d.inner);
            state.insert(format!("{}.{}", if i % 2 == 0 { "xk" } else { "xv" }, i / 2), Tensor::F32(kv));
        }
        let cross_mask = (0..d.enc_cap).map(|j| if j < n { 0.0 } else { MASKED }).collect();
        Ok(Seq { state, len: 0, n_enc: n, cross_mask })
    }

    /// Source positions of `seq`.
    pub fn n_src(&self, seq: &Seq) -> usize {
        seq.n_enc
    }

    /// Feeds `tok` at decoder position `seq.len`; returns the next-token logits `[vocab]`.
    pub fn step(&self, seq: &mut Seq, tok: u32) -> Result<Vec<f32>, String> {
        let d = &self.d;
        let pos = seq.len;
        if pos >= d.self_cap {
            return Err(format!("decoder position {pos} exceeds the compiled {} positions", d.self_cap));
        }
        let mut inputs = vec![Tensor::U32(vec![tok]), Tensor::U32(vec![pos as u32])];
        if let Some(off) = d.row_offset {
            inputs.push(Tensor::U32(vec![(pos + off) as u32]));
        }
        if d.self_mask {
            let mask = (0..d.self_cap).map(|j| if j <= pos { 0.0 } else { MASKED }).collect();
            inputs.push(Tensor::F32(mask));
        }
        inputs.push(Tensor::F32(seq.cross_mask.clone()));
        let mut p = self.dec.lock();
        std::mem::swap(&mut p.state, &mut seq.state);
        let out = p.run(inputs);
        std::mem::swap(&mut p.state, &mut seq.state);
        let logits = out?.into_iter().next().ok_or("decoder returned no logits")?.into_f32()?;
        if pos == 0 {
            // The first run copied the cross K/V into device buffers; the host copies are dead.
            for l in 0..d.layers {
                seq.state.buffers.remove(&format!("xk.{l}"));
                seq.state.buffers.remove(&format!("xv.{l}"));
            }
        }
        seq.len += 1;
        Ok(logits)
    }

    /// A copy of `seq` that continues independently (beam search): cross buffers are shared, the
    /// filled self-attention rows copied.
    #[cfg(target_os = "macos")]
    pub fn fork(&self, seq: &Seq) -> Result<Seq, String> {
        let mut state = StateStore::new();
        state.buffers = seq.state.buffers.clone();
        state.metal_f16 = seq.state.metal_f16.clone();
        let device = crate::graph::metal::system_device()?;
        for (name, (buf, n)) in &seq.state.metal_buffers {
            if name.starts_with('x') {
                state.metal_buffers.insert(name.clone(), (buf.clone(), *n));
                continue;
            }
            // ponytail: host memcpy of the filled rows (unified memory) instead of a blit /
            // `StateReorder`, which cannot write state in place.
            // Self-attention caches may be f16 (see `Session::assign_kv_dtypes`); copy bytes.
            let eb = if seq.state.metal_f16.contains(name) { 2 } else { 4 };
            let fresh = crate::graph::metal::new_shared_buffer(&device, *n * eb)?;
            let rows = (seq.len * self.d.inner).min(*n);
            // SAFETY: both buffers are shared-storage with at least `rows` elements of `eb` bytes;
            // steps are synchronous so no GPU work is in flight.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.contents() as *const u8,
                    fresh.contents() as *mut u8,
                    rows * eb,
                );
            }
            state.metal_buffers.insert(name.clone(), (Arc::new(fresh), *n));
        }
        Ok(Seq { state, len: seq.len, n_enc: seq.n_enc, cross_mask: seq.cross_mask.clone() })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn fork(&self, _seq: &Seq) -> Result<Seq, String> {
        Err("encoder-decoder Metal programs need macOS".into())
    }
}

/// Self-attention state declarations of decoder layer `l` (`sk.{l}`, `sv.{l}`).
pub fn self_states(g: &mut Graph, l: usize, d: &Dims) -> (ValueId, ValueId) {
    let shape = vec![d.self_cap, d.inner];
    (
        g.state(format!("sk.{l}"), shape.clone(), DType::F32),
        g.state(format!("sv.{l}"), shape, DType::F32),
    )
}

/// Cross-attention state declarations of decoder layer `l` (`xk.{l}`, `xv.{l}`).
pub fn cross_states(g: &mut Graph, l: usize, d: &Dims) -> (ValueId, ValueId) {
    let shape = vec![d.enc_cap, d.inner];
    (
        g.state(format!("xk.{l}"), shape.clone(), DType::F32),
        g.state(format!("xv.{l}"), shape, DType::F32),
    )
}
