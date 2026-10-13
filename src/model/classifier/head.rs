//! Graph plumbing shared by the Metal encoder classifiers: a name-deduplicating graph builder that
//! also registers weights, the classification head (pool → dense → activation → norm → output →
//! score activation) as graph nodes, and the Metal session that runs per-length graphs.

use super::Lin;
use crate::graph::{BinaryKind, DType, Graph, Op, ReduceKind, Tensor, UnaryKind, ValueId, Weights};
use std::collections::HashMap;

/// Builds a [`Graph`] and, on the first (registering) build, the [`Weights`] it reads. Weight names
/// are deduplicated, so shared (ALBERT) layers declare each tensor once.
pub struct Gb {
    pub g: Graph,
    /// `Some` while registering: every new weight's data is cloned in.
    pub w: Option<Weights>,
    /// Prepended to node names (not weight names), so a shared layer's repeats stay distinct.
    pub scope: String,
    seen: HashMap<String, ValueId>,
}

impl Gb {
    pub fn new(register: bool) -> Self {
        Self::from_graph(Graph::new(), register)
    }

    pub fn from_graph(g: Graph, register: bool) -> Self {
        Self {
            g,
            w: register.then(Weights::new),
            scope: String::new(),
            seen: HashMap::new(),
        }
    }

    pub fn weight(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        data: impl FnOnce() -> Vec<f32>,
    ) -> ValueId {
        if let Some(&v) = self.seen.get(name) {
            return v;
        }
        let v = self.g.weight(name, shape, DType::F32);
        if let Some(w) = &mut self.w {
            w.insert(name, Tensor::F32(data()));
        }
        self.seen.insert(name.to_string(), v);
        v
    }

    pub fn add(&mut self, name: &str, op: Op, ins: &[ValueId], shape: Vec<usize>) -> ValueId {
        self.g
            .add(format!("{}{name}", self.scope), op, ins, shape, DType::F32)
    }

    fn shape(&self, v: ValueId) -> Vec<usize> {
        self.g.value(v).shape.clone()
    }

    /// `x [rows, in_f]` → `[rows, out_f]` with `name.weight` (`[out_f, in_f]`) and optional `name.bias`.
    pub fn linear(
        &mut self,
        x: ValueId,
        name: &str,
        w: &[f32],
        b: Option<&[f32]>,
        in_f: usize,
        out_f: usize,
    ) -> ValueId {
        let rows = self.shape(x)[0];
        let mut ins = vec![
            x,
            self.weight(&format!("{name}.weight"), vec![out_f, in_f], || w.to_vec()),
        ];
        if let Some(b) = b {
            ins.push(self.weight(&format!("{name}.bias"), vec![out_f], || b.to_vec()));
        }
        self.add(name, Op::Linear, &ins, vec![rows, out_f])
    }

    pub(super) fn lin(&mut self, x: ValueId, name: &str, l: &Lin) -> ValueId {
        self.linear(x, name, &l.w, l.b.as_deref(), l.in_f, l.out_f)
    }

    pub fn ln(
        &mut self,
        x: ValueId,
        name: &str,
        (g, b): &(Vec<f32>, Vec<f32>),
        eps: f32,
    ) -> ValueId {
        let shape = self.shape(x);
        let d = g.len();
        let gv = self.weight(&format!("{name}.weight"), vec![d], || g.clone());
        let bv = self.weight(&format!("{name}.bias"), vec![d], || b.clone());
        self.add(name, Op::LayerNorm { eps }, &[x, gv, bv], shape)
    }

    pub fn unary(&mut self, name: &str, kind: UnaryKind, x: ValueId) -> ValueId {
        let shape = self.shape(x);
        self.add(name, Op::Unary { kind }, &[x], shape)
    }

    pub fn act(&mut self, name: &str, a: Act, x: ValueId) -> ValueId {
        let shape = self.shape(x);
        let op = match a {
            Act::Tanh => return self.unary(name, UnaryKind::Tanh, x),
            Act::Relu => Op::Relu,
            Act::Gelu => Op::Gelu { tanh: false },
            Act::GeluTanh => Op::Gelu { tanh: true },
            Act::Silu => Op::Silu,
        };
        self.add(name, op, &[x], shape)
    }

    pub fn slice(
        &mut self,
        name: &str,
        x: ValueId,
        axis: isize,
        start: usize,
        end: usize,
    ) -> ValueId {
        let mut shape = self.shape(x);
        let ax = if axis < 0 {
            shape.len() as isize + axis
        } else {
            axis
        } as usize;
        shape[ax] = end - start;
        let op = Op::Slice {
            axis,
            start: start as i64,
            end: end as i64,
        };
        self.add(name, op, &[x], shape)
    }

    pub fn reshape(&mut self, name: &str, x: ValueId, shape: Vec<usize>) -> ValueId {
        self.add(
            name,
            Op::Reshape {
                shape: Some(shape.clone()),
            },
            &[x],
            shape,
        )
    }
}

/// Activation between a head's dense layer and its output (and inside ModernBERT's MLP).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Tanh,
    Relu,
    Gelu,
    GeluTanh,
    Silu,
}

impl HeadPost {
    pub fn of(a: super::Activation) -> Self {
        match a {
            super::Activation::Softmax => Self::Softmax,
            super::Activation::Sigmoid => Self::Sigmoid,
            super::Activation::Identity => Self::Logits,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadPool {
    /// Row 0 of the last hidden states.
    First,
    /// Mean over the (all-ones) mask.
    Mean,
    Max,
}

/// What turns the output linear's logits into the scores the classifier reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadPost {
    Logits,
    Softmax,
    Sigmoid,
    /// sklearn `LogisticRegression.predict_proba`: OvR (normalised sigmoids) or multinomial; a
    /// single decision value is expanded to the two class probabilities.
    Logistic {
        ovr: bool,
    },
}

pub struct HeadLin {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub in_f: usize,
    pub out_f: usize,
}

impl From<&Lin> for HeadLin {
    fn from(l: &Lin) -> Self {
        Self {
            w: l.w.clone(),
            b: l.b.clone(),
            in_f: l.in_f,
            out_f: l.out_f,
        }
    }
}

/// The classification head as graph nodes under `head.*`.
pub struct HeadGraph {
    pub pool: HeadPool,
    /// F.normalize of the pooled vector (SetFit).
    pub normalize: bool,
    pub dense: Option<(HeadLin, Act)>,
    /// LayerNorm `(gamma, beta, eps)` after the dense activation (ModernBERT).
    pub norm: Option<(Vec<f32>, Vec<f32>, f32)>,
    pub out: HeadLin,
    pub post: HeadPost,
}

impl HeadGraph {
    /// `x [n, h]` (last hidden states) → scores `[1, labels]`. `mask [n]` is needed for `Mean`.
    pub fn nodes(&self, gb: &mut Gb, x: ValueId, mask: Option<ValueId>) -> Result<ValueId, String> {
        let h = gb.shape(x)[1];
        let mut v = match self.pool {
            HeadPool::First => gb.slice("head.first", x, 0, 0, 1),
            HeadPool::Mean => {
                let mask = mask.ok_or("mean pooling needs the attention mask input")?;
                let m = gb.add("head.mean", Op::MeanPool, &[x, mask], vec![h]);
                gb.reshape("head.mean.row", m, vec![1, h])
            }
            HeadPool::Max => {
                let op = Op::Reduce {
                    kind: ReduceKind::Max,
                    axes: vec![0],
                    keep_dims: true,
                };
                gb.add("head.max", op, &[x], vec![1, h])
            }
        };
        if self.normalize {
            v = gb.add("head.l2", Op::L2Normalize, &[v], vec![1, h]);
        }
        if let Some((l, a)) = &self.dense {
            v = gb.linear(v, "head.dense", &l.w, l.b.as_deref(), l.in_f, l.out_f);
            v = gb.act("head.dense.act", *a, v);
        }
        if let Some((g, b, eps)) = &self.norm {
            v = gb.ln(v, "head.norm", &(g.clone(), b.clone()), *eps);
        }
        let o = &self.out;
        let logits = gb.linear(v, "head.out", &o.w, o.b.as_deref(), o.in_f, o.out_f);
        let l = o.out_f;
        Ok(match self.post {
            HeadPost::Logits => logits,
            HeadPost::Softmax => gb.add(
                "head.softmax",
                Op::Softmax { axis: -1 },
                &[logits],
                vec![1, l],
            ),
            HeadPost::Sigmoid => gb.unary("head.sigmoid", UnaryKind::Sigmoid, logits),
            HeadPost::Logistic { ovr } if l == 1 => {
                // [1 - sigmoid(d), sigmoid(d)] = sigmoid([-d, d]); multinomial: softmax([-d, d]).
                let neg = gb.unary("head.neg", UnaryKind::Neg, logits);
                let both = gb.add(
                    "head.both",
                    Op::Concat { axis: -1 },
                    &[neg, logits],
                    vec![1, 2],
                );
                if ovr {
                    gb.unary("head.sigmoid", UnaryKind::Sigmoid, both)
                } else {
                    gb.add(
                        "head.softmax",
                        Op::Softmax { axis: -1 },
                        &[both],
                        vec![1, 2],
                    )
                }
            }
            HeadPost::Logistic { ovr: false } => gb.add(
                "head.softmax",
                Op::Softmax { axis: -1 },
                &[logits],
                vec![1, l],
            ),
            HeadPost::Logistic { ovr: true } => {
                let s = gb.unary("head.sigmoid", UnaryKind::Sigmoid, logits);
                let op = Op::Reduce {
                    kind: ReduceKind::Sum,
                    axes: vec![-1],
                    keep_dims: true,
                };
                let sum = gb.add("head.sum", op, &[s], vec![1, 1]);
                gb.add(
                    "head.norm_ovr",
                    Op::Binary {
                        kind: BinaryKind::Div,
                    },
                    &[s, sum],
                    vec![1, l],
                )
            }
        })
    }

    /// Every weight the head reads, under its graph name.
    pub fn weights(&self, h: usize) -> Weights {
        let mut gb = Gb::new(true);
        let x = gb.g.input("x", vec![1, h], DType::F32);
        let m = gb.g.input("mask", vec![1], DType::U32);
        self.nodes(&mut gb, x, Some(m)).expect("mask supplied");
        gb.w.expect("registering")
    }
}

/// A Metal session with the model's weights uploaded once; runs per-sequence-length graphs.
pub struct GpuRun {
    #[cfg(target_os = "macos")]
    session: crate::graph::metal::Session,
}

impl GpuRun {
    #[cfg(target_os = "macos")]
    pub fn new(weights: &Weights) -> Result<Self, String> {
        let rt = crate::graph::metal::Runtime::shared()?;
        Ok(Self {
            session: crate::graph::metal::Session::new(rt, weights)?,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new(_: &Weights) -> Result<Self, String> {
        Err("the Metal classifier programs need macOS".into())
    }

    /// Runs `graph` and returns its first output as f32.
    #[cfg(target_os = "macos")]
    pub fn run(&self, graph: &Graph, inputs: Vec<Tensor>) -> Result<Vec<f32>, String> {
        self.session
            .run(graph, inputs)?
            .into_iter()
            .next()
            .ok_or("classifier graph produced no output")?
            .into_f32()
    }

    #[cfg(not(target_os = "macos"))]
    pub fn run(&self, _: &Graph, _: Vec<Tensor>) -> Result<Vec<f32>, String> {
        Err("the Metal classifier programs need macOS".into())
    }
}
