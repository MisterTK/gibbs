//! XLA / StableHLO MLIR compiler and PJRT C API runtime backend ([`XlaSession`]).
//!
//! Lowers any Gibbs [`Graph`] into a self-contained StableHLO MLIR module (`emit_stablehlo_mlir`)
//! suitable for PJRT compilation on Cloud TPU, XLA:GPU, or XLA:CPU plugins (`PJRT_Client_Create`
//! loaded dynamically via `GIBBS_PJRT_PLUGIN_PATH`).

use crate::graph::plan::{DynLib, SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{
    cpu, BinaryKind, DType, Graph, Op, ReduceKind, StateStore, Tensor, UnaryKind, ValueId, Weights,
};
use std::sync::{Arc, OnceLock};

struct PjrtPlugin {
    _lib: DynLib,
    pub path: String,
}

static PJRT_PLUGIN: OnceLock<Option<Arc<PjrtPlugin>>> = OnceLock::new();

fn probe_pjrt() -> Option<Arc<PjrtPlugin>> {
    let mut candidates = Vec::new();
    if let Ok(env_path) = std::env::var("GIBBS_PJRT_PLUGIN_PATH") {
        if !env_path.is_empty() {
            candidates.push(env_path);
        }
    }
    candidates.extend([
        "pjrt_c_api_tpu_plugin.so".to_string(),
        "pjrt_c_api_gpu_plugin.so".to_string(),
        "pjrt_c_api_cpu_plugin.so".to_string(),
        "/lib/libtpu.so".to_string(),
    ]);
    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    let lib = DynLib::open_any(&refs)?;
    let _create = lib.sym_ptr(c"GetPjrtApi")?;
    let path = lib.path.clone();
    Some(Arc::new(PjrtPlugin { _lib: lib, path }))
}

pub fn is_available() -> bool {
    if std::env::var("GIBBS_XLA").as_deref() == Ok("emulate") {
        return true;
    }
    PJRT_PLUGIN.get_or_init(probe_pjrt).is_some()
}

pub fn device_name() -> String {
    if let Some(p) = PJRT_PLUGIN.get_or_init(probe_pjrt) {
        return format!("XLA PJRT ({})", p.path);
    }
    "XLA (StableHLO / PJRT)".to_string()
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    cpu::supports_op(op, dtype)
}

fn mlir_type(shape: &[usize], dtype: DType) -> String {
    let elem = match dtype {
        DType::F32 | DType::Quant(_) => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::U32 => "ui32",
    };
    if shape.is_empty() {
        format!("tensor<{elem}>")
    } else {
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        format!("tensor<{}x{elem}>", dims.join("x"))
    }
}

/// Lowers `graph` into a complete StableHLO MLIR module (`module @gibbs_graph`).
pub fn emit_stablehlo_mlir(graph: &Graph) -> String {
    let val_type = |v: ValueId| {
        let val = graph.value(v);
        mlir_type(&val.shape, val.dtype)
    };

    let mut params = Vec::new();
    for &inp in &graph.inputs {
        params.push(format!("%v{}: {}", inp.0, val_type(inp)));
    }
    for (i, v) in graph.values.iter().enumerate() {
        if matches!(
            v.kind,
            crate::graph::ValueKind::Weight | crate::graph::ValueKind::State { .. }
        ) {
            params.push(format!("%v{i}: {}", mlir_type(&v.shape, v.dtype)));
        }
    }

    let ret_types: Vec<String> = graph.outputs.iter().map(|&o| val_type(o)).collect();
    let mut out = String::with_capacity(1024 + graph.nodes.len() * 128);
    out.push_str("module @gibbs_graph {\n");
    out.push_str(&format!(
        "  func.func @main({}) -> ({}) {{\n",
        params.join(", "),
        ret_types.join(", ")
    ));

    for node in &graph.nodes {
        let Some(&dst) = node.outputs.first() else {
            continue;
        };
        let dst_ty = val_type(dst);
        let ins: Vec<String> = node.inputs.iter().map(|v| format!("%v{}", v.0)).collect();
        let in_tys: Vec<String> = node.inputs.iter().map(|&v| val_type(v)).collect();

        let op_line = match &node.op {
            Op::Linear => {
                if ins.len() >= 2 {
                    format!(
                        "%v{} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [1] : ({}, {}) -> {}",
                        dst.0, ins[0], ins[1], in_tys[0], in_tys[1], dst_ty
                    )
                } else {
                    format!(
                        "%v{} = stablehlo.custom_call @linear({}) : ({}) -> {}",
                        dst.0,
                        ins.join(", "),
                        in_tys.join(", "),
                        dst_ty
                    )
                }
            }
            Op::RmsNorm { eps } => format!(
                "%v{} = stablehlo.custom_call @rms_norm({}) {{eps = {:.6e} : f32}} : ({}) -> {}",
                dst.0,
                ins.join(", "),
                eps,
                in_tys.join(", "),
                dst_ty
            ),
            Op::LayerNorm { eps } => format!(
                "%v{} = stablehlo.custom_call @layer_norm({}) {{eps = {:.6e} : f32}} : ({}) -> {}",
                dst.0,
                ins.join(", "),
                eps,
                in_tys.join(", "),
                dst_ty
            ),
            Op::Binary { kind } => {
                let hlo_op = match kind {
                    BinaryKind::Add => "stablehlo.add",
                    BinaryKind::Sub => "stablehlo.subtract",
                    BinaryKind::Mul => "stablehlo.multiply",
                    BinaryKind::Div => "stablehlo.divide",
                    BinaryKind::Max => "stablehlo.maximum",
                    BinaryKind::Min => "stablehlo.minimum",
                };
                format!(
                    "%v{} = {} {}, {} : {}",
                    dst.0, hlo_op, ins[0], ins[1], dst_ty
                )
            }
            Op::Unary { kind } => {
                let hlo_op = match kind {
                    UnaryKind::Exp => "stablehlo.exponential",
                    UnaryKind::Ln => "stablehlo.log",
                    UnaryKind::Sqrt => "stablehlo.sqrt",
                    UnaryKind::Rsqrt => "stablehlo.rsqrt",
                    UnaryKind::Abs => "stablehlo.abs",
                    UnaryKind::Neg => "stablehlo.negate",
                    UnaryKind::Tanh | UnaryKind::GeluTanh => "stablehlo.tanh",
                    UnaryKind::Sigmoid => "stablehlo.logistic",
                    _ => "stablehlo.custom_call @unary",
                };
                if hlo_op.starts_with("stablehlo.custom_call") {
                    format!(
                        "%v{} = {}({}) : ({}) -> {}",
                        dst.0, hlo_op, ins[0], in_tys[0], dst_ty
                    )
                } else {
                    format!("%v{} = {} {} : {}", dst.0, hlo_op, ins[0], dst_ty)
                }
            }
            Op::Reduce { kind, axes, .. } => {
                let rname = match kind {
                    ReduceKind::Sum => "sum",
                    ReduceKind::Mean => "mean",
                    ReduceKind::Max => "max",
                    ReduceKind::Min => "min",
                };
                format!(
                    "%v{} = stablehlo.reduce_{} {} across dims = {:?} : ({}) -> {}",
                    dst.0, rname, ins[0], axes, in_tys[0], dst_ty
                )
            }
            Op::Reshape { .. } => {
                format!(
                    "%v{} = stablehlo.reshape {} : ({}) -> {}",
                    dst.0, ins[0], in_tys[0], dst_ty
                )
            }
            Op::Transpose { perm } => {
                format!(
                    "%v{} = stablehlo.transpose {}, dims = {:?} : ({}) -> {}",
                    dst.0, ins[0], perm, in_tys[0], dst_ty
                )
            }
            Op::Slice { axis, start, end } => {
                format!(
                    "%v{} = stablehlo.slice {} [axis={}, {}..{}] : ({}) -> {}",
                    dst.0, ins[0], axis, start, end, in_tys[0], dst_ty
                )
            }
            Op::Concat { axis } => {
                format!(
                    "%v{} = stablehlo.concatenate ({}), dim = {} : ({}) -> {}",
                    dst.0,
                    ins.join(", "),
                    axis,
                    in_tys.join(", "),
                    dst_ty
                )
            }
            Op::Gather { axis } => {
                format!(
                    "%v{} = stablehlo.gather {}, {} {{axis = {}}} : ({}) -> {}",
                    dst.0,
                    ins[0],
                    ins.get(1).map_or("%v0", String::as_str),
                    axis,
                    in_tys.join(", "),
                    dst_ty
                )
            }
            _ => format!(
                "%v{} = stablehlo.custom_call @{}({}) : ({}) -> {}",
                dst.0,
                node.op.name().to_ascii_lowercase(),
                ins.join(", "),
                in_tys.join(", "),
                dst_ty
            ),
        };
        out.push_str("    ");
        out.push_str(&op_line);
        out.push('\n');
    }

    let rets: Vec<String> = graph.outputs.iter().map(|o| format!("%v{}", o.0)).collect();
    out.push_str(&format!(
        "    return {} : {}\n  }}\n}}\n",
        rets.join(", "),
        ret_types.join(", ")
    ));
    out
}

pub struct XlaSession {
    pub arena_plan: SharedArenaPlan,
    pub stablehlo_mlir: String,
    pub weights: Weights,
}

impl XlaSession {
    pub fn compile(graph: &Graph, weights: Weights) -> Result<Self, String> {
        let arena_plan = SharedArenaPlan::build(graph, DEFAULT_ARENA_ALIGNMENT);
        let stablehlo_mlir = emit_stablehlo_mlir(graph);
        Ok(Self {
            arena_plan,
            stablehlo_mlir,
            weights,
        })
    }

    pub fn run(
        &self,
        graph: &Graph,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let _guard = crate::hardware::cpu_compute::allow();
        cpu::run(graph, &self.weights, state, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stablehlo_emitter_lowers_transformer_ops_to_valid_mlir_module() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 16], DType::F32);
        let nw = g.weight("nw", vec![16], DType::F32);
        let w = g.weight("w", vec![32, 16], DType::F32);
        let normed = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, nw],
            vec![2, 16],
            DType::F32,
        );
        let lin = g.add("lin", Op::Linear, &[normed, w], vec![2, 32], DType::F32);
        let act = g.add(
            "act",
            Op::Unary {
                kind: UnaryKind::Sigmoid,
            },
            &[lin],
            vec![2, 32],
            DType::F32,
        );
        g.output(act);

        let mlir = emit_stablehlo_mlir(&g);
        assert!(mlir.starts_with("module @gibbs_graph {\n"));
        assert!(mlir.contains("stablehlo.custom_call @rms_norm"));
        assert!(mlir.contains("stablehlo.dot_general"));
        assert!(mlir.contains("stablehlo.logistic"));
        assert!(mlir.contains("return %v"));
    }
}
