//! Parity of the `gemma4` text spec with `model::gemma4::text` (deleted with it).

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::{StateStore, Tensor};
use crate::hardware::Device;

fn fixture(case: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/golden-fixtures").join(case)
}

/// The decode step graph against `TextModel::to_decode_graph` on the golden fixtures (built by
/// `cargo test --test golden gemma4_*`), CPU executor, prompt fed token by token.
#[test]
fn decode_steps_match_text_model() {
    let _g = crate::hardware::cpu_compute::allow();
    for case in ["gemma_gemma4_decode", "gemma_gemma4_ple", "gemma_gemma4_moe"] {
        let dir = fixture(case);
        if !dir.exists() {
            eprintln!("skip {case}: run the golden test first");
            continue;
        }
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let w = crate::model::weights::Weights::open(&dir.join("model.safetensors")).unwrap();
        let tm = crate::model::gemma4::text::TextModel::load(&w, &cfg["text_config"]).unwrap();
        let (g, gw) = tm.to_decode_graph().unwrap();
        let mut r = Runner::load(&dir, Some("gemma4"), Device::Cpu, tm.max_pos).unwrap();
        let inst = r.instance("text", &[("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))]).unwrap();
        let ops = |g: &crate::graph::Graph| g.nodes.iter().map(|n| format!("{:?}", n.op)).collect::<Vec<_>>();
        let (a, b) = (ops(&g), ops(&inst.graph));
        let pad = b.len().saturating_sub(a.len());
        for (i, (x, y)) in a.iter().zip(&b[pad.min(b.len())..]).enumerate() {
            if x != y {
                eprintln!("{case} node {i}: old {x} new {y}");
            }
        }
        let nw = r.model.store.materialize(&inst.weights).unwrap();
        let vals = |t: &Tensor| -> Vec<f32> {
            match t {
                Tensor::F32(v) => v.clone(),
                Tensor::Mat(m) => m.to_f32(),
                _ => vec![],
            }
        };
        for (i, (on, nn)) in g.nodes.iter().zip(&inst.graph.nodes[pad..]).enumerate() {
            for (oi, ni) in on.inputs.iter().zip(&nn.inputs) {
                let (ov, nv) = (g.value(*oi), inst.graph.value(*ni));
                if ov.kind == crate::graph::ValueKind::Weight && nv.kind == crate::graph::ValueKind::Weight {
                    let (x, y) = (vals(gw.get(&ov.name).unwrap()), vals(nw.get(&nv.name).unwrap()));
                    if x.len() != y.len() || x.iter().zip(&y).any(|(p, q)| (p - q).abs() > 1e-6 * p.abs().max(1.0)) {
                        eprintln!("{case} node {i} {}: weight {} vs {} (len {} / {}) {:?} {:?}", on.name, ov.name, nv.name, x.len(), y.len(), &x[..x.len().min(4)], &y[..y.len().min(4)]);
                    }
                } else if let (crate::graph::ValueKind::Node(a), crate::graph::ValueKind::Node(b)) = (&ov.kind, &nv.kind) {
                    if a.0 as usize + pad != b.0 as usize {
                        eprintln!("{case} node {i} {}: wired to old node {} vs new node {} ({} / {})", on.name, a.0, b.0, ov.name, nv.name);
                    }
                } else if std::mem::discriminant(&ov.kind) != std::mem::discriminant(&nv.kind) {
                    eprintln!("{case} node {i} {}: operand kinds {:?} vs {:?}", on.name, ov.kind, nv.kind);
                }
            }
            if on.inputs.len() != nn.inputs.len() {
                eprintln!("{case} node {i} {}: {} vs {} operands", on.name, on.inputs.len(), nn.inputs.len());
            }
        }
        eprintln!("{case}: old {} nodes, new {}", a.len(), b.len());
        {
            let mut e = vec![0.0f32; tm.hidden];
            tm.embed_row(10, &mut e);
            let t = vals(nw.get("model.language_model.embed_tokens.weight").unwrap());
            let n: Vec<f32> = t[10 * tm.hidden..11 * tm.hidden].iter().map(|v| v * 8.0).collect();
            eprintln!("embed old {:?} new {:?}", &e[..4], &n[..4]);
            let names: Vec<_> = inst.graph.nodes.iter().take(3).map(|n| (n.name.clone(), format!("{:?}", n.op))).collect();
            eprintln!("{names:?} states old {:?}", g.values.iter().filter(|v| matches!(v.kind, crate::graph::ValueKind::State{..})).map(|v| (v.name.clone(), v.shape.clone())).collect::<Vec<_>>());
            eprintln!("states new {:?}", inst.graph.values.iter().filter(|v| matches!(v.kind, crate::graph::ValueKind::State{..})).map(|v| (v.name.clone(), v.shape.clone())).collect::<Vec<_>>());
        }
        let mut st2 = StateStore::new();
        let mut st = StateStore::new();
        for (pos, t) in [2u32, 10, 37, 54, 8].into_iter().enumerate() {
            let mut e = vec![0.0f32; tm.hidden];
            tm.embed_row(t, &mut e);
            let mut ins = vec![Tensor::F32(e), Tensor::U32(vec![pos as u32])];
            if let Some(pl) = &tm.ple {
                let mut row = vec![0.0f32; tm.layers_n * pl.dim];
                pl.embed.get_row(t as usize, &mut row);
                row.iter_mut().for_each(|v| *v *= (pl.dim as f32).sqrt());
                ins.push(Tensor::F32(row));
            }
            let want = crate::graph::cpu::run(&g, &gw, &mut st, ins).unwrap()[0].as_f32().unwrap().to_vec();
            let got = r.lm(&[t], pos, "last", "logits", vec![]);
            let raw = crate::graph::cpu::run(&inst.graph, &nw, &mut st2, vec![Tensor::U32(vec![t]), Tensor::U32(vec![pos as u32])]).unwrap()[0].as_f32().unwrap().to_vec();
            eprintln!("raw vs old {:?} prog vs raw {:?}", compare(&raw, &want), compare(&got, &raw));
            let (rel, cos) = compare(&got, &want);
            assert!(rel < 1e-4 && cos > 0.99999, "{case} pos {pos}: rel {rel:e} cos {cos}");
        }
    }
}
