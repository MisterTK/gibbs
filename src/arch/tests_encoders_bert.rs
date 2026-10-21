//! The `bert` spec's pooling heads (`cls`, `max`, `mean_l2`) against its own `hidden` states
//! pooled on the host, CPU and Metal (SetFit pooling modes; `mean_l2` is System 1's embedding).
//! The `classify` head and the encoder body are pinned by the `enc` goldens.

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::classifier::synth;

#[test]
fn bert_pooling_heads_match_host_pooling() {
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    let (cfg, ts) = synth::bert(None);
    let cache = synth::write_repo("spec-bert-pool", cfg, &ts);
    let snap = cache.join("models--t--spec-bert-pool/snapshots/main");
    let (ids, h) = (vec![3u32, 7, 1, 9, 2, 5], 16);
    let n = ids.len();
    for dev in devs {
        let mut r = Runner::load(&snap, Some("bert"), dev, 32).unwrap();
        let mut run = |head: &str| {
            let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str(head.into()))];
            let mut ins = vec![Tensor::U32(ids.clone()), Tensor::U32((0..n as u32).collect()), Tensor::U32(vec![0; n])];
            if head.starts_with("mean") {
                ins.push(Tensor::U32(vec![1; n]));
            }
            let mut o = r.run("encoder", &vars, ins).unwrap_or_else(|e| panic!("{head}: {e}"));
            match o.remove(if head == "hidden" { "hidden" } else { "embedding" }) {
                Some(Tensor::F32(v)) => v,
                other => panic!("{head}: {other:?}"),
            }
        };
        let hidden = run("hidden");
        let rows: Vec<&[f32]> = hidden.chunks(h).collect();
        let max: Vec<f32> = (0..h).map(|j| rows.iter().map(|r| r[j]).fold(f32::MIN, f32::max)).collect();
        let mean: Vec<f32> = (0..h).map(|j| rows.iter().map(|r| r[j]).sum::<f32>() / n as f32).collect();
        let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mean_l2: Vec<f32> = mean.iter().map(|v| v / norm).collect();
        for (head, want) in [("cls", rows[0].to_vec()), ("max", max), ("mean_l2", mean_l2)] {
            let (rel, cos) = compare(&run(head), &want);
            eprintln!("bert {head} {dev:?}: rel {rel:.2e} cos {cos:.8}");
            assert!(rel < 1e-5, "{head} {dev:?}: rel {rel}");
        }
    }
}
