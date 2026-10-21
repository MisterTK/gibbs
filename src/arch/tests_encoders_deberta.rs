//! DeBERTa v1 / v2 encoder spec (`deberta`) against the old `classifier.rs` `Deberta` host forward
//! on the synthetic fixtures, CPU and Metal. The old scores (`SequenceClassifier::classify` on the
//! CPU at 251be4c: softmax over labels A, B, C) are pinned, so this outlives the old code.

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::classifier::synth::{self, PAIR, TEXT};

/// (fixture, pair input, old scores).
const PINNED: [(&str, bool, [f32; 3]); 4] = [
    ("v1", false, [0.2053402, 0.40398476, 0.39067504]),
    ("v1", true, [0.20440647, 0.39659, 0.39900357]),
    ("v2", false, [0.3586392, 0.27447483, 0.36688596]),
    ("v2", true, [0.37104803, 0.2698484, 0.35910356]),
];

#[test]
fn deberta_specs_match_old_classifier() {
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for (fixture, (cfg, ts)) in [("v1", synth::deberta_v1()), ("v2", synth::deberta_v2())] {
        let name = format!("spec-deberta-{fixture}");
        let snap = synth::write_repo(&name, cfg, &ts).join(format!("models--t--{name}/snapshots/main"));
        let tok = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).unwrap();
        for pair in [false, true] {
            let e = if pair { tok.encode((TEXT, PAIR), false) } else { tok.encode(TEXT, false) }.unwrap();
            let ids = e.get_ids().to_vec();
            let n = ids.len();
            let want = PINNED.iter().find(|p| p.0 == fixture && p.1 == pair).unwrap().2;
            for &dev in &devs {
                let mut r = Runner::load(&snap, None, dev, 64).unwrap_or_else(|e| panic!("{fixture}: {e}"));
                assert_eq!(r.model.spec.family, "deberta");
                let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("classify".into()))];
                let mut ins = vec![Tensor::U32(ids.clone())];
                if fixture == "v1" {
                    ins.push(Tensor::U32((0..n as u32).collect()));
                }
                let logits = match r.run("encoder", &vars, ins).unwrap_or_else(|e| panic!("{fixture}: {e}")).remove("logits") {
                    Some(Tensor::F32(v)) => v,
                    o => panic!("{fixture}: logits {o:?}"),
                };
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let z: f32 = logits.iter().map(|v| (v - m).exp()).sum();
                let got: Vec<f32> = logits.iter().map(|v| (v - m).exp() / z).collect();
                let (rel, cos) = compare(&got, &want);
                eprintln!("deberta {fixture} pair={pair} {dev:?}: rel {rel:.2e} cos {cos:.8}");
                assert!(rel < 1e-3 && cos > 0.9999, "{fixture} {dev:?}: {got:?} vs {want:?}");
                assert!(r.unconsumed().is_empty(), "{fixture}: unconsumed {:?}", r.unconsumed());
            }
        }
    }
}
