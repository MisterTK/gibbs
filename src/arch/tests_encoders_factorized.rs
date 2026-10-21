//! ELECTRA / ALBERT encoder specs on the synthetic fixtures (`synth::electra` / `synth::albert`),
//! CPU and Metal, against the scores the old `SequenceClassifier` (classifier/factorized.rs, CPU
//! host forward, commit 251be4c) gave: softmax over labels A, B, C of `TEXT` and (`TEXT`, `PAIR`).

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::classifier::synth::{self, PAIR, TEXT};

const OLD: [(&str, [[f32; 3]; 2]); 2] = [
    ("electra", [[0.35458475, 0.45721343, 0.18820187], [0.3591797, 0.4508206, 0.18999973]]),
    ("albert", [[0.39195293, 0.29302567, 0.31502134], [0.40488175, 0.3077748, 0.28734347]]),
];

#[test]
fn electra_albert_specs_match_old_classifier() {
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for ((family, old), (cfg, ts)) in OLD.into_iter().zip([synth::electra(), synth::albert()]) {
        let cache = synth::write_repo(&format!("spec-{family}"), cfg, &ts);
        let snap = cache.join(format!("models--t--spec-{family}/snapshots/main"));
        let tok = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).unwrap();
        for (pair, want) in [None, Some(PAIR)].into_iter().zip(old) {
            let e = match pair {
                Some(p) => tok.encode((TEXT, p), false),
                None => tok.encode(TEXT, false),
            }
            .unwrap();
            let (ids, types) = (e.get_ids().to_vec(), e.get_type_ids().to_vec());
            let n = ids.len();
            for &dev in &devs {
                let mut r = Runner::load(&snap, Some(family), dev, 64).unwrap_or_else(|e| panic!("{family}: {e}"));
                let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("classify".into()))];
                let ins = vec![Tensor::U32(ids.clone()), Tensor::U32((0..n as u32).collect()), Tensor::U32(types.clone())];
                let logits = match r.run("encoder", &vars, ins).unwrap_or_else(|e| panic!("{family}: {e}")).remove("logits") {
                    Some(Tensor::F32(v)) => v,
                    o => panic!("{family}: logits {o:?}"),
                };
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let z: f32 = logits.iter().map(|v| (v - m).exp()).sum();
                let got: Vec<f32> = logits.iter().map(|v| (v - m).exp() / z).collect();
                let (rel, cos) = compare(&got, &want);
                eprintln!("{family} pair={} {dev:?}: rel {rel:.2e} cos {cos:.8}", pair.is_some());
                assert!(rel < 1e-3 && cos > 0.9999, "{family} {dev:?}: {got:?} vs {want:?}");
                assert!(r.unconsumed().is_empty(), "{family}: unconsumed {:?}", r.unconsumed());
            }
        }
    }
}
