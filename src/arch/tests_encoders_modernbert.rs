//! ModernBERT encoder spec against the old `SequenceClassifier` CPU host forward (modernbert.rs,
//! scores pinned from commit 251be4c) on the synthetic fixtures (CLS pooling; mean pooling + YaRN),
//! CPU and Metal.

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::classifier::synth::{self, PAIR, TEXT};

/// Old softmax scores of labels A, B, C: (pooling, yarn, [text only, with pair]).
const OLD: [(&str, bool, [[f32; 3]; 2]); 2] = [
    ("cls", false, [[0.15232842, 0.13803738, 0.7096342], [0.15763302, 0.13483696, 0.70753]]),
    ("mean", true, [[0.10009905, 0.68568933, 0.21421164], [0.20956019, 0.58162004, 0.20881984]]),
];

#[test]
fn modernbert_spec_matches_old_classifier() {
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for (pooling, yarn, wants) in OLD {
        let name = format!("spec-modernbert-{pooling}");
        let (cfg, ts) = synth::modernbert(pooling, yarn);
        let cache = synth::write_repo(&name, cfg, &ts);
        let snap = cache.join(format!("models--t--{name}/snapshots/main"));
        let tok = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).unwrap();
        for (pair, want) in [None, Some(PAIR)].into_iter().zip(wants) {
            let e = match pair {
                Some(p) => tok.encode((TEXT, p), false),
                None => tok.encode(TEXT, false),
            }
            .unwrap();
            let ids = e.get_ids().to_vec();
            let n = ids.len();
            for &dev in &devs {
                let mut r = Runner::load(&snap, Some("modernbert"), dev, 64).unwrap_or_else(|e| panic!("{name}: {e}"));
                let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("classify".into()))];
                let mut ins = vec![Tensor::U32(ids.clone()), Tensor::U32((0..n as u32).collect())];
                if pooling == "mean" {
                    ins.push(Tensor::U32(vec![1; n]));
                }
                let logits = match r.run("encoder", &vars, ins).unwrap_or_else(|e| panic!("{name}: {e}")).remove("logits") {
                    Some(Tensor::F32(v)) => v,
                    o => panic!("{name}: logits {o:?}"),
                };
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let z: f32 = logits.iter().map(|v| (v - m).exp()).sum();
                let got: Vec<f32> = logits.iter().map(|v| (v - m).exp() / z).collect();
                let (rel, cos) = compare(&got, &want);
                eprintln!("{name} pair={} {dev:?}: rel {rel:.2e} cos {cos:.8}", pair.is_some());
                assert!(rel < 1e-3 && cos > 0.9999, "{name} {dev:?}: {got:?} vs {want:?}");
                assert!(r.unconsumed().is_empty(), "{name}: unconsumed {:?}", r.unconsumed());
            }
        }
    }
}
