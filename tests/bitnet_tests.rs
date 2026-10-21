//! BitNet b1.58: the three published encodings of microsoft/bitnet-b1.58-2B-4T (packed 2-bit
//! safetensors, bf16 master weights ternarized at load, bitnet.cpp I2_S GGUF) must decode to the same
//! ternary weights (up to bf16 master-weight ties, below), and the packed 2B-4T and 1bitLLM's
//! `llama`-typed checkpoint must answer coherently through the family spec (`runtime::lm::SpecLm`).
//! BitLinear activation quant rounding: `arch::build` `act_quant_rounds_half_to_even_and_clamps_like_torch`.

use gibbs_engine::arch::load::ternarize;
use gibbs_engine::hardware::PlacementMode;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::model::loaders::{SafetensorsWeightReader, WeightReader};
use gibbs_engine::model::tokenizer::Tokenizer;
use gibbs_engine::scheduler::ExecutableModel;
use gibbs_engine::model::{bf16_to_f32, f32_to_bf16, UniversalModelLoader};
use std::path::PathBuf;

const PROJS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];
const GGUF_PROJS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];
const LAYERS: usize = 30;

/// `file` of `repo` in the shared `disk/` cache or the large-model cache `/tmp/gibbs-models`.
fn snapshot_file(repo: &str, file: &str) -> Option<PathBuf> {
    let dir = format!("models--{}/snapshots/main", repo.replace('/', "--"));
    [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("disk"),
        PathBuf::from("/tmp/gibbs-models"),
    ]
    .into_iter()
    .map(|root| root.join(&dir).join(file))
    .find(|p| p.is_file())
}

fn bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// Serializes the real-weight checks: each holds GBs of weights.
static HEAVY: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Greedy completion of `prompt` by `repo` loaded through the spec loader (detected family, re-encode
/// gate); `None` (after printing SKIP) when the checkpoint is not downloaded.
fn complete(repo: &str, prompt: impl Fn(&Tokenizer) -> String) -> Option<String> {
    let Some(weights) = snapshot_file(repo, "model.safetensors") else {
        println!("SKIP: {repo} not downloaded");
        return None;
    };
    let _heavy = HEAVY.lock();
    let dir = weights.parent().unwrap().to_path_buf();
    let artifacts = ModelArtifacts {
        model_id: repo.into(),
        local_dir: Some(dir.clone()),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(weights),
        ..Default::default()
    };
    let reg = gibbs_engine::arch::registry().unwrap();
    let lm = gibbs_engine::runtime::lm::load(&artifacts, reg, None, PlacementMode::Auto).unwrap();
    let m = ExecutableModel::Native(lm);
    let (out, _, _) = m
        .generate_text_measured_plain(&prompt(&m.tokenizer()), 12, 0.0)
        .unwrap();
    println!("{repo}: {out:?}");
    Some(out)
}

/// Packed 2B-4T through the loader: unpacking, relu² MLP, sub-norms and activation quant together.
#[test]
fn bitnet_2b4t_packed_answers_capital_of_france() {
    let Some(out) = complete("microsoft/bitnet-b1.58-2B-4T", |t| {
        t.apply_chat_template(&[("user", "What is the capital of France?")])
    }) else {
        return;
    };
    assert!(out.contains("Paris"), "{out:?}");
}

/// 1bitLLM ships `model_type: llama` with `weight_bits: 1`: BitNet must come from the config (online
/// ternarization, int8 activations) and the checkpoint's `inner_attn_ln` / `ffn_layernorm` norms.
#[test]
fn bitnet_1bitllm_large_continues_prefix() {
    let Some(out) = complete("1bitLLM/bitnet_b1_58-large", |_| {
        "The capital of France is".into()
    }) else {
        return;
    };
    assert!(out.contains("Paris"), "{out:?}");
}

/// The packed checkpoint was ternarized from f32 master weights; the bf16 repo stores those masters
/// rounded to bf16, which lands ~1-2% of them in the one bf16 bucket straddling the ±mean/2
/// rounding threshold, where the packed codes split both ways for the same bf16 value. So: every
/// tensor's scale must match exactly, and codes may differ only by one step for weights within one
/// bf16 ulp of the threshold (never a sign flip, never elsewhere).
#[test]
fn bitnet_2b4t_packed_matches_ternarized_bf16_per_tensor() {
    let packed = "microsoft/bitnet-b1.58-2B-4T";
    let master = "microsoft/bitnet-b1.58-2B-4T-bf16";
    let (Some(p), Some(m)) = (
        snapshot_file(packed, "model.safetensors"),
        snapshot_file(master, "model.safetensors"),
    ) else {
        println!("SKIP: {packed} or {master} model.safetensors not downloaded");
        return;
    };
    let _heavy = HEAVY.lock();
    let (p, m) = (
        SafetensorsWeightReader::new(p).unwrap(),
        SafetensorsWeightReader::new(m).unwrap(),
    );
    let absmax = |w: &[f32]| w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let (mut ties, mut total) = (0usize, 0usize);
    for l in 0..LAYERS {
        for proj in PROJS {
            let name = format!("model.layers.{l}.{proj}.weight");
            let unpacked = p.load_tensor(&name).unwrap();
            let raw = m.load_tensor(&name).unwrap();
            let mut ternarized = raw.clone();
            ternarize(&mut ternarized, bf16);
            assert_eq!(unpacked.len(), ternarized.len(), "{name}");
            let step = absmax(&unpacked);
            assert_eq!(absmax(&ternarized), step, "{name}: scale");
            let half =
                (raw.iter().map(|v| v.abs() as f64).sum::<f64>() / raw.len() as f64) as f32 / 2.0;
            for ((&a, &b), &w) in unpacked.iter().zip(&ternarized).zip(&raw) {
                if a != b {
                    assert_eq!((a - b).abs(), step, "{name}: sign flip at w = {w}");
                    assert!(
                        (w.abs() - half).abs() <= half * 2f32.powi(-7),
                        "{name}: w = {w} differs away from the threshold {half}"
                    );
                    ties += 1;
                }
            }
            total += unpacked.len();
        }
    }
    println!("{ties} of {total} codes differ, all bf16 threshold ties");
}

#[test]
fn bitnet_2b4t_i2_s_gguf_matches_packed_safetensors() {
    let (Some(st), Some(gg)) = (
        snapshot_file("microsoft/bitnet-b1.58-2B-4T", "model.safetensors"),
        snapshot_file("microsoft/bitnet-b1.58-2B-4T-gguf", "ggml-model-i2_s.gguf"),
    ) else {
        println!("SKIP: microsoft/bitnet-b1.58-2B-4T(-gguf) not downloaded");
        return;
    };
    let _heavy = HEAVY.lock();
    let st = SafetensorsWeightReader::new(st).unwrap();
    let file = std::fs::File::open(gg).unwrap();
    let gguf = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let manifest = UniversalModelLoader::parse_gguf(&gguf).unwrap();
    for l in 0..LAYERS {
        for (proj, gproj) in PROJS.iter().zip(GGUF_PROJS) {
            let (name, gname) = (
                format!("model.layers.{l}.{proj}.weight"),
                format!("blk.{l}.{gproj}.weight"),
            );
            let hf = st.load_tensor(&name).unwrap();
            let i2 = UniversalModelLoader::extract_tensor_from_manifest(&manifest, &gguf, &gname)
                .unwrap();
            assert_eq!(hf.len(), i2.len(), "{gname}");
            // Same ternary codes; the GGUF keeps the f32 scale the checkpoint stores as bf16.
            let absmax = |w: &[f32]| w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let (s_hf, s_i2) = (absmax(&hf), absmax(&i2));
            assert_eq!(bf16(s_i2), s_hf, "{gname} scale");
            let codes_differ = hf
                .iter()
                .zip(&i2)
                .filter(|(a, b)| *a / s_hf != *b / s_i2)
                .count();
            assert_eq!(codes_differ, 0, "{gname}: ternary codes differ from {name}");
        }
    }
}
