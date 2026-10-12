//! T5-family real-weight checks (t5-small, flan-t5, mT5, MADLAD), greedy through the model's chat
//! path (the latest message is the encoder input as is). Each test prints `SKIP` when its
//! checkpoint is not in the repo's `disk/` hub cache (or `GIBBS_T5_STORAGE`).

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::model::native::{load_native, NativeLm};
use gibbs_engine::model::t5::T5Encoder;
use gibbs_engine::model::weights::Weights;
use std::path::PathBuf;
use std::sync::Arc;

fn snapshot(repo: &str) -> PathBuf {
    let root = std::env::var("GIBBS_T5_STORAGE").unwrap_or_else(|_| "disk".into());
    PathBuf::from(root)
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots/main")
}

fn load(repo: &str, weights: &str) -> Option<Arc<dyn NativeLm>> {
    load_on(repo, weights, Device::Cpu)
}

fn load_on(repo: &str, weights: &str, device: Device) -> Option<Arc<dyn NativeLm>> {
    let dir = snapshot(repo);
    if !dir.join(weights).exists() {
        eprintln!("SKIP: no {repo} checkpoint in {}", dir.display());
        return None;
    }
    let artifacts = ModelArtifacts {
        model_id: repo.into(),
        config_path: Some(dir.join("config.json")),
        weights_path: Some(dir.join(weights)),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, device)
            .expect("T5 types are native")
            .expect("T5 checkpoint loads"),
    )
}

/// Greedy reply to one user message.
fn reply(m: &dyn NativeLm, text: &str, max_new: usize) -> String {
    let tok = m.tokenizer();
    let prompt = tok.apply_chat_template(&[("user", text)]);
    assert_eq!(prompt, text, "the encoder input is the user's text as is");
    let (ids, mut state) = m.prepare(tok.encode(&prompt), &[]).expect("prepare");
    assert_eq!(ids.len(), 1, "the decoder prompt is decoder_start_token_id");
    let mut logits = m.forward(&ids, 0, &mut state).expect("prefill");
    let mut out = Vec::new();
    while out.len() < max_new {
        let next = (0..logits.len().min(tok.vocab_size()))
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        if tok.ends_generation(next) {
            break;
        }
        out.push(next);
        logits = m.forward(&[next], out.len(), &mut state).expect("decode");
    }
    let s = tok.decode(&out).trim().to_string();
    eprintln!("{text:?} -> {s:?}");
    s
}

#[test]
fn t5_small_translates_english_to_german() {
    let Some(m) = load("google-t5/t5-small", "model.safetensors") else {
        return;
    };
    assert_eq!(
        reply(
            m.as_ref(),
            "translate English to German: The house is wonderful.",
            32
        ),
        "Das Haus ist wunderbar."
    );
}

/// t5-small on the Metal programs vs the native CPU forward: per-step logits agree (f32 vs f32)
/// through the translation, and the Metal reply is the reference translation.
#[test]
fn t5_small_metal_matches_cpu_and_translates() {
    let (Some(cpu), Some(gpu)) = (
        load("google-t5/t5-small", "model.safetensors"),
        load_on("google-t5/t5-small", "model.safetensors", Device::Metal),
    ) else {
        return;
    };
    assert_eq!(gpu.device(), Device::Metal);
    let text = "translate English to German: The house is wonderful.";
    let tok = cpu.tokenizer();
    let (ids, mut sc) = cpu.prepare(tok.encode(text), &[]).expect("cpu prepare");
    let (_, mut sg) = gpu.prepare(tok.encode(text), &[]).expect("metal prepare");
    let mut next = ids[0];
    let mut worst = 0.0f32;
    for pos in 0..12 {
        let a = cpu.forward(&[next], pos, &mut sc).expect("cpu step");
        let b = gpu.forward(&[next], pos, &mut sg).expect("metal step");
        let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum();
        let e = (num / den).sqrt();
        worst = worst.max(e);
        assert!(e < 1e-3, "step {pos}: rel err {e}");
        next = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap() as u32;
        if tok.ends_generation(next) {
            break;
        }
    }
    eprintln!("t5-small Metal vs CPU: worst step rel err {worst:e}");
    assert_eq!(reply(gpu.as_ref(), text, 32), "Das Haus ist wunderbar.");
}

/// The CLI's generation loop (`--prompt`: builtin drafter lookup, chunked prefill, sampler) on
/// the Metal model gives the same translation as on the CPU. Greedy: the sampler's PRNG hashes
/// the candidates' probability bits, so a sampled draw is not comparable across backends.
#[test]
fn t5_small_cli_generation_loop_on_metal() {
    use gibbs_engine::scheduler::ExecutableModel;
    for device in [Device::Cpu, Device::Metal] {
        let Some(m) = load_on("google-t5/t5-small", "model.safetensors", device) else {
            return;
        };
        let exec = ExecutableModel::Native(m);
        let (out, _, _) = exec
            .generate_text_measured("translate English to German: The house is wonderful.", 32, 0.0, None)
            .expect("generation");
        eprintln!("{device:?}: {out:?}");
        assert_eq!(out.trim(), "Das Haus ist wunderbar.", "{device:?}");
    }
}

#[test]
fn flan_t5_answers_a_question() {
    let Some(m) = load("google/flan-t5-base", "model.safetensors") else {
        return;
    };
    for (q, a) in [
        (
            "Answer the following question. What is the capital of Germany?",
            "berlin",
        ),
        ("What color is the sky on a clear day?", "blue"),
        (
            "Is the following sentence positive or negative? I really loved this movie, it was great!",
            "positive",
        ),
    ] {
        assert_eq!(reply(m.as_ref(), q, 16), a);
    }
}

#[test]
fn mt5_summarizes() {
    // csebuetnlp/mT5_multilingual_XLSum: PyTorch .bin + spiece.model (no tokenizer.json).
    let Some(m) = load("csebuetnlp/mT5_multilingual_XLSum", "pytorch_model.bin") else {
        return;
    };
    let article = "Videos that say approved vaccines are dangerous and cause autism, cancer or \
        infertility are among those that will be taken down, the company said. The policy includes \
        the termination of accounts of anti-vaccine influencers. Tech giants have been criticised \
        for not doing more to counter false health information on their sites. In July, US \
        President Joe Biden said social media platforms were largely responsible for people's \
        scepticism in getting vaccinated by spreading misinformation, and appealed for them to \
        address the issue. YouTube, which is owned by Google, said 130,000 videos were removed \
        from its platform since last year, when it implemented a ban on content spreading \
        misinformation about Covid vaccines.";
    let s = reply(m.as_ref(), article, 64).to_lowercase();
    assert!(s.contains("youtube") && s.contains("vaccine"), "{s}");
}

/// A diffusers T5-XXL text encoder (`text_encoder_2` of a FLUX repo, bf16 shards; set
/// `GIBBS_T5XXL_DIR` to it, `tokenizer_2/` beside it) through [`T5Encoder`]: finite, normed
/// states whose mean-pooled embeddings put a cat nearer a kitten than a tax report.
#[test]
fn t5_xxl_text_encoder_embeds_meaningfully() {
    let Ok(dir) = std::env::var("GIBBS_T5XXL_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: GIBBS_T5XXL_DIR is not set");
        return;
    };
    let w = Weights::open(&dir.join("model.safetensors.index.json")).expect("shards open");
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let enc = T5Encoder::load(&w, &cfg, "").expect("encoder loads");
    let tok = gibbs_engine::model::hf_loader::load_json_tokenizer(
        &dir.join("../tokenizer_2/tokenizer.json"),
    )
    .expect("tokenizer_2");
    let d = enc.d_model();
    let embed = |text: &str| {
        let mut ids = tok.encode(text);
        ids.push(1); // </s>
        let h = enc.encode(&ids);
        assert!(h.iter().all(|v| v.is_finite()), "{text}");
        let mut mean = vec![0.0f32; d];
        for row in h.chunks_exact(d) {
            mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
        }
        let n = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        mean.iter().map(|v| v / n).collect::<Vec<f32>>()
    };
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let (cat, kitten, tax) = (
        embed("a photo of a cat sitting on a sofa"),
        embed("a picture of a kitten resting on a couch"),
        embed("quarterly corporate tax report spreadsheet"),
    );
    let (near, far) = (cos(&cat, &kitten), cos(&cat, &tax));
    eprintln!("cos(cat, kitten) = {near:.4}, cos(cat, tax) = {far:.4}");
    assert!(near > far + 0.05, "{near} vs {far}");
}

/// Mesh-TF score `-sum log p(target | input)` with teacher forcing, as transformers'
/// `test_small_integration_test`s compute it (`-(labels.len() * loss)`).
fn score(m: &dyn NativeLm, input: &str, target: &str) -> (Vec<u32>, Vec<u32>, f64) {
    let tok = m.tokenizer();
    let (input_ids, mut labels) = (tok.encode(input), tok.encode(target));
    labels.push(1); // </s>
    let (ids, mut state) = m.prepare(input_ids.clone(), &[]).expect("prepare");
    let mut prev = ids[0];
    let mut total = 0.0f64;
    for (pos, &label) in labels.iter().enumerate() {
        let logits = m.forward(&[prev], pos, &mut state).expect("decode");
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + logits
                .iter()
                .map(|&l| (l as f64 - max).exp())
                .sum::<f64>()
                .ln();
        total += logits[label as usize] as f64 - lse;
        prev = label;
    }
    (input_ids, labels, total)
}

/// transformers `T5ModelIntegrationTests`: t5-small (v1.0: relu, tied head) scores -19.0845 and
/// t5-v1_1-small (gated-gelu, untied head; `.bin` + `spiece.model`) -59.0293, to 1e-4.
#[test]
fn t5_scores_match_transformers() {
    for (repo, weights, want) in [
        ("google-t5/t5-small", "model.safetensors", -19.0845),
        ("google/t5-v1_1-small", "pytorch_model.bin", -59.0293),
    ] {
        let Some(m) = load(repo, weights) else {
            continue;
        };
        let (input, labels, got) = score(m.as_ref(), "Hello there", "Hi I am");
        eprintln!("{repo}: input {input:?} labels {labels:?} score {got:.4}");
        assert_eq!((input, labels), (vec![8774, 132], vec![2018, 27, 183, 1]));
        assert!((got - want).abs() < 1e-3, "{repo}: {got} vs {want}");
    }
}

/// transformers `UMT5IntegrationTest` (per-layer relative bias): google/umt5-small's greedy span
/// filling of its tokenized inputs, decoded with special tokens like `batch_decode`.
#[test]
fn umt5_fills_spans_like_transformers() {
    let Some(m) = load("google/umt5-small", "pytorch_model.bin") else {
        return;
    };
    let tok = m.tokenizer();
    for (input, want) in [
        (
            &[517u32, 256299, 14869, 281, 301, 256298, 275, 119983, 1][..],
            "<extra_id_0> door<extra_id_1>, the door<extra_id_2> 피해[/</s>",
        ),
        (
            &[1460, 339, 312, 19014, 10620, 758, 256299, 2355, 274, 1][..],
            "<extra_id_0> are not going to be a part of the world. We are not going to be a part \
             of<extra_id_1> and<extra_id_2><0x0A><extra_id_48>.<extra_id_48></s>",
        ),
    ] {
        let (ids, mut state) = m.prepare(input.to_vec(), &[]).expect("prepare");
        let mut logits = m.forward(&ids, 0, &mut state).expect("prefill");
        let mut got = String::new();
        for pos in 1..64 {
            let next = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap() as u32;
            got.push_str(&tok.inv_vocab[&next].replace('\u{2581}', " "));
            if next == 1 {
                break;
            }
            logits = m.forward(&[next], pos, &mut state).expect("decode");
        }
        assert_eq!(got, want);
    }
}
