//! Gemma 4 (google/gemma-4-E2B-it) real-weight checks: text, image and audio prompts answered
//! greedily through the model's own chat template. Each test prints `SKIP` when the checkpoint is
//! not in `GIBBS_GEMMA4_DIR` (default: the repo's `disk/` hub cache).

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hf_loader::ChatMessage;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::model::native::{load_native, MediaInput, NativeLm};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

static MODEL: LazyLock<Option<Arc<dyn NativeLm>>> = LazyLock::new(|| {
    let dir = PathBuf::from(
        std::env::var("GIBBS_GEMMA4_DIR")
            .unwrap_or_else(|_| "disk/models--google--gemma-4-E2B-it/snapshots/main".into()),
    );
    if !dir.join("model.safetensors").exists() {
        println!("SKIP: no Gemma 4 checkpoint in {}", dir.display());
        return None;
    }
    let artifacts = ModelArtifacts {
        model_id: "google/gemma-4-E2B-it".into(),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, Device::Cpu)
            .expect("gemma4 is native")
            .expect("gemma4 loads"),
    )
});

fn model() -> Option<Arc<dyn NativeLm>> {
    MODEL.clone()
}

/// Greedy answer to one user turn whose content parts are `parts` (HF content-part JSON), after
/// checking the cheap `prompt_len` count against what `prepare` lays out.
fn ask(m: &dyn NativeLm, parts: serde_json::Value, media: &[MediaInput], max_new: usize) -> String {
    let tok = m.tokenizer();
    let msg = ChatMessage {
        role: "user".into(),
        content: parts,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let prompt = tok.apply_chat_template_messages(&[msg], None);
    let counted = m
        .prompt_len(tok.encode(&prompt), media)
        .expect("prompt_len");
    let t0 = Instant::now();
    let (ids, mut state) = m.prepare(tok.encode(&prompt), media).expect("prepare");
    let t_prep = t0.elapsed();
    assert_eq!(counted, ids.len(), "prompt_len disagrees with prepare");
    let t1 = Instant::now();
    let mut logits = m.forward(&ids, 0, &mut state).expect("prefill");
    let t_prefill = t1.elapsed();
    let mut out = Vec::new();
    let t2 = Instant::now();
    while out.len() < max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        if tok.is_special(next) {
            break;
        }
        out.push(next);
        logits = m
            .forward(&[next], ids.len() + out.len() - 1, &mut state)
            .expect("decode");
    }
    let t_decode = t2.elapsed();
    let text = tok.decode(&out);
    println!(
        "[{} prompt tokens: media {:.2}s, prefill {:.1} tok/s; {} tokens decoded at {:.1} tok/s]\n{text}",
        ids.len(),
        t_prep.as_secs_f64(),
        ids.len() as f64 / t_prefill.as_secs_f64(),
        out.len(),
        out.len() as f64 / t_decode.as_secs_f64(),
    );
    text
}

#[test]
fn gemma4_text_answers_facts() {
    let Some(m) = model() else { return };
    let a = ask(
        &*m,
        "What is the capital of France? Answer with one word.".into(),
        &[],
        16,
    );
    assert!(a.trim().starts_with("Paris"), "{a}");
    let a = ask(
        &*m,
        "Name the largest planet in the solar system in one word.".into(),
        &[],
        16,
    );
    assert!(a.trim().starts_with("Jupiter"), "{a}");
}

/// Past the 512-token sliding window: chunked prefill (the scheduler's path, which trims the
/// sliding caches) must match one-shot prefill, and a fact from the first tokens must survive.
#[test]
fn gemma4_long_prompt_chunked_matches_one_shot() {
    let Some(m) = model() else { return };
    let tok = m.tokenizer();
    let filler: String = (0..90)
        .map(|i| format!("Line {i}: the river carried leaves past the old mill while birds sang. "))
        .collect();
    let text = format!("Remember this: the secret code is 4721.\n{filler}\nWhat is the secret code? Answer with the number only.");
    let prompt = tok.apply_chat_template_messages(&[ChatMessage::new("user", &text)], None);
    let (ids, mut one) = m.prepare(tok.encode(&prompt), &[]).unwrap();
    assert!(ids.len() > 1100, "prompt is only {} tokens", ids.len());
    let a = m.forward(&ids, 0, &mut one).unwrap();
    let (_, mut chunked) = m.prepare(ids.clone(), &[]).unwrap();
    let mut b = Vec::new();
    for (i, c) in ids.chunks(200).enumerate() {
        b = m.forward(c, i * 200, &mut chunked).unwrap();
    }
    let argmax = |l: &[f32]| {
        l.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0
    };
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(argmax(&a), argmax(&b), "max logit diff {diff}");
    assert!(diff < 0.25, "max logit diff {diff}");
    let mut out = vec![argmax(&b) as u32];
    while out.len() < 6 && !tok.is_special(*out.last().unwrap()) {
        let l = m
            .forward(
                &out[out.len() - 1..],
                ids.len() + out.len() - 1,
                &mut chunked,
            )
            .unwrap();
        out.push(argmax(&l) as u32);
    }
    let answer = tok.decode(&out);
    assert!(answer.contains("4721"), "{answer}");
}
#[test]
fn gemma4_image_is_recognised() {
    let Some(m) = model() else { return };
    let path = "/tmp/gibbs-testimg/cat.png";
    let Ok(png) = std::fs::read(path) else {
        println!("SKIP: {path} missing");
        return;
    };
    let parts = serde_json::json!([
        {"type": "image"},
        {"type": "text", "text": "What animal is in this picture? Answer with one word."}
    ]);
    let a = ask(&*m, parts, &[MediaInput::Image(png)], 16);
    assert!(a.trim().to_lowercase().starts_with("cat"), "{a}");
}

#[test]
fn gemma4_speech_is_understood() {
    let Some(m) = model() else { return };
    // A 22.05 kHz clip, so the resampler is on the path too.
    let path = std::env::temp_dir().join("gibbs_gemma4_speech.wav");
    let said = std::process::Command::new("say")
        .args(["-o", path.to_str().unwrap(), "--data-format=LEI16@22050"])
        .arg("My favourite fruit is a banana, and I eat one every day.")
        .status();
    if !matches!(said, Ok(s) if s.success()) {
        println!("SKIP: macOS `say` unavailable to synthesise speech");
        return;
    }
    let wav = std::fs::read(&path).unwrap();
    let parts = serde_json::json!([
        {"type": "audio"},
        {"type": "text", "text": "Which fruit does the speaker mention? Answer with one word."}
    ]);
    let a = ask(&*m, parts, &[MediaInput::Audio(wav)], 16);
    assert!(a.trim().to_lowercase().starts_with("banana"), "{a}");
}

/// The llama.cpp `gemma4` GGUF of google/gemma-4-E2B-it with its `mmproj` (ggml-org/gemma-4-E2B-it-GGUF:
/// `gemma-4-E2B-it-Q4_0.gguf` + `mmproj-gemma-4-E2B-it-Q8_0.gguf`), from `GIBBS_GEMMA4_GGUF_DIR`
/// (default: that repo's snapshot under `disk/` or `GIBBS_TEST_STORAGE`).
fn load_gguf(device: Device) -> Option<Arc<dyn NativeLm>> {
    let snapshot = |root: String| {
        PathBuf::from(root).join("models--ggml-org--gemma-4-E2B-it-GGUF/snapshots/main")
    };
    let dir = std::env::var("GIBBS_GEMMA4_GGUF_DIR")
        .map(PathBuf::from)
        .ok()
        .into_iter()
        .chain(["disk".to_string()].into_iter().map(snapshot))
        .chain(std::env::var("GIBBS_TEST_STORAGE").map(snapshot))
        .find(|d| d.join("gemma-4-E2B-it-Q4_0.gguf").exists());
    let Some(dir) = dir else {
        println!("SKIP: no ggml-org/gemma-4-E2B-it-GGUF snapshot (set GIBBS_GEMMA4_GGUF_DIR)");
        return None;
    };
    let artifacts = ModelArtifacts {
        model_id: "ggml-org/gemma-4-E2B-it-GGUF".into(),
        weights_path: Some(dir.join("gemma-4-E2B-it-Q4_0.gguf")),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, device)
            .expect("gemma4 GGUF is native")
            .expect("gemma4 GGUF loads"),
    )
}

static GGUF_MODEL: LazyLock<Option<Arc<dyn NativeLm>>> = LazyLock::new(|| load_gguf(Device::Cpu));

/// Next-token logits after "The capital of France is" against llama.cpp (build 11304,
/// `llama-debug --save-logits` on the same GGUF): same ids, same top tokens, close logits.
fn check_against_llama_cpp(m: &dyn NativeLm) {
    let ids = m.tokenizer().encode("The capital of France is");
    assert_eq!(
        ids,
        [2, 818, 5279, 529, 7001, 563],
        "tokenization differs from llama.cpp"
    );
    let (ids, mut state) = m.prepare(ids, &[]).unwrap();
    let logits = m.forward(&ids, 0, &mut state).unwrap();
    let mut top: Vec<usize> = (0..logits.len()).collect();
    top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let reference = [
        (9079, 25.271),
        (5213, 23.431),
        (7001, 19.392),
        (506, 19.030),
        (236743, 18.164),
    ];
    let got: Vec<(usize, f32)> = top[..5].iter().map(|&i| (i, logits[i])).collect();
    println!("ours {got:?}\nllama.cpp {reference:?}");
    assert_eq!(top[0], reference[0].0);
    for (i, want) in reference {
        assert!(
            (logits[i] - want).abs() < 0.25,
            "token {i}: {} vs llama.cpp {want}",
            logits[i]
        );
    }
}

#[test]
fn gemma4_gguf_matches_llama_cpp() {
    let Some(m) = GGUF_MODEL.clone() else { return };
    check_against_llama_cpp(&*m);
}

/// The Metal program (generated/hand GEMV, ext attention) against the same llama.cpp reference.
#[test]
fn gemma4_gguf_metal_matches_llama_cpp() {
    if !gibbs_engine::hardware::available_devices().contains(&Device::Metal) {
        return;
    }
    let Some(m) = load_gguf(Device::Metal) else {
        return;
    };
    check_against_llama_cpp(&*m);
}

/// `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ2_XXS (a sparse MoE: 128 experts, top-8) from the hub snapshot under
/// `disk/` (or `GIBBS_GEMMA4_MOE_DIR`); the tests skip when the 9.9 GB file is not there.
fn load_moe_gguf(device: Device) -> Option<Arc<dyn NativeLm>> {
    let dir = std::env::var("GIBBS_GEMMA4_MOE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("disk/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/main")
        });
    let file = dir.join("gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf");
    if !file.exists() {
        println!(
            "SKIP: no gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf in {}",
            dir.display()
        );
        return None;
    }
    let artifacts = ModelArtifacts {
        model_id: "unsloth/gemma-4-26B-A4B-it-GGUF".into(),
        weights_path: Some(file),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, device)
            .expect("gemma4 MoE GGUF is native")
            .expect("gemma4 MoE GGUF loads"),
    )
}

/// Next-token top-5 after "The capital of France is" on `m`.
fn moe_top5(m: &dyn NativeLm) -> Vec<(usize, f32)> {
    let ids = m.tokenizer().encode("The capital of France is");
    assert_eq!(
        ids,
        [2, 818, 5279, 529, 7001, 563],
        "tokenization differs from llama.cpp"
    );
    let (ids, mut state) = m.prepare(ids, &[]).unwrap();
    let logits = m.forward(&ids, 0, &mut state).unwrap();
    let mut top: Vec<usize> = (0..logits.len()).collect();
    top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    top[..5].iter().map(|&i| (i, logits[i])).collect()
}

/// The Metal MoE program (indexed expert GEMVs on IQ2_XXS banks) against llama.cpp build 11304
/// (`llama-debug --save-logits` on the same file): same top token and ranking, close logits.
#[test]
fn gemma4_moe_gguf_metal_matches_llama_cpp() {
    if !gibbs_engine::hardware::available_devices().contains(&Device::Metal) {
        return;
    }
    let Some(m) = load_moe_gguf(Device::Metal) else {
        return;
    };
    let reference = [
        (789usize, 5.443f32),
        (3003, 5.001),
        (10463, 4.041),
        (1847, 3.789),
        (2395, 3.778),
    ];
    let got = moe_top5(&*m);
    println!("ours {got:?}\nllama.cpp {reference:?}");
    // The top three agree in order; ranks 4 and 5 are a 0.011 tie in llama.cpp, so only their set is pinned.
    assert_eq!(
        got[..3].iter().map(|g| g.0).collect::<Vec<_>>(),
        reference[..3].iter().map(|r| r.0).collect::<Vec<_>>()
    );
    let mut ids: Vec<usize> = got.iter().map(|g| g.0).collect();
    ids.sort();
    let mut want_ids: Vec<usize> = reference.iter().map(|r| r.0).collect();
    want_ids.sort();
    assert_eq!(ids, want_ids);
    for (i, want) in reference {
        let g = got.iter().find(|x| x.0 == i).unwrap().1;
        assert!((g - want).abs() < 0.1, "token {i}: {g} vs llama.cpp {want}");
    }
}
#[test]
fn gemma4_gguf_answers_text_image_and_speech() {
    let Some(m) = GGUF_MODEL.clone() else { return };
    let a = ask(
        &*m,
        "What is the capital of France? Answer with one word.".into(),
        &[],
        16,
    );
    assert!(a.trim().starts_with("Paris"), "{a}");
    if let Ok(png) = std::fs::read("/tmp/gibbs-testimg/cat.png") {
        let parts = serde_json::json!([
            {"type": "image"},
            {"type": "text", "text": "What animal is in this picture? Answer with one word."}
        ]);
        let a = ask(&*m, parts, &[MediaInput::Image(png)], 16);
        assert!(a.trim().to_lowercase().starts_with("cat"), "{a}");
    } else {
        println!("SKIP: /tmp/gibbs-testimg/cat.png missing");
    }
    let path = std::env::temp_dir().join("gibbs_gemma4_gguf_speech.wav");
    let said = std::process::Command::new("say")
        .args(["-o", path.to_str().unwrap(), "--data-format=LEI16@22050"])
        .arg("My favourite fruit is a banana, and I eat one every day.")
        .status();
    if matches!(said, Ok(s) if s.success()) {
        let parts = serde_json::json!([
            {"type": "audio"},
            {"type": "text", "text": "Which fruit does the speaker mention? Answer with one word."}
        ]);
        let a = ask(
            &*m,
            parts,
            &[MediaInput::Audio(std::fs::read(&path).unwrap())],
            16,
        );
        assert!(a.trim().to_lowercase().starts_with("banana"), "{a}");
    } else {
        println!("SKIP: macOS `say` unavailable to synthesise speech");
    }
}

/// A Gemma 4 GGUF with an `mmproj` beside it, loaded on `device` (`GIBBS_GEMMA4_MM_DIR`: a hub
/// snapshot such as `unsloth/gemma-4-E2B-it-qat-GGUF` holding `*.gguf` + `mmproj-*.gguf`).
fn load_mm_gguf(device: Device) -> Option<Arc<dyn NativeLm>> {
    let dir = PathBuf::from(std::env::var("GIBBS_GEMMA4_MM_DIR").ok()?);
    let gguf = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.ends_with(".gguf") && !n.starts_with("mmproj")
        })?;
    let artifacts = ModelArtifacts {
        model_id: "gemma4-mm-gguf".into(),
        weights_path: Some(gguf),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, device)
            .expect("gemma4 GGUF is native")
            .expect("gemma4 GGUF loads"),
    )
}

/// Image and speech prompts through the Metal vision / conformer programs and the GPU row scatter
/// answer like the host forward on the same weights (set `GIBBS_GEMMA4_MM_DIR`; image is a red disc
/// on white, speech is synthesised with `say`).
#[test]
fn gemma4_gguf_metal_text_image_and_speech() {
    if !gibbs_engine::hardware::available_devices().contains(&Device::Metal) {
        return;
    }
    let (Some(gpu), Some(cpu)) = (load_mm_gguf(Device::Metal), load_mm_gguf(Device::Cpu)) else {
        println!("SKIP: set GIBBS_GEMMA4_MM_DIR to a gemma-4 GGUF snapshot with an mmproj");
        return;
    };
    assert_eq!(gpu.device(), Device::Metal);
    let mut img = gibbs_engine::model::image::ImageBuffer::new(224, 224);
    for (i, px) in img.data.chunks_exact_mut(3).enumerate() {
        let (x, y) = ((i % 224) as f32 - 112.0, (i / 224) as f32 - 112.0);
        px.copy_from_slice(if x * x + y * y < 70.0 * 70.0 {
            &[220, 20, 20]
        } else {
            &[255, 255, 255]
        });
    }
    let parts = serde_json::json!([
        {"type": "image"},
        {"type": "text", "text": "What colour is the disc in this picture? Answer with one word."}
    ]);
    let media = [MediaInput::Image(img.to_png())];
    let g = ask(&*gpu, parts.clone(), &media, 8);
    let c = ask(&*cpu, parts, &media, 8);
    println!("image: metal {g:?} cpu {c:?}");
    assert_eq!(g, c);
    assert!(g.trim().to_lowercase().starts_with("red"), "{g}");
    let path = std::env::temp_dir().join("gibbs_gemma4_mm_speech.wav");
    let said = std::process::Command::new("say")
        .args(["-o", path.to_str().unwrap(), "--data-format=LEI16@22050"])
        .arg("My favourite fruit is a banana, and I eat one every day.")
        .status();
    if matches!(said, Ok(s) if s.success()) {
        let parts = serde_json::json!([
            {"type": "audio"},
            {"type": "text", "text": "Which fruit does the speaker mention? Answer with one word."}
        ]);
        let media = [MediaInput::Audio(std::fs::read(&path).unwrap())];
        let g = ask(&*gpu, parts.clone(), &media, 8);
        let c = ask(&*cpu, parts, &media, 8);
        println!("speech: metal {g:?} cpu {c:?}");
        assert_eq!(g, c);
        assert!(g.trim().to_lowercase().starts_with("banana"), "{g}");
    } else {
        println!("SKIP: macOS `say` unavailable to synthesise speech");
    }
}
