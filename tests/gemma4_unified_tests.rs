//! Gemma 4 unified (google/gemma-4-12B-it, `gemma4_unified`) real-weight checks: text, image and
//! speech prompts answered greedily through the model's chat template, and chunked prefill that
//! splits an image block (attended bidirectionally) matching one-shot prefill. Each test prints
//! `SKIP` when the checkpoint is not in `GIBBS_GEMMA4U_DIR` (default: the repo's `disk/` cache).
//! The checkpoint is 24 GB of bf16 (memory-mapped); run these one binary at a time.

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hf_loader::ChatMessage;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::model::native::{load_native, MediaInput, NativeLm};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

static MODEL: LazyLock<Option<Arc<dyn NativeLm>>> = LazyLock::new(|| {
    let dir = PathBuf::from(
        std::env::var("GIBBS_GEMMA4U_DIR")
            .unwrap_or_else(|_| "disk/models--google--gemma-4-12B-it/snapshots/main".into()),
    );
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: no Gemma 4 unified checkpoint in {}", dir.display());
        return None;
    }
    let artifacts = ModelArtifacts {
        model_id: "google/gemma-4-12B-it".into(),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        local_dir: Some(dir),
        is_cached: true,
        ..Default::default()
    };
    Some(
        load_native(&artifacts, Device::Cpu)
            .expect("gemma4_unified is native")
            .expect("gemma4_unified loads"),
    )
});

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn prompt(m: &dyn NativeLm, parts: serde_json::Value) -> Vec<u32> {
    let tok = m.tokenizer();
    let msg = ChatMessage {
        role: "user".into(),
        content: parts,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    tok.encode(&tok.apply_chat_template_messages(&[msg], None))
}

/// Greedy answer to one user turn whose content parts are `parts` (HF content-part JSON), after
/// checking the cheap `prompt_len` count against what `prepare` lays out.
fn ask(m: &dyn NativeLm, parts: serde_json::Value, media: &[MediaInput], max_new: usize) -> String {
    let tok = m.tokenizer();
    let ids = prompt(m, parts);
    let counted = m.prompt_len(ids.clone(), media).expect("prompt_len");
    let t0 = Instant::now();
    let (ids, mut state) = m.prepare(ids, media).expect("prepare");
    assert_eq!(counted, ids.len(), "prompt_len disagrees with prepare");
    let t_prep = t0.elapsed();
    let t1 = Instant::now();
    let mut logits = m.forward(&ids, 0, &mut state).expect("prefill");
    let t_prefill = t1.elapsed();
    let mut out = Vec::new();
    let t2 = Instant::now();
    while out.len() < max_new {
        let next = argmax(&logits);
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
    eprintln!(
        "[{} prompt tokens: media {:.2}s, prefill {:.1} tok/s; {} tokens decoded at {:.2} tok/s]\n{text}",
        ids.len(),
        t_prep.as_secs_f64(),
        ids.len() as f64 / t_prefill.as_secs_f64(),
        out.len(),
        out.len() as f64 / t_decode.as_secs_f64(),
    );
    text
}

#[test]
fn gemma4u_text_answers_facts() {
    let Some(m) = MODEL.clone() else { return };
    let a = ask(
        &*m,
        "What is the capital of France? Answer with one word.".into(),
        &[],
        8,
    );
    assert!(a.trim().starts_with("Paris"), "{a}");
}

/// The scheduler prefills in chunks that can end inside an image block; the block is
/// bidirectional on sliding layers, so the model must finish it before answering, and the
/// result must match one-shot prefill. Also checks the image is actually seen.
#[test]
fn gemma4u_image_chunked_matches_one_shot() {
    let Some(m) = MODEL.clone() else { return };
    let path = "/tmp/gibbs-testimg/cat.png";
    let Ok(png) = std::fs::read(path) else {
        eprintln!("SKIP: {path} missing");
        return;
    };
    let media = [MediaInput::Image(png)];
    let parts = serde_json::json!([
        {"type": "image"},
        {"type": "text", "text": "What animal is in this picture? Answer with one word."}
    ]);
    let a = ask(&*m, parts.clone(), &media, 8);
    assert!(a.trim().to_lowercase().starts_with("cat"), "{a}");

    let (ids, mut one) = m.prepare(prompt(&*m, parts.clone()), &media).unwrap();
    let full = m.forward(&ids, 0, &mut one).unwrap();
    let (_, mut chunked) = m.prepare(prompt(&*m, parts), &media).unwrap();
    let mut last = Vec::new();
    for (i, c) in ids.chunks(64).enumerate() {
        last = m.forward(c, i * 64, &mut chunked).unwrap();
    }
    let diff = full
        .iter()
        .zip(&last)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(argmax(&full), argmax(&last), "max logit diff {diff}");
    assert!(diff < 0.25, "max logit diff {diff}");
}

#[test]
fn gemma4u_speech_is_transcribed() {
    let Some(m) = MODEL.clone() else { return };
    // A 22.05 kHz clip, so the resampler is on the path too.
    let path = std::env::temp_dir().join("gibbs_gemma4u_speech.wav");
    let said = std::process::Command::new("say")
        .args(["-o", path.to_str().unwrap(), "--data-format=LEI16@22050"])
        .arg("The quick brown fox jumps over the lazy dog.")
        .status();
    if !matches!(said, Ok(s) if s.success()) {
        eprintln!("SKIP: macOS `say` unavailable to synthesise speech");
        return;
    }
    let wav = std::fs::read(&path).unwrap();
    let parts = serde_json::json!([
        {"type": "audio"},
        {"type": "text", "text": "Transcribe this audio."}
    ]);
    let a = ask(&*m, parts, &[MediaInput::Audio(wav)], 20).to_lowercase();
    assert!(
        a.contains("quick brown fox") && a.contains("lazy dog"),
        "{a}"
    );
}
