//! Qwen3.5-0.8B (hybrid Gated DeltaNet decoder + ViT) end to end on real weights from the shared
//! `disk/` cache: the model's own chat template, `prepare` with images, greedy decode. Without it,
//! the text checks run on the cached text-only Qwen3.5-2B GGUF (no vision tower).

use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::hf_loader::ChatMessage;
use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::native::{MediaInput, NativeLm};
use std::sync::Arc;

const REPO: &str = "Qwen/Qwen3.5-0.8B";
const TEXT_ONLY: &str = "unsloth/Qwen3.5-2B-GGUF";

/// The first cached of [`REPO`], [`TEXT_ONLY`], and whether it has the vision tower.
fn model() -> Option<(Arc<dyn NativeLm>, bool)> {
    let hub = HuggingFaceHub::with_cache_dir("disk".into());
    let Some((repo, art)) = [REPO, TEXT_ONLY]
        .into_iter()
        .map(|r| (r, hub.locate_model(r)))
        .find(|(_, a)| a.weights_path.is_some())
    else {
        println!(
            "SKIP: neither {REPO} nor {TEXT_ONLY} in disk/ (gibbs --model {REPO} --storage disk)"
        );
        return None;
    };
    let reg = gibbs_engine::arch::registry().unwrap();
    let m = gibbs_engine::runtime::lm::load(&art, reg, None, PlacementMode::Pin(Device::Cpu))
        .unwrap_or_else(|e| panic!("{repo} loads: {e}"));
    Some((m, repo == REPO))
}

/// Greedy answer to one user turn of `text` after `images`, prefilling in uneven chunks so
/// the recurrent/conv state is carried across `forward` calls.
fn ask(m: &dyn NativeLm, text: &str, images: Vec<Vec<u8>>, max: usize) -> String {
    let mut parts: Vec<_> = images
        .iter()
        .map(|_| serde_json::json!({"type": "image"}))
        .collect();
    parts.push(serde_json::json!({"type": "text", "text": text}));
    let msg = ChatMessage {
        role: "user".into(),
        content: serde_json::Value::Array(parts),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let tok = m.tokenizer();
    let prompt = tok.apply_chat_template_messages(&[msg], None);
    let media: Vec<_> = images.into_iter().map(MediaInput::Image).collect();
    let (ids, mut st) = m.prepare(tok.encode(&prompt), &media).unwrap();
    let mut logits = Vec::new();
    let mut pos = 0;
    for c in ids.chunks(37) {
        logits = m.forward(c, pos, &mut st).unwrap();
        pos += c.len();
    }
    let mut out = Vec::new();
    while out.len() < max {
        let next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        if tok.is_special(next) {
            break;
        }
        out.push(next);
        logits = m.forward(&[next], pos, &mut st).unwrap();
        pos += 1;
    }
    tok.decode(&out)
}

#[test]
fn qwen3_5_answers_text_and_image_and_refuses_bad_input() {
    let Some((m, vision)) = model() else { return };
    let text = ask(m.as_ref(), "What is the capital of France?", vec![], 30);
    assert!(text.contains("Paris"), "factual answer: {text:?}");

    let tok = m.tokenizer();
    let ids = tok.encode("<|vision_start|><|image_pad|><|vision_end|>hi");
    if !vision {
        println!("SKIP: image checks ({TEXT_ONLY} has no vision tower; needs {REPO})");
        // The spec refuses every media input on a text-only checkpoint.
        assert!(
            m.prepare(ids, &[MediaInput::Image(vec![0; 4])]).is_err(),
            "image on a text-only model"
        );
    } else {
        // A solid colour exercises the image processor, vision tower, placeholder expansion and
        // M-RoPE positions; a broken path answers with something other than the colour.
        let mut img = ImageBuffer::new(224, 160);
        for y in 0..160 {
            for x in 0..224 {
                img.set_pixel(x, y, 220, 20, 20);
            }
        }
        let answer = ask(
            m.as_ref(),
            "What color is this image? Answer with one word.",
            vec![img.to_png()],
            8,
        );
        assert!(
            answer.to_lowercase().contains("red"),
            "image answer: {answer:?}"
        );
        // Contract errors: placeholder/media mismatch and audio.
        assert!(
            m.prepare(ids.clone(), &[]).is_err(),
            "placeholder without an image"
        );
        assert!(
            m.prepare(ids, &[MediaInput::Audio(vec![0; 4])]).is_err(),
            "audio is not an input"
        );
    }
    // The recurrent state cannot rewind.
    let (ids, mut st) = m.prepare(tok.encode("hello"), &[]).unwrap();
    m.forward(&ids, 0, &mut st).unwrap();
    assert!(
        m.forward(&ids, 0, &mut st).is_err(),
        "recurrent state cannot rewind"
    );
}
