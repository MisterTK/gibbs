//! Qwen3.5-0.8B (hybrid Gated DeltaNet decoder + ViT) end to end on real weights from the shared
//! `disk/` cache: the model's own chat template, `prepare` with images, greedy decode.

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hf_loader::ChatMessage;
use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::native::{load_native, MediaInput, NativeLm};
use std::sync::Arc;

const REPO: &str = "Qwen/Qwen3.5-0.8B";

fn model() -> Option<Arc<dyn NativeLm>> {
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(REPO);
    if art.weights_path.is_none() {
        println!("SKIP: {REPO} not in disk/ (gibbs --model {REPO} --storage disk)");
        return None;
    }
    Some(load_native(&art, Device::Cpu)?.expect("Qwen3.5-0.8B loads"))
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
    let Some(m) = model() else { return };
    let text = ask(m.as_ref(), "What is the capital of France?", vec![], 30);
    assert!(text.contains("Paris"), "factual answer: {text:?}");

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

    // Contract errors: placeholder/media mismatch, audio, and rewinding the recurrent state.
    let tok = m.tokenizer();
    let ids = tok.encode("<|vision_start|><|image_pad|><|vision_end|>hi");
    assert!(
        m.prepare(ids.clone(), &[]).is_err(),
        "placeholder without an image"
    );
    assert!(
        m.prepare(ids, &[MediaInput::Audio(vec![0; 4])]).is_err(),
        "audio is not an input"
    );
    let (ids, mut st) = m.prepare(tok.encode("hello"), &[]).unwrap();
    m.forward(&ids, 0, &mut st).unwrap();
    assert!(
        m.forward(&ids, 0, &mut st).is_err(),
        "recurrent state cannot rewind"
    );
}
