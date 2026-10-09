//! Draft-model speculative decoding on real weights from the shared `disk/` cache: hidden-state
//! drafters (DFlash, DFlash2, DSpark) and a plain same-tokenizer draft model keep greedy output
//! token-identical to plain decoding and get their drafts accepted, on a native target
//! (Qwen3.5-2B) and a GraphLm one (Llama-3.2-1B-Instruct); mismatched pairs are refused at attach.

use gibbs_engine::hardware::{Device, QuantMode};
use gibbs_engine::kv::PagedKVPool;
use gibbs_engine::model::draft::load_drafter;
use gibbs_engine::model::hf_loader::load_causal_lm;
use gibbs_engine::model::hub::{HuggingFaceHub, ModelArtifacts};
use gibbs_engine::model::native::load_native;
use gibbs_engine::scheduler::{ContinuousBatchScheduler, ExecutableModel, SamplingParams};
use gibbs_engine::telemetry::EngineTelemetry;
use std::sync::Arc;

const TARGET: &str = "Qwen/Qwen3.5-2B";
const PROMPT: &str = "Write a Python function that returns the n-th Fibonacci number iteratively.";

fn located(repo: &str) -> Option<ModelArtifacts> {
    let a = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if a.weights_path.is_none() {
        eprintln!("SKIP: {repo} not in disk/ (gibbs --model {repo} --storage disk)");
        return None;
    }
    Some(a)
}

/// Greedy output of `prompt` (chat-templated) and the (drafted, accepted) counts.
fn generate(
    target: &ModelArtifacts,
    draft: Option<&ModelArtifacts>,
    prompt: &str,
    n: usize,
) -> (Vec<u32>, u64, u64) {
    let model = match load_native(target, Device::Cpu) {
        Some(m) => ExecutableModel::Native(m.unwrap()),
        None => ExecutableModel::from(load_causal_lm(target, QuantMode::Uncompressed).unwrap()),
    };
    let telemetry = EngineTelemetry::new();
    let sched = ContinuousBatchScheduler::with_telemetry(
        target.model_id.clone(),
        Arc::new(PagedKVPool::new(1024)),
        Some(model),
        telemetry.clone(),
    );
    // The checkpoint's own MTP head stays off: the baseline is plain decoding.
    sched.set_drafter(draft.map(|d| load_drafter(d, target).unwrap()));
    let params = SamplingParams {
        temperature: Some(0.0),
        top_p: None,
        max_new_tokens: n,
        top_logprobs: None,
    };
    let (_, mut rx, ..) = sched.submit_with_params(params, |t| {
        t.encode(&t.apply_chat_template(&[("user", prompt)]))
    });
    let mut out = Vec::new();
    loop {
        sched.step();
        loop {
            match rx.try_recv() {
                Ok(t) => out.push(t),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(_) => {
                    let s = telemetry.get_summary();
                    return (out, s.spec_drafted, s.spec_accepted);
                }
            }
        }
    }
}

/// Every drafter in `drafts` (repo, least acceptance rate) keeps `target`'s greedy output.
fn check(target: &str, drafts: &[(&str, f64)]) {
    let Some(target) = located(target) else {
        return;
    };
    let (plain, drafted, _) = generate(&target, None, PROMPT, 32);
    assert_eq!(drafted, 0);
    assert!(plain.len() > 16, "plain output {plain:?}");
    for &(repo, min_rate) in drafts {
        let Some(draft) = located(repo) else { continue };
        let (out, drafted, accepted) = generate(&target, Some(&draft), PROMPT, 32);
        let rate = accepted as f64 / drafted.max(1) as f64;
        eprintln!("{repo}: {accepted}/{drafted} drafts accepted ({rate:.2})");
        assert_eq!(out, plain, "{repo} changed the greedy output");
        // A drafter with wrong numerics still verifies to the same text, but almost nothing
        // it proposes is accepted.
        assert!(
            drafted > 0 && rate >= min_rate,
            "{repo}: acceptance {rate:.2}"
        );
    }
}

#[test]
fn qwen3_5_drafters_keep_greedy_output_and_get_accepted() {
    check(
        TARGET,
        &[
            ("taobao-mnn/Qwen3.5-2B-Dflash", 0.4),
            ("rasyosef/Qwen3.5-2B-DSpark", 0.4),
            ("rasyosef/Qwen3.5-2B-DFlash2", 0.4),
            ("Qwen/Qwen3.5-0.8B", 0.4),
        ],
    );
}

/// Taps from GraphLm (llama) feed a DSpark drafter with llama3 rope and a pruned draft vocab.
#[test]
fn llama_dspark_keeps_greedy_output_and_gets_accepted() {
    check(
        "unsloth/Llama-3.2-1B-Instruct",
        &[("rasyosef/Llama-3.2-1B-Instruct-DSpark", 0.4)],
    );
}

#[test]
fn drafter_for_another_target_is_refused() {
    let (Some(small), Some(dspark)) = (
        located("Qwen/Qwen3.5-0.8B"),
        located("rasyosef/Qwen3.5-2B-DSpark"),
    ) else {
        return;
    };
    let err = load_drafter(&dspark, &small)
        .err()
        .expect("2B drafter on the 0.8B target");
    assert!(
        err.contains("drafter hidden 2048 != target embedding width 1024"),
        "{err}"
    );
    let Some(tiny) = located("HuggingFaceTB/SmolLM2-135M-Instruct") else {
        return;
    };
    let err = load_drafter(&tiny, &small)
        .err()
        .expect("different tokenizer");
    assert!(err.contains("tokenizes differently"), "{err}");
}
