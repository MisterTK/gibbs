//! Speculative decoding with Qwen3.5-0.8B's multi-token-prediction head on real weights from the
//! shared `disk/` cache: the Gated DeltaNet / conv state rewinds exactly, and greedy output through
//! the scheduler (concurrent sequences) is token for token the same with and without MTP.

use gibbs_engine::hardware::Device;
use gibbs_engine::kv::PagedKVPool;
use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::native::{load_native, NativeLm};
use gibbs_engine::scheduler::{ContinuousBatchScheduler, ExecutableModel, SamplingParams};
use std::sync::Arc;

const REPO: &str = "Qwen/Qwen3.5-0.8B";

fn model() -> Option<Arc<dyn NativeLm>> {
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(REPO);
    if art.weights_path.is_none() {
        eprintln!("SKIP: {REPO} not in disk/ (gibbs --model {REPO} --storage disk)");
        return None;
    }
    Some(load_native(&art, Device::Cpu)?.expect("Qwen3.5-0.8B loads"))
}

/// Verifying four positions, rewinding to two, and continuing gives bit-identical logits to
/// running those two positions one token at a time: the recurrent state, conv window and KV
/// caches are restored exactly. Rewinding outside the verified span errs.
#[test]
fn deltanet_rollback_is_exact() {
    let Some(m) = model() else { return };
    let tok = m.tokenizer();
    let prompt = tok.encode("The quick brown fox jumps over the lazy dog, and then");
    let t = tok.encode(" it ran into the forest to hide");
    let n = prompt.len();

    let (_, mut a) = m.prepare(prompt.clone(), &[]).unwrap();
    m.forward(&prompt, 0, &mut a).unwrap();
    // Two right tokens, two wrong ones.
    m.forward_all(&[t[0], t[1], t[5], t[6]], n, &mut a).unwrap();
    m.truncate(&mut a, n + 2).unwrap();
    let got = m.forward_all(&[t[2], t[3]], n + 2, &mut a).unwrap();

    let (_, mut b) = m.prepare(prompt.clone(), &[]).unwrap();
    m.forward(&prompt, 0, &mut b).unwrap();
    m.forward(&[t[0]], n, &mut b).unwrap();
    m.forward(&[t[1]], n + 1, &mut b).unwrap();
    let want = m.forward_all(&[t[2], t[3]], n + 2, &mut b).unwrap();
    assert!(got == want, "rewound state diverges from the plain run");

    // The last forward_all spans n+2..n+4: n+1 lies before it, n+5 past the sequence.
    assert!(m.truncate(&mut a, n + 1).is_err());
    assert!(m.truncate(&mut a, n + 5).is_err());
    // A plain forward drops the checkpoints: nothing can be rewound any more.
    m.forward(&[t[4]], n + 4, &mut a).unwrap();
    assert!(m.truncate(&mut a, n + 4).is_err());
}

/// Greedy output of three concurrent requests: with the MTP drafter on (default) and off.
fn run(m: &Arc<dyn NativeLm>, spec: bool, prompts: &[&str], max: usize) -> (Vec<Vec<u32>>, f32) {
    let sched = ContinuousBatchScheduler::new(
        REPO,
        Arc::new(PagedKVPool::new(4096)),
        Some(ExecutableModel::Native(m.clone())),
    );
    if !spec {
        sched.set_drafter(None);
    }
    let params = SamplingParams {
        temperature: Some(0.0),
        top_p: None,
        max_new_tokens: max,
        top_logprobs: None,
    };
    let mut rxs: Vec<_> = prompts
        .iter()
        .map(|p| {
            sched
                .submit_with_params(params, |t| t.encode(&t.apply_chat_template(&[("user", p)])))
                .1
        })
        .collect();
    while sched.step().active_sequences_count > 0 || !sched.waiting_queue.read().is_empty() {}
    let out = rxs
        .iter_mut()
        .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
        .collect();
    let s = sched.telemetry.get_summary();
    assert_eq!(s.spec_forwards > 0, spec, "speculation on = {spec}");
    (out, s.spec_accept_rate)
}

#[test]
fn mtp_greedy_matches_plain_decoding() {
    let Some(m) = model() else { return };
    let prompts = [
        "Write a long story about a dragon.",
        "Explain how hash maps work.",
        "Describe the history of Rome.",
    ];
    // All prompts prefill in the scheduler's first step (its first budget is 64 tokens), so both
    // runs split prefill identically and only decoding differs.
    let tok = m.tokenizer();
    let total: usize = prompts
        .iter()
        .map(|p| tok.encode(&tok.apply_chat_template(&[("user", p)])).len())
        .sum();
    assert!(total <= 64, "prompts take {total} tokens");
    let (plain, _) = run(&m, false, &prompts, 200);
    let (spec, accept) = run(&m, true, &prompts, 200);
    for (i, (a, b)) in plain.iter().zip(&spec).enumerate() {
        assert_eq!(a.len(), 200, "prompt {i}: only {} tokens", a.len());
        assert_eq!(a, b, "prompt {i}: MTP output differs");
    }
    // The head drafts from the model's own hidden states: most drafts should land.
    assert!(accept > 0.4, "MTP acceptance {accept}");
}

/// The scheduler's sampling distribution (top-k 50, softmax at `temp`, top-p 0.9 cut).
fn target_dist(logits: &[f32], temp: f32) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.truncate(50);
    let max = logits[idx[0] as usize] / temp;
    let e: Vec<f32> = idx
        .iter()
        .map(|&i| (logits[i as usize] / temp - max).exp())
        .collect();
    let sum: f32 = e.iter().sum();
    let (mut out, mut cum) = (Vec::new(), 0.0);
    for (&i, &w) in idx.iter().zip(&e) {
        out.push((i, w / sum));
        cum += w / sum;
        if cum >= 0.9 {
            break;
        }
    }
    let kept: f32 = out.iter().map(|e| e.1).sum();
    out.iter().map(|&(i, p)| (i, p / kept)).collect()
}

/// Sampling with MTP drafts (temperature 1): the second output token, which the first
/// speculative step emits through rejection sampling, follows the target's own distribution.
#[test]
fn mtp_sampling_keeps_target_distribution() {
    let Some(m) = model() else { return };
    let ids = m.tokenizer().encode("My favourite colour is");
    // Room for one speculative step (k = 1) after the first token.
    let params = SamplingParams {
        temperature: Some(1.0),
        top_p: None,
        max_new_tokens: 3,
        top_logprobs: None,
    };
    let n = 200;
    let mut first = None;
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for _ in 0..n {
        // A fresh scheduler's k tuner starts at k = 1, so every sample takes a speculative step.
        let sched = ContinuousBatchScheduler::new(
            REPO,
            Arc::new(PagedKVPool::new(64)),
            Some(ExecutableModel::Native(m.clone())),
        );
        let mut rx = sched.submit_with_params(params, |_| ids.clone()).1;
        while sched.step().active_sequences_count > 0 {}
        let out: Vec<u32> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(out.len() >= 2, "{out:?}");
        assert_eq!(sched.telemetry.get_summary().spec_forwards, 1);
        // The first token comes from prefill's deterministic sampler: the same every time.
        assert_eq!(*first.get_or_insert(out[0]), out[0]);
        *counts.entry(out[1]).or_default() += 1;
    }
    let (ctx, mut st) = m
        .prepare([ids.clone(), vec![first.unwrap()]].concat(), &[])
        .unwrap();
    let logits = m.forward(&ctx, 0, &mut st).unwrap();
    let p = target_dist(&logits[..m.tokenizer().vocab_size().min(logits.len())], 1.0);
    eprintln!("target p {p:?}\nsampled {counts:?}");
    for t in counts.keys() {
        assert!(
            p.iter().any(|e| e.0 == *t),
            "token {t} is outside the target's nucleus"
        );
    }
    for &(t, pt) in &p {
        let f = counts.get(&t).copied().unwrap_or(0) as f32 / n as f32;
        let sd = (pt * (1.0 - pt) / n as f32).sqrt();
        assert!(
            (f - pt).abs() <= 4.0 * sd + 0.01,
            "token {t}: {f} sampled vs p {pt}"
        );
    }
}
