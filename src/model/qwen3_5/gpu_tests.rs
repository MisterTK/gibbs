//! The Qwen3.5 Metal programs (decoder with image positions and deepstack, vision tower, MTP head,
//! verify/rewind) against the host forward, on tiny synthetic checkpoints.
//!
//! Tolerance: both sides read the same bf16 weights; the host accumulates f32 in NEON tiles, the
//! GPU in f32 simdgroups, so logits agree to a few 1e-5 relative to their scale: bound 1e-3.

use super::fixture::{checkpoint, H, IMAGE_TOKEN, V};
use super::*;
use crate::hardware::{cpu_compute, PlacementMode};
use crate::model::native::MediaInput;

const PRE: &str = r#"{"size":{"shortest_edge":64,"longest_edge":100000},
    "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;

fn tokenizer() -> Tokenizer {
    Tokenizer::from_tokens((0..V).map(|i| format!("t{i}")).collect())
}

/// Host and Metal models over one checkpoint.
fn pair(vision: bool, mtp: bool) -> (Arc<dyn NativeLm>, Arc<dyn NativeLm>) {
    let (ck, cfg) = checkpoint(vision, mtp);
    let build = |mode: Device| {
        let text = Arc::new(TextModel::load(&ck, text_config(&cfg), Arch::Qwen35).unwrap());
        let vis = load_vision(&ck, &cfg, &text, || {
            Preprocessor::parse(PRE.as_bytes(), "test")
        })
        .unwrap();
        assemble(tokenizer(), text, vis, PlacementMode::Pin(mode)).unwrap()
    };
    (build(Device::Cpu), build(Device::Metal))
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn png(w: usize, h: usize) -> Vec<u8> {
    let mut img = ImageBuffer::new(w, h);
    for (i, px) in img.data.iter_mut().enumerate() {
        *px = ((i * 37 + (i / 7) * 11) % 251) as u8;
    }
    img.to_png()
}

/// One image, text before and after; six prompt-side forwards then four decode steps.
#[test]
fn image_prompt_program_matches_host() {
    let _g = cpu_compute::allow();
    let (cpu, gpu) = pair(true, false);
    let ids: Vec<u32> = vec![3, 7, IMAGE_TOKEN, 11, 5, 9];
    let media = [MediaInput::Image(png(16, 24))];
    let (ids_c, mut sc) = cpu.prepare(ids.clone(), &media).unwrap();
    let (ids_g, mut sg) = gpu.prepare(ids, &media).unwrap();
    assert_eq!(ids_c, ids_g);
    assert!(ids_c.len() > 20, "image expanded to its 4x6 merged grid");
    let (lc, lg) = (
        cpu.forward(&ids_c, 0, &mut sc).unwrap(),
        gpu.forward(&ids_g, 0, &mut sg).unwrap(),
    );
    let e = rel(&lc, &lg);
    println!("prefill (image) rel err {e:e}");
    assert!(e < 1e-3, "{e}");
    assert_eq!(argmax(&lc), argmax(&lg));
    let (mut tok, mut pos) = (argmax(&lc), ids_c.len());
    for step in 0..4 {
        let (a, b) = (
            cpu.forward(&[tok], pos, &mut sc).unwrap(),
            gpu.forward(&[tok], pos, &mut sg).unwrap(),
        );
        let e = rel(&a, &b);
        println!("decode {step} rel err {e:e}");
        assert!(e < 1e-3, "decode {step}: {e}");
        tok = argmax(&a);
        pos += 1;
    }
}

/// The vision tower program alone (merged features and the deepstack feature).
#[test]
fn vision_tower_program_matches_host() {
    let _g = cpu_compute::allow();
    let (ck, cfg) = checkpoint(true, false);
    let pre = Preprocessor::parse(PRE.as_bytes(), "test").unwrap();
    let host = Vision::load(&ck, &cfg["vision_config"]).unwrap();
    let dev = Vision::load(&ck, &cfg["vision_config"])
        .unwrap()
        .on_gpu(PlacementMode::Pin(Device::Metal))
        .unwrap();
    for (w, h) in [(16, 24), (24, 8)] {
        let img = ImageBuffer::decode(&png(w, h)).unwrap();
        let (a, b) = (
            host.encode(&img, &pre).unwrap(),
            dev.encode(&img, &pre).unwrap(),
        );
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        assert_eq!(a.deepstack.len(), 1);
        let (e, ed) = (
            rel(&a.embeds, &b.embeds),
            rel(&a.deepstack[0], &b.deepstack[0]),
        );
        println!("{w}x{h}: embeds rel err {e:e}, deepstack {ed:e}");
        assert!(e < 1e-3 && ed < 1e-3, "{w}x{h}: {e} {ed}");
    }
}

/// Verify four positions, rewind to two, continue: bit-identical to running them one at a time
/// (the Metal-resident recurrent state and conv window come back from the snapshot); rewinding
/// outside the verified span errs; a plain forward drops the checkpoints.
#[test]
fn verify_rewind_is_exact_on_metal() {
    let (_, gpu) = pair(false, false);
    let prompt = vec![4u32, 8, 15, 16, 23];
    let n = prompt.len();
    let (_, mut a) = gpu.prepare(prompt.clone(), &[]).unwrap();
    gpu.forward(&prompt, 0, &mut a).unwrap();
    gpu.forward_all(&[30, 31, 50, 51], n, &mut a).unwrap();
    gpu.truncate(&mut a, n + 2).unwrap();
    let got = gpu.forward_all(&[32, 33], n + 2, &mut a).unwrap();

    let (_, mut b) = gpu.prepare(prompt.clone(), &[]).unwrap();
    gpu.forward(&prompt, 0, &mut b).unwrap();
    gpu.forward(&[30], n, &mut b).unwrap();
    gpu.forward(&[31], n + 1, &mut b).unwrap();
    let want = gpu.forward_all(&[32, 33], n + 2, &mut b).unwrap();
    assert!(got == want, "rewound state diverges from the plain run");

    assert!(
        gpu.truncate(&mut a, n + 1).is_err(),
        "before the verified span"
    );
    assert!(gpu.truncate(&mut a, n + 5).is_err(), "past the sequence");
    gpu.forward(&[34], n + 4, &mut a).unwrap();
    assert!(gpu.truncate(&mut a, n + 4).is_err(), "checkpoints dropped");
}

/// MTP head: the Metal program drafts the same tokens from the same logits as the host head,
/// across a catch-up call after a partial accept.
#[test]
fn mtp_drafter_program_matches_host() {
    let _g = cpu_compute::allow();
    let (cpu, gpu) = pair(false, true);
    let (dc, dg) = (
        cpu.drafter().expect("host MTP"),
        gpu.drafter().expect("Metal MTP"),
    );
    assert_eq!(dc.tap_layers(), dg.tap_layers());
    let prompt = vec![4u32, 8, 15, 16, 23, 42];
    let (_, mut sc) = cpu.prepare(prompt.clone(), &[]).unwrap();
    let (_, mut sg) = gpu.prepare(prompt.clone(), &[]).unwrap();
    cpu.set_taps(&mut sc, &dc.tap_layers()).unwrap();
    gpu.set_taps(&mut sg, &dg.tap_layers()).unwrap();
    let (mut stc, mut stg) = (dc.new_state(), dg.new_state());
    let n = prompt.len();
    // The target has run all but the last token.
    cpu.forward(&prompt[..n - 1], 0, &mut sc).unwrap();
    gpu.forward(&prompt[..n - 1], 0, &mut sg).unwrap();
    let (hc, hg) = (cpu.take_taps(&mut sc), gpu.take_taps(&mut sg));
    let mut errs = Vec::new();
    let draw = |d: &Arc<dyn Drafter>,
                st: &mut crate::scheduler::spec::DraftState,
                toks: &[u32],
                h: &[f32]| {
        let mut seen = Vec::new();
        let out = d
            .propose(st, toks, h, 3, &mut |l: &[f32]| {
                seen.push(l.to_vec());
                argmax(l)
            })
            .unwrap();
        (out, seen)
    };
    let (oc, lc) = draw(&dc, &mut stc, &prompt, &hc);
    let (og, lg) = draw(&dg, &mut stg, &prompt, &hg);
    errs.extend(lc.iter().zip(&lg).map(|(a, b)| rel(a, b)));
    assert_eq!(oc, og);
    // Verify [last, d0, d1, d2], accept two, then ask again: one catch-up row per position run.
    let ver = [prompt[n - 1], oc[0], oc[1], oc[2]];
    cpu.forward_all(&ver, n - 1, &mut sc).unwrap();
    gpu.forward_all(&ver, n - 1, &mut sg).unwrap();
    cpu.truncate(&mut sc, n + 2).unwrap();
    gpu.truncate(&mut sg, n + 2).unwrap();
    let (hc, hg) = (cpu.take_taps(&mut sc), gpu.take_taps(&mut sg));
    // Taps hold all four verified rows; the drafter only wants the accepted three.
    let d = H;
    let mut toks = prompt.clone();
    toks.extend([oc[0], oc[1], 77]);
    let (oc2, lc2) = draw(&dc, &mut stc, &toks, &hc[..3 * d]);
    let (og2, lg2) = draw(&dg, &mut stg, &toks, &hg[..3 * d]);
    errs.extend(lc2.iter().zip(&lg2).map(|(a, b)| rel(a, b)));
    println!("MTP draft logits rel err: {errs:?}");
    assert!(errs.iter().all(|&e| e < 1e-3), "{errs:?}");
    assert_eq!(oc2, og2);
}

/// Greedy output through the scheduler on the Metal program, with the MTP head drafting and
/// without: token for token the same (verify runs the same n = 1 program as plain decoding and
/// rejected drafts rewind the recurrent state from its snapshot).
#[test]
fn scheduler_mtp_speculation_on_metal_keeps_greedy_output() {
    use crate::kv::PagedKVPool;
    use crate::scheduler::{ContinuousBatchScheduler, ExecutableModel, SamplingParams};
    let (_, gpu) = pair(false, true);
    let run = |spec: bool| {
        let sched = ContinuousBatchScheduler::new(
            "tiny",
            Arc::new(PagedKVPool::new(256)),
            Some(ExecutableModel::Native(gpu.clone())),
        );
        if !spec {
            sched.set_drafter(None);
        }
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 40,
            top_logprobs: None,
        };
        let mut rx = sched
            .submit_with_params(params, |_| vec![4, 8, 15, 16, 23])
            .1;
        while sched.step().active_sequences_count > 0 || !sched.waiting_queue.read().is_empty() {}
        let out: Vec<u32> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        let s = sched.telemetry.get_summary();
        assert_eq!(s.spec_forwards > 0, spec, "speculation on = {spec}");
        (out, s.spec_accept_rate)
    };
    let (plain, _) = run(false);
    let (spec, accept) = run(true);
    println!(
        "MTP on Metal: {} tokens, acceptance {accept:.2}",
        plain.len()
    );
    assert_eq!(plain.len(), 40);
    assert_eq!(plain, spec);
}
