//! Golden outputs of every model family on deterministic tiny on-disk fixtures (see
//! `golden/util.rs`). `GIBBS_GOLDEN_WRITE=1 cargo test --release --test golden` rewrites them.

#[path = "golden/util.rs"]
mod util;

#[test]
fn harness_roundtrip() {
    let mut r = util::Rng::new(7);
    let a: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
    let mut r = util::Rng::new(7);
    assert_eq!(a, (0..3).map(|_| r.next_u64()).collect::<Vec<_>>());
    assert_eq!(util::f32_to_f16(1.0), 0x3C00);
    assert_eq!(util::f32_to_f16(-2.5), 0xC100);
    assert_eq!(util::f32_to_bf16(1.0), 0x3F80);
}
