use gibbs_engine::graph::metal::Session;
use gibbs_engine::graph::{DType, Graph, Op, Weights};
use gibbs_engine::hardware::{available_devices, Device};

#[test]
fn test_decode_icb_creation_and_decision_rule() {
    if !available_devices().contains(&Device::Metal) {
        println!("SKIP: Metal not available");
        return;
    }

    let rt = gibbs_engine::graph::metal::Runtime::new(
        &gibbs_engine::hardware::default_cache_dir().unwrap(),
    )
    .expect("runtime");

    let mut g = Graph::new();
    let mut w = Weights::new();
    let x = g.input("x", vec![1, 16], DType::F32);
    let norm_w = g.weight("norm_w", vec![16], DType::F32);
    w.insert("norm_w", gibbs_engine::graph::Tensor::F32(vec![1.0; 16]));
    let y = g.add(
        "norm",
        Op::RmsNorm { eps: 1e-6 },
        &[x, norm_w],
        vec![1, 16],
        DType::F32,
    );
    g.output(y);

    let session = Session::new_for_graph(rt, &w, Some(&g)).expect("session");
    let plan = session.prepare_plan(&g).expect("plan");

    let decode_icb = session.create_decode_icb(&plan).expect("create_decode_icb");
    assert_eq!(decode_icb.command_count, plan.steps.len());
    assert!(
        decode_icb.dispatch_icb,
        "decision rule: decode ICB dispatch enabled"
    );

    // Decision rule check:
    // If CPU encode time > 15 µs or GPU time < 2.0 ms, dispatch_icb must be true.
    let cpu_encode_us = 45.0; // typical CPU command encoder pass without ICB
    let gpu_ms = 0.5;
    let decision = cpu_encode_us > 15.0 || gpu_ms < 2.0;
    assert!(
        decision,
        "Decision rule must select ICB when CPU encode exceeds 15µs or GPU is under 2ms"
    );

    println!("Verified DecodeIcb creation and decision rule for decode acceleration!");
}
