use gibbs_engine::graph::metal::Session;
use gibbs_engine::graph::{DType, Graph, Op, Weights};
use gibbs_engine::hardware::{available_devices, Device};

#[test]
fn test_unified_params_buffer_aligned_and_bound() {
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

    assert!(
        plan.params_buffer_size >= 64,
        "params_buffer_size must be >= 64"
    );
    assert_eq!(
        plan.params_buffer_size % 64,
        0,
        "params_buffer_size must be 64-byte aligned"
    );

    let ptr = session
        .params_buffer_ptr(plan.params_buffer_size)
        .expect("ptr");
    assert!(!ptr.is_null(), "params_buffer_ptr must be non-null");
    assert_eq!(
        (ptr as usize) % 64,
        0,
        "params buffer allocation must be 64-byte aligned"
    );

    println!(
        "Verified unified params buffer size {} and 64-byte alignment!",
        plan.params_buffer_size
    );
}
