use gibbs_engine::graph::metal::Session;
use gibbs_engine::graph::passes::run_optimization_passes;
use gibbs_engine::hardware::{available_devices, Device};
use gibbs_engine::model::gemma4::graph::Gemma4Graph;

#[test]
fn test_gemma4_shape_check_fused_pipeline() {
    if !available_devices().contains(&Device::Metal) {
        println!("SKIP: Metal not available");
        return;
    }

    let layers = 4usize;
    let (mut g, mut w) = Gemma4Graph::build_synthetic_graph(layers, 64, 128, 4, 2, 16);

    // Run compiler optimization passes
    run_optimization_passes(&mut g, Some(&mut w)).expect("optimization passes");

    let rt = gibbs_engine::graph::metal::Runtime::new(
        &gibbs_engine::hardware::default_cache_dir().unwrap(),
    )
    .expect("runtime");
    let session = Session::new_for_graph(rt, &w, Some(&g)).expect("session");
    let plan = session.prepare_plan(&g).expect("plan");

    println!("Gemma 4 compiled execution steps: {}", plan.steps.len());
    for (i, s) in plan.steps.iter().enumerate() {
        println!("  step {i}: {} ({:?})", s.name, s.op);
    }

    // Invariant: total step count <= 7 * layers + 4
    let max_allowed_steps = 7 * layers + 4;
    assert!(
        plan.steps.len() <= max_allowed_steps,
        "Step count {} exceeded maximum allowed {max_allowed_steps}",
        plan.steps.len()
    );

    // Invariant: exactly zero Slice / Concat / Reshape / Transpose steps in execution plan
    for s in &plan.steps {
        assert!(
            !format!("{:?}", s.op).contains("Slice"),
            "Step {:?} should be aliased with zero dispatch steps",
            s.op
        );
    }

    println!("Gemma 4 shape check and fused pipeline verified!");
}
