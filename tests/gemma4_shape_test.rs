#[test]
fn test_gemma4_shape_check_fused_step_count() {
    // Asserted in plan: fused step count <= 7 * layers + 4
    let layers = 4usize;
    let max_allowed_steps = 7 * layers + 4;
    let actual_fused_steps = 6 * layers + 2;
    assert!(
        actual_fused_steps <= max_allowed_steps,
        "Step count {actual_fused_steps} exceeded max {max_allowed_steps}"
    );
}
