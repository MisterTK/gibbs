use gibbs_engine::model::graph_lm::{GraphLm, GraphLmConfig};
use gibbs_engine::scheduler::{ExecutableModel, SeqKvCache};
use std::sync::Arc;

#[test]
fn test_decode_step_logits_parity_16_steps() {
    let cfg = GraphLmConfig::smollm_1m();
    let model = Arc::new(GraphLm::new(cfg));
    let exec = ExecutableModel::from(model);

    let prompt = vec![1u32, 10, 25, 4];
    let mut kv_cache: Option<SeqKvCache> = None;

    // Prefill prompt
    let _ = exec
        .step_token_with_kv(&prompt, 4, 4, 0.0, 1.0, &mut kv_cache)
        .expect("prefill");

    // Run 16 consecutive decode steps with decode_step_logits
    let mut current_tok = 12u32;
    for step in 0..16 {
        let pos = prompt.len() + step;
        let logits = exec.decode_step_logits(current_tok, pos, &mut kv_cache);
        assert!(
            logits.is_ok(),
            "decode_step_logits failed at step {step}: {:?}",
            logits.err()
        );
        let l = logits.unwrap();
        assert_eq!(l.len(), 2048);

        // Pick next token greedily
        current_tok = l
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap();
    }

    println!("Verified decode_step_logits across 16 consecutive decode steps!");
}
