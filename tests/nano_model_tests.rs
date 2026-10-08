use gibbs_engine::foundry_engine;
use foundry_engine::model::nano::{NanoTransformerConfig, NanoTransformerModel};

#[test]
fn test_nano_transformer_forward_decode() {
    let cfg = NanoTransformerConfig {
        vocab_size: 256,
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        eps: 1e-5,
    };

    let model = NanoTransformerModel::new(cfg);
    
    // Execute a multi-step token generation trajectory
    let mut current_token = 42u32;
    let mut generated_tokens = Vec::new();

    for pos in 0..10 {
        let next_token = model.forward_token(current_token, pos);
        assert!(next_token < 256);
        generated_tokens.push(next_token);
        current_token = next_token;
    }

    assert_eq!(generated_tokens.len(), 10);
}
