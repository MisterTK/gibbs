use foundry_engine::kv::PagedKVPool;
use foundry_engine::model::hf_loader::HuggingFaceModelDir;
use foundry_engine::scheduler::ContinuousBatchScheduler;
use gibbs_engine::foundry_engine;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_huggingface_disk_directory_loader_and_scheduler() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path();

    // Create synthetic config.json
    let config_json = r#"{
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "vocab_size": 256
    }"#;
    fs::write(model_dir.join("config.json"), config_json).unwrap();

    // Load model from directory
    let model = HuggingFaceModelDir::load_from_dir(model_dir).unwrap();
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.vocab_size, 256);

    // Wire into scheduler and execute real forward step
    let kv_pool = Arc::new(PagedKVPool::new(512));
    let scheduler = ContinuousBatchScheduler::new("", kv_pool, Some(model.into()));

    let (_, _rx) = scheduler.submit_request_stream(vec![1, 10, 20]);
    let step_out = scheduler.step();
    assert!(step_out.tokens_generated >= 1);
}
