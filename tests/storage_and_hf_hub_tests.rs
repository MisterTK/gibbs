use gibbs_engine::model::hub::{HuggingFaceDownloader, HuggingFaceHub, StorageBackend};
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_storage_backend_uri_parsing() {
    // 1. S3 URI
    let s3_backend =
        StorageBackend::from_uri("s3://foundry-models-us-east/checkpoints/v1").unwrap();
    if let StorageBackend::S3 {
        bucket,
        prefix,
        endpoint,
    } = s3_backend
    {
        assert_eq!(bucket, "foundry-models-us-east");
        assert_eq!(prefix, "checkpoints/v1");
        assert!(endpoint.is_none());
    } else {
        panic!("Expected StorageBackend::S3");
    }

    // 2. GCS gs:// URI
    let gcs_backend = StorageBackend::from_uri("gs://ai-bucket-prod/llm/smollm").unwrap();
    if let StorageBackend::Gcs { bucket, prefix } = gcs_backend {
        assert_eq!(bucket, "ai-bucket-prod");
        assert_eq!(prefix, "llm/smollm");
    } else {
        panic!("Expected StorageBackend::Gcs");
    }

    // 3. GCS gcs:// URI
    let gcs_backend2 = StorageBackend::from_uri("gcs://enterprise-cache/models").unwrap();
    if let StorageBackend::Gcs { bucket, prefix } = gcs_backend2 {
        assert_eq!(bucket, "enterprise-cache");
        assert_eq!(prefix, "models");
    } else {
        panic!("Expected StorageBackend::Gcs");
    }

    // 4. Local Disk URI
    let disk_backend = StorageBackend::from_uri("/var/data/huggingface_cache").unwrap();
    if let StorageBackend::LocalDisk { cache_dir } = disk_backend {
        assert_eq!(cache_dir, PathBuf::from("/var/data/huggingface_cache"));
    } else {
        panic!("Expected StorageBackend::LocalDisk");
    }
}

#[test]
fn test_huggingface_repo_id_parsing() {
    // Standard org/repo
    let (org1, repo1) =
        HuggingFaceDownloader::parse_hf_repo("unsloth/SmolLM2-135M-Instruct-GGUF").unwrap();
    assert_eq!(org1, "unsloth");
    assert_eq!(repo1, "SmolLM2-135M-Instruct-GGUF");

    let (org2, repo2) = HuggingFaceDownloader::parse_hf_repo("roneneldan/TinyStories-1M").unwrap();
    assert_eq!(org2, "roneneldan");
    assert_eq!(repo2, "TinyStories-1M");

    // Single-segment repo name
    let (org3, repo3) = HuggingFaceDownloader::parse_hf_repo("gpt2").unwrap();
    assert_eq!(org3, "");
    assert_eq!(repo3, "gpt2");

    // Empty or invalid repo
    assert!(HuggingFaceDownloader::parse_hf_repo("").is_none());
    assert!(HuggingFaceDownloader::parse_hf_repo("   ").is_none());
}

#[test]
fn test_hub_auto_download_and_cache_directory_layout() {
    let dir = tempdir().unwrap();
    let cache_root = dir.path().to_path_buf();
    let hub = HuggingFaceHub::with_cache_dir(cache_root.clone());

    // 1. First test: simulated pre-cached model
    let model_id = "test-org/micro-llama-3m";
    let snapshot_dir = cache_root
        .join("models--test-org--micro-llama-3m")
        .join("snapshots")
        .join("commit12345");
    fs::create_dir_all(&snapshot_dir).unwrap();
    fs::write(
        snapshot_dir.join("config.json"),
        r#"{"hidden_size": 128, "num_hidden_layers": 2}"#,
    )
    .unwrap();
    fs::write(snapshot_dir.join("tokenizer.json"), r#"{"vocab": {}}"#).unwrap();
    fs::write(snapshot_dir.join("model.safetensors"), b"fake_weights").unwrap();

    // locate_model should find the cached model
    let artifacts = hub.locate_model(model_id);
    assert!(artifacts.is_cached);
    assert!(artifacts.config_path.is_some());
    assert!(artifacts.tokenizer_path.is_some());
    assert!(artifacts.weights_path.is_some());

    // 2. auto_download_and_cache on already-cached model should be immediate
    let resolved = hub.auto_download_and_cache(model_id, None).unwrap();
    assert!(resolved.is_cached);
    assert_eq!(resolved.model_id, model_id);

    // 3. auto_download_and_cache on a tiny model that doesn't exist yet should synthesize and populate cache
    let tiny_id = "foundry/tiny-5m";
    let tiny_artifacts = hub.auto_download_and_cache(tiny_id, None).unwrap();
    assert!(tiny_artifacts.is_cached);
    assert!(tiny_artifacts.config_path.is_some());
}
