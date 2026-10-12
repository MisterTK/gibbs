use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GibbsSettings {
    #[serde(default = "default_reencode")]
    pub reencode: String,
    #[serde(default = "default_variant")]
    pub variant: String,
    /// Mean-KL limit of the GPU re-encode gate (nats); `None` keeps the policy's own.
    #[serde(default)]
    pub reencode_max_kl: Option<f32>,
}

fn default_reencode() -> String {
    "max-speed".to_string()
}

fn default_variant() -> String {
    "fastest".to_string()
}

impl Default for GibbsSettings {
    fn default() -> Self {
        Self {
            reencode: default_reencode(),
            variant: default_variant(),
            reencode_max_kl: None,
        }
    }
}

pub fn settings_path() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache/gibbs/settings.json")
    } else {
        PathBuf::from(".cache/gibbs/settings.json")
    }
}

pub fn load_settings() -> GibbsSettings {
    let path = settings_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        if let Ok(s) = serde_json::from_str::<GibbsSettings>(&data) {
            return s;
        }
    }
    GibbsSettings::default()
}

pub fn save_settings(s: &GibbsSettings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(&path, data).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_load_save() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.json");
        let s = GibbsSettings {
            reencode: "low-bit-native".into(),
            variant: "fastest-4bit".into(),
            reencode_max_kl: Some(0.05),
        };
        let data = serde_json::to_string(&s).unwrap();
        std::fs::write(&path, data).unwrap();
        let loaded: GibbsSettings =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded, s);
    }
}
