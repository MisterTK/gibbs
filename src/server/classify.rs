//! `/v1/classify` and `/v1/rerank` for the active sequence classifier.

use super::{api_error, AppState};
use crate::model::classifier::SequenceClassifier;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClassifyInput {
    One(String),
    Many(Vec<String>),
    /// `[[text, text_pair], ...]` for cross-encoders and rerankers.
    Pairs(Vec<(String, String)>),
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub input: ClassifyInput,
    /// Labels per input, highest first (default: all).
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object { text: String },
}

/// Jina/Cohere (`documents`) and TEI (`texts`) rerank requests.
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    #[serde(alias = "texts")]
    pub documents: Vec<RerankDocument>,
    pub top_n: Option<usize>,
}

fn active(state: &AppState) -> Result<Arc<SequenceClassifier>, axum::response::Response> {
    state.classifier.read().clone().ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "no text-classification model is loaded; start with --model <classifier repo> or load one in the TUI",
        )
    })
}

/// Runs `inputs` (text, optional pair) through `c` off the async runtime.
async fn run(
    c: Arc<SequenceClassifier>,
    inputs: Vec<(String, Option<String>)>,
) -> Result<Vec<(Vec<crate::model::classifier::LabelScore>, usize)>, axum::response::Response> {
    tokio::task::spawn_blocking(move || {
        inputs
            .iter()
            .map(|(t, p)| c.classify(t, p.as_deref()))
            .collect::<Result<Vec<_>, String>>()
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| api_error(StatusCode::BAD_REQUEST, e))
}

pub async fn handle_classify(
    State(state): State<AppState>,
    Json(req): Json<ClassifyRequest>,
) -> axum::response::Response {
    let c = match active(&state) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let inputs: Vec<(String, Option<String>)> = match req.input {
        ClassifyInput::One(t) => vec![(t, None)],
        ClassifyInput::Many(v) => v.into_iter().map(|t| (t, None)).collect(),
        ClassifyInput::Pairs(v) => v.into_iter().map(|(a, b)| (a, Some(b))).collect(),
    };
    if inputs.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "input is empty");
    }
    if req.top_k == Some(0) {
        return api_error(StatusCode::BAD_REQUEST, "top_k must be at least 1");
    }
    let model = c.repo_id.clone();
    let results = match run(c, inputs).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    let tokens: usize = results.iter().map(|(_, n)| n).sum();
    let data: Vec<_> = results
        .into_iter()
        .enumerate()
        .map(|(index, (labels, _))| {
            let labels: Vec<_> = labels
                .into_iter()
                .take(req.top_k.unwrap_or(usize::MAX))
                .map(|l| serde_json::json!({ "label": l.label, "score": l.score }))
                .collect();
            serde_json::json!({ "index": index, "labels": labels })
        })
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "model": model,
        "data": data,
        "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
    }))
    .into_response()
}

pub async fn handle_rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> axum::response::Response {
    let c = match active(&state) {
        Ok(c) => c,
        Err(e) => return e,
    };
    if c.labels.len() != 1 {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "{} has {} labels; rerank needs a 1-label pair model (cross-encoder / reranker)",
                c.repo_id,
                c.labels.len()
            ),
        );
    }
    if req.documents.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "documents is empty");
    }
    let inputs = req
        .documents
        .into_iter()
        .map(|d| match d {
            RerankDocument::Text(t) | RerankDocument::Object { text: t } => {
                (req.query.clone(), Some(t))
            }
        })
        .collect();
    let model = c.repo_id.clone();
    let results = match run(c, inputs).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    let tokens: usize = results.iter().map(|(_, n)| n).sum();
    let mut ranked: Vec<(usize, f32)> = results
        .iter()
        .enumerate()
        .map(|(i, (labels, _))| (i, labels[0].score))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(req.top_n.unwrap_or(usize::MAX));
    let results: Vec<_> = ranked
        .into_iter()
        .map(|(index, s)| serde_json::json!({ "index": index, "relevance_score": s }))
        .collect();
    Json(serde_json::json!({
        "model": model,
        "results": results,
        "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
    }))
    .into_response()
}
