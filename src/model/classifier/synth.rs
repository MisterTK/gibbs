//! Tiny synthetic classifier repos (random f32 weights, a word-level tokenizer) for the Metal
//! parity tests. Included by path from the lib tests and from `tests/cpu_free_matrix.rs`, so it
//! depends on `serde_json` and `std` only.

use serde_json::{json, Value};
use std::path::PathBuf;

pub struct Ts {
    pub t: Vec<(String, Vec<usize>, Vec<f32>)>,
    s: u64,
}

impl Ts {
    pub fn new(seed: u64) -> Self {
        Self { t: Vec::new(), s: seed }
    }
    pub fn rnd(&mut self) -> f32 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    pub fn w(&mut self, name: &str, shape: &[usize]) {
        let v = (0..shape.iter().product::<usize>()).map(|_| self.rnd() * 0.6).collect();
        self.t.push((name.into(), shape.to_vec(), v));
    }
    /// `name.weight` `[out, in]` and `name.bias`.
    pub fn lin(&mut self, name: &str, out: usize, inp: usize) {
        self.w(&format!("{name}.weight"), &[out, inp]);
        self.w(&format!("{name}.bias"), &[out]);
    }
    pub fn ln(&mut self, name: &str, d: usize) {
        self.gamma(name, d);
        self.w(&format!("{name}.bias"), &[d]);
    }
    /// `name.weight` alone, near 1 (a bias-free norm).
    pub fn gamma(&mut self, name: &str, d: usize) {
        let g = (0..d).map(|_| 1.0 + self.rnd() * 0.2).collect();
        self.t.push((format!("{name}.weight"), vec![d], g));
    }
    fn bytes(&self) -> Vec<u8> {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape, v) in &self.t {
            let start = data.len();
            v.iter().for_each(|x| data.extend(x.to_le_bytes()));
            header.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        out
    }
}

pub const VOCAB: usize = 40;
pub const TEXT: &str = "w3 w7 w1 w9 w2 w5 w11 w4 w6";
pub const PAIR: &str = "w8 w13 w2";

/// Writes `models--t--{name}` (config.json, model.safetensors, a word-level tokenizer.json) under a
/// fresh temp dir and returns that cache dir; the repo id is `t/{name}`.
pub fn write_repo(name: &str, mut cfg: Value, ts: &Ts) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gibbs-cls-{}-{name}", std::process::id()));
    let snap = dir.join(format!("models--t--{name}/snapshots/main"));
    std::fs::create_dir_all(&snap).unwrap();
    cfg["id2label"] = json!({"0": "A", "1": "B", "2": "C"});
    cfg["vocab_size"] = json!(VOCAB);
    std::fs::write(snap.join("config.json"), cfg.to_string()).unwrap();
    std::fs::write(snap.join("model.safetensors"), ts.bytes()).unwrap();
    let vocab: serde_json::Map<String, Value> = (0..VOCAB).map(|i| (format!("w{i}"), json!(i))).collect();
    let tok = json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "w0"}});
    std::fs::write(snap.join("tokenizer.json"), tok.to_string()).unwrap();
    dir
}

fn bert_cfg(arch: &str, problem: Option<&str>) -> Value {
    json!({"architectures": [arch], "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 24, "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
        "problem_type": problem})
}

fn bert_layer(ts: &mut Ts, p: &str, h: usize, inter: usize) {
    for s in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"] {
        ts.lin(&format!("{p}.{s}"), h, h);
    }
    ts.ln(&format!("{p}.attention.output.LayerNorm"), h);
    ts.lin(&format!("{p}.intermediate.dense"), inter, h);
    ts.lin(&format!("{p}.output.dense"), h, inter);
    ts.ln(&format!("{p}.output.LayerNorm"), h);
}

/// `BertForSequenceClassification`; `problem` is the `problem_type` (None = single label).
pub fn bert(problem: Option<&str>) -> (Value, Ts) {
    let mut ts = Ts::new(1);
    ts.w("bert.embeddings.word_embeddings.weight", &[VOCAB, 16]);
    ts.w("bert.embeddings.position_embeddings.weight", &[32, 16]);
    ts.w("bert.embeddings.token_type_embeddings.weight", &[2, 16]);
    ts.ln("bert.embeddings.LayerNorm", 16);
    for l in 0..2 {
        bert_layer(&mut ts, &format!("bert.encoder.layer.{l}"), 16, 24);
    }
    ts.lin("bert.pooler.dense", 16, 16);
    ts.lin("classifier", 3, 16);
    (bert_cfg("BertForSequenceClassification", problem), ts)
}

pub fn modernbert(pooling: &str, yarn: bool) -> (Value, Ts) {
    let (h, inter, layers) = (16, 24, 3);
    let mut cfg = json!({"architectures": ["ModernBertForSequenceClassification"], "hidden_size": h,
        "num_attention_heads": 2, "intermediate_size": inter, "num_hidden_layers": layers,
        "local_attention": 4, "global_attn_every_n_layers": 3, "norm_eps": 1e-5, "norm_bias": false,
        "classifier_pooling": pooling, "max_position_embeddings": 64});
    if yarn {
        cfg["rope_scaling"] = json!({"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32});
    }
    let mut ts = Ts::new(2);
    ts.w("model.embeddings.tok_embeddings.weight", &[VOCAB, h]);
    ts.gamma("model.embeddings.norm", h);
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        if l > 0 {
            ts.gamma(&format!("{p}.attn_norm"), h);
        }
        ts.w(&format!("{p}.attn.Wqkv.weight"), &[3 * h, h]);
        ts.w(&format!("{p}.attn.Wo.weight"), &[h, h]);
        ts.gamma(&format!("{p}.mlp_norm"), h);
        ts.w(&format!("{p}.mlp.Wi.weight"), &[2 * inter, h]);
        ts.w(&format!("{p}.mlp.Wo.weight"), &[h, inter]);
    }
    ts.gamma("model.final_norm", h);
    ts.w("head.dense.weight", &[h, h]);
    ts.gamma("head.norm", h);
    ts.lin("classifier", 3, h);
    (cfg, ts)
}

pub fn electra() -> (Value, Ts) {
    let (h, e, inter) = (16, 8, 24);
    let mut cfg = bert_cfg("ElectraForSequenceClassification", None);
    cfg["embedding_size"] = json!(e);
    let mut ts = Ts::new(3);
    ts.w("electra.embeddings.word_embeddings.weight", &[VOCAB, e]);
    ts.w("electra.embeddings.position_embeddings.weight", &[32, e]);
    ts.w("electra.embeddings.token_type_embeddings.weight", &[2, e]);
    ts.ln("electra.embeddings.LayerNorm", e);
    ts.lin("electra.embeddings_project", h, e);
    for l in 0..2 {
        bert_layer(&mut ts, &format!("electra.encoder.layer.{l}"), h, inter);
    }
    ts.lin("classifier.dense", h, h);
    ts.lin("classifier.out_proj", 3, h);
    (cfg, ts)
}

/// Three steps over one shared layer group, tanh GELU.
pub fn albert() -> (Value, Ts) {
    let (h, e, inter) = (16, 8, 24);
    let mut cfg = bert_cfg("AlbertForSequenceClassification", None);
    cfg["embedding_size"] = json!(e);
    cfg["num_hidden_layers"] = json!(3);
    cfg["num_hidden_groups"] = json!(1);
    cfg["inner_group_num"] = json!(1);
    cfg["hidden_act"] = json!("gelu_new");
    let mut ts = Ts::new(4);
    ts.w("albert.embeddings.word_embeddings.weight", &[VOCAB, e]);
    ts.w("albert.embeddings.position_embeddings.weight", &[32, e]);
    ts.w("albert.embeddings.token_type_embeddings.weight", &[2, e]);
    ts.ln("albert.embeddings.LayerNorm", e);
    ts.lin("albert.encoder.embedding_hidden_mapping_in", h, e);
    let p = "albert.encoder.albert_layer_groups.0.albert_layers.0";
    for s in ["attention.query", "attention.key", "attention.value", "attention.dense"] {
        ts.lin(&format!("{p}.{s}"), h, h);
    }
    ts.ln(&format!("{p}.attention.LayerNorm"), h);
    ts.lin(&format!("{p}.ffn"), inter, h);
    ts.lin(&format!("{p}.ffn_output"), h, inter);
    ts.ln(&format!("{p}.full_layer_layer_norm"), h);
    ts.lin("albert.pooler", h, h);
    ts.lin("classifier", 3, h);
    (cfg, ts)
}

fn deberta_layers(ts: &mut Ts, h: usize, inter: usize, layers: usize, v1: bool) {
    for l in 0..layers {
        let p = format!("deberta.encoder.layer.{l}");
        if v1 {
            ts.w(&format!("{p}.attention.self.in_proj.weight"), &[3 * h, h]);
            ts.w(&format!("{p}.attention.self.q_bias"), &[h]);
            ts.w(&format!("{p}.attention.self.v_bias"), &[h]);
            ts.w(&format!("{p}.attention.self.pos_proj.weight"), &[h, h]);
            ts.lin(&format!("{p}.attention.self.pos_q_proj"), h, h);
        } else {
            for s in ["query_proj", "key_proj", "value_proj"] {
                ts.lin(&format!("{p}.attention.self.{s}"), h, h);
            }
        }
        ts.lin(&format!("{p}.attention.output.dense"), h, h);
        ts.ln(&format!("{p}.attention.output.LayerNorm"), h);
        ts.lin(&format!("{p}.intermediate.dense"), inter, h);
        ts.lin(&format!("{p}.output.dense"), h, inter);
        ts.ln(&format!("{p}.output.LayerNorm"), h);
    }
    ts.lin("pooler.dense", h, h);
    ts.lin("classifier", 3, h);
}

/// v2/v3: shared keys, no absolute positions, log-bucketed relative positions (4 buckets, so the
/// 9-12 token inputs reach the log branch), `norm_rel_ebd = layer_norm`.
pub fn deberta_v2() -> (Value, Ts) {
    let h = 16;
    let cfg = json!({"architectures": ["DebertaV2ForSequenceClassification"], "hidden_size": h,
        "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 24,
        "max_position_embeddings": 32, "relative_attention": true, "pos_att_type": ["p2c", "c2p"],
        "share_att_key": true, "position_biased_input": false, "norm_rel_ebd": "layer_norm",
        "position_buckets": 4, "max_relative_positions": 16, "type_vocab_size": 0,
        "layer_norm_eps": 1e-7});
    let mut ts = Ts::new(5);
    ts.w("deberta.embeddings.word_embeddings.weight", &[VOCAB, h]);
    ts.ln("deberta.embeddings.LayerNorm", h);
    ts.w("deberta.encoder.rel_embeddings.weight", &[8, h]);
    ts.ln("deberta.encoder.LayerNorm", h);
    deberta_layers(&mut ts, h, 24, 2, false);
    (cfg, ts)
}

/// v1: `in_proj` with per-head interleaved q/k/v, separate pos projections, absolute positions.
pub fn deberta_v1() -> (Value, Ts) {
    let h = 16;
    let cfg = json!({"architectures": ["DebertaForSequenceClassification"], "hidden_size": h,
        "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 24,
        "max_position_embeddings": 32, "relative_attention": true, "pos_att_type": "c2p|p2c",
        "position_biased_input": true, "max_relative_positions": 6, "type_vocab_size": 0,
        "layer_norm_eps": 1e-7});
    let mut ts = Ts::new(6);
    ts.w("deberta.embeddings.word_embeddings.weight", &[VOCAB, h]);
    ts.w("deberta.embeddings.position_embeddings.weight", &[32, h]);
    ts.ln("deberta.embeddings.LayerNorm", h);
    ts.w("deberta.encoder.rel_embeddings.weight", &[12, h]);
    deberta_layers(&mut ts, h, 24, 2, true);
    (cfg, ts)
}
