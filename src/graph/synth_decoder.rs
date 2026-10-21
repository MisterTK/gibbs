//! A synthetic decoder graph (embedding, RMSNorm, attention, GeGLU MLP, tied head) for the
//! fusion tests: fixed shapes and weights, no checkpoint.

use crate::graph::{DType, Graph, Op, Tensor, Weights};

/// `layers` decoder layers at the given sizes, one decode row.
pub fn synthetic_decoder(
    layers: usize,
    hidden: usize,
    inter: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> (Graph, Weights) {
    let mut g = Graph::new();
    let mut w = Weights::new();
    let n = 1usize;
    let vocab = 256usize;

    let ids = g.input("token_ids", vec![n], DType::U32);
    let pos = g.input("positions", vec![n], DType::U32);

    let embed_w = g.weight("embed_tokens.weight", vec![vocab, hidden], DType::F32);
    w.insert(
        "embed_tokens.weight",
        Tensor::F32(vec![0.1; vocab * hidden]),
    );
    let mut x = g.add(
        "embed",
        Op::Embedding,
        &[embed_w, ids],
        vec![n, hidden],
        DType::F32,
    );

    for l in 0..layers {
        let p = format!("layers.{l}");

        // 1. Input Norm
        let in_norm_w = g.weight(
            format!("{p}.input_layernorm.weight"),
            vec![hidden],
            DType::F32,
        );
        w.insert(
            format!("{p}.input_layernorm.weight"),
            Tensor::F32(vec![1.0; hidden]),
        );
        let norm_x = g.add(
            format!("{p}.input_norm"),
            Op::RmsNorm { eps: 1e-6 },
            &[x, in_norm_w],
            vec![n, hidden],
            DType::F32,
        );

        // 2. QKV Projections
        let q_w = g.weight(
            format!("{p}.self_attn.q_proj.weight"),
            vec![heads * head_dim, hidden],
            DType::F32,
        );
        let k_w = g.weight(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv_heads * head_dim, hidden],
            DType::F32,
        );
        let v_w = g.weight(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv_heads * head_dim, hidden],
            DType::F32,
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            Tensor::F32(vec![0.02; heads * head_dim * hidden]),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            Tensor::F32(vec![0.02; kv_heads * head_dim * hidden]),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            Tensor::F32(vec![0.02; kv_heads * head_dim * hidden]),
        );

        let q = g.add(
            format!("{p}.self_attn.q_proj"),
            Op::Linear,
            &[norm_x, q_w],
            vec![n, heads * head_dim],
            DType::F32,
        );
        let k = g.add(
            format!("{p}.self_attn.k_proj"),
            Op::Linear,
            &[norm_x, k_w],
            vec![n, kv_heads * head_dim],
            DType::F32,
        );
        let v = g.add(
            format!("{p}.self_attn.v_proj"),
            Op::Linear,
            &[norm_x, v_w],
            vec![n, kv_heads * head_dim],
            DType::F32,
        );

        // 3. RoPE
        let inv_freq = g.weight(format!("{p}.inv_freq"), vec![head_dim / 2], DType::F32);
        w.insert(
            format!("{p}.inv_freq"),
            Tensor::F32(vec![1.0; head_dim / 2]),
        );
        let qr = g.add(
            format!("{p}.self_attn.rope_q"),
            Op::Rope {
                head_dim,
                rotary_dim: head_dim,
                variant: crate::graph::RopeVariant::HalfSplit,
                mscale: 1.0,
            },
            &[q, pos, inv_freq],
            vec![n, heads * head_dim],
            DType::F32,
        );
        let kr = g.add(
            format!("{p}.self_attn.rope_k"),
            Op::Rope {
                head_dim,
                rotary_dim: head_dim,
                variant: crate::graph::RopeVariant::HalfSplit,
                mscale: 1.0,
            },
            &[k, pos, inv_freq],
            vec![n, kv_heads * head_dim],
            DType::F32,
        );

        // 4. Attention
        let ks = g.state(
            format!("kv.{l}.k"),
            vec![8192, kv_heads * head_dim],
            DType::F32,
            false,
        );
        let vs = g.state(
            format!("kv.{l}.v"),
            vec![8192, kv_heads * head_dim],
            DType::F32,
            false,
        );
        let att = g.add(
            format!("{p}.self_attn.att"),
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim: head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                causal: true,
                window: Some(512),
                soft_cap: None,
                write: true,
            },
            &[qr, kr, v, ks, vs, pos],
            vec![n, heads * head_dim],
            DType::F32,
        );

        // 5. Output Projection + Residual Add
        let o_w = g.weight(
            format!("{p}.self_attn.o_proj.weight"),
            vec![hidden, heads * head_dim],
            DType::F32,
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            Tensor::F32(vec![0.02; hidden * heads * head_dim]),
        );
        let o = g.add(
            format!("{p}.self_attn.o_proj"),
            Op::Linear,
            &[att, o_w],
            vec![n, hidden],
            DType::F32,
        );
        x = g.add(
            format!("{p}.add_att"),
            Op::Binary { kind: crate::graph::BinaryKind::Add },
            &[x, o],
            vec![n, hidden],
            DType::F32,
        );

        // 6. Post-Attn Norm
        let post_norm_w = g.weight(
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden],
            DType::F32,
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::F32(vec![1.0; hidden]),
        );
        let norm_ffn = g.add(
            format!("{p}.post_attn_norm"),
            Op::RmsNorm { eps: 1e-6 },
            &[x, post_norm_w],
            vec![n, hidden],
            DType::F32,
        );

        // 7. Gate-Up MLP
        let gate_w = g.weight(
            format!("{p}.mlp.gate_proj.weight"),
            vec![inter, hidden],
            DType::F32,
        );
        let up_w = g.weight(
            format!("{p}.mlp.up_proj.weight"),
            vec![inter, hidden],
            DType::F32,
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            Tensor::F32(vec![0.02; inter * hidden]),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            Tensor::F32(vec![0.02; inter * hidden]),
        );
        let gate = g.add(
            format!("{p}.mlp.gate_proj"),
            Op::Linear,
            &[norm_ffn, gate_w],
            vec![n, inter],
            DType::F32,
        );
        let up = g.add(
            format!("{p}.mlp.up_proj"),
            Op::Linear,
            &[norm_ffn, up_w],
            vec![n, inter],
            DType::F32,
        );
        let act = g.add(
            format!("{p}.mlp.act"),
            Op::GeGlu,
            &[gate, up],
            vec![n, inter],
            DType::F32,
        );

        // 8. Down Projection + Residual Add
        let down_w = g.weight(
            format!("{p}.mlp.down_proj.weight"),
            vec![hidden, inter],
            DType::F32,
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            Tensor::F32(vec![0.02; hidden * inter]),
        );
        let down = g.add(
            format!("{p}.mlp.down_proj"),
            Op::Linear,
            &[act, down_w],
            vec![n, hidden],
            DType::F32,
        );
        x = g.add(
            format!("{p}.add_ffn"),
            Op::Binary { kind: crate::graph::BinaryKind::Add },
            &[x, down],
            vec![n, hidden],
            DType::F32,
        );
    }

    let final_norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
    w.insert("norm.weight", Tensor::F32(vec![1.0; hidden]));
    let x_norm = g.add(
        "final_norm",
        Op::RmsNorm { eps: 1e-6 },
        &[x, final_norm_w],
        vec![n, hidden],
        DType::F32,
    );

    let logits = g.add(
        "logits",
        Op::Linear,
        &[x_norm, embed_w],
        vec![n, vocab],
        DType::F32,
    );
    g.output(logits);

    (g, w)
}
