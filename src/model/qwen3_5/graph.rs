use crate::graph::{DType, Graph, Op, Tensor, Weights};

pub struct Qwen35Graph;

impl Qwen35Graph {
    pub fn build_graph(
        n: usize,
        hidden_size: usize,
        num_layers: usize,
        heads: usize,
        head_dim: usize,
        vocab_size: usize,
    ) -> (Graph, Weights) {
        let mut g = Graph::new();
        let mut w = Weights::new();

        let ids = g.input("token_ids", vec![n], DType::U32);
        let positions = g.input("positions", vec![n], DType::U32);

        let embed_w = g.weight(
            "model.embed_tokens.weight",
            vec![vocab_size, hidden_size],
            DType::F32,
        );
        let mut x = g.add(
            "embed",
            Op::Embedding,
            &[embed_w, ids],
            vec![n, hidden_size],
            DType::F32,
        );

        for l in 0..num_layers {
            let p = format!("model.layers.{l}");
            let in_norm_w = g.weight(
                format!("{p}.input_layernorm.weight"),
                vec![hidden_size],
                DType::F32,
            );
            let a = g.add(
                format!("{p}.input_norm"),
                Op::RmsNorm { eps: 1e-6 },
                &[x, in_norm_w],
                vec![n, hidden_size],
                DType::F32,
            );

            let q_w = g.weight(
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * head_dim, hidden_size],
                DType::F32,
            );
            let q = g.add(
                format!("{p}.q"),
                Op::Linear,
                &[a, q_w],
                vec![n, heads * head_dim],
                DType::F32,
            );

            let k_w = g.weight(
                format!("{p}.self_attn.k_proj.weight"),
                vec![heads * head_dim, hidden_size],
                DType::F32,
            );
            let k = g.add(
                format!("{p}.k"),
                Op::Linear,
                &[a, k_w],
                vec![n, heads * head_dim],
                DType::F32,
            );

            let v_w = g.weight(
                format!("{p}.self_attn.v_proj.weight"),
                vec![heads * head_dim, hidden_size],
                DType::F32,
            );
            let v = g.add(
                format!("{p}.v"),
                Op::Linear,
                &[a, v_w],
                vec![n, heads * head_dim],
                DType::F32,
            );

            let inv_freq = g.weight(format!("{p}.inv_freq"), vec![head_dim / 2], DType::F32);
            w.insert(
                format!("{p}.inv_freq"),
                Tensor::F32(vec![1.0; head_dim / 2]),
            );

            let qr = g.add(
                format!("{p}.qr"),
                Op::Rope {
                    head_dim,
                    rotary_dim: head_dim,
                    variant: crate::graph::RopeVariant::HalfSplit,
                    mscale: 1.0,
                },
                &[q, positions, inv_freq],
                vec![n, heads * head_dim],
                DType::F32,
            );
            let kr = g.add(
                format!("{p}.kr"),
                Op::Rope {
                    head_dim,
                    rotary_dim: head_dim,
                    variant: crate::graph::RopeVariant::HalfSplit,
                    mscale: 1.0,
                },
                &[k, positions, inv_freq],
                vec![n, heads * head_dim],
                DType::F32,
            );

            let ks = g.state(
                format!("kv.{l}.k"),
                vec![8192, heads * head_dim],
                DType::F32,
            );
            let vs = g.state(
                format!("kv.{l}.v"),
                vec![8192, heads * head_dim],
                DType::F32,
            );

            let att = g.add(
                format!("{p}.att"),
                Op::CachedAttention {
                    heads,
                    kv_heads: heads,
                    head_dim,
                    v_head_dim: head_dim,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    causal: true,
                    window: None,
                    soft_cap: None,
                    write: true,
                },
                &[qr, kr, v, ks, vs, positions],
                vec![n, heads * head_dim],
                DType::F32,
            );

            let o_w = g.weight(
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden_size, heads * head_dim],
                DType::F32,
            );
            let o = g.add(
                format!("{p}.o"),
                Op::Linear,
                &[att, o_w],
                vec![n, hidden_size],
                DType::F32,
            );

            x = g.add(
                format!("{p}.add_att"),
                Op::Binary { kind: BinaryKind::Add },
                &[x, o],
                vec![n, hidden_size],
                DType::F32,
            );
        }

        let final_norm_w = g.weight("model.norm.weight", vec![hidden_size], DType::F32);
        let x_norm = g.add(
            "final_norm",
            Op::RmsNorm { eps: 1e-6 },
            &[x, final_norm_w],
            vec![n, hidden_size],
            DType::F32,
        );

        let lm_head_w = g.weight("lm_head.weight", vec![vocab_size, hidden_size], DType::F32);
        let logits = g.add(
            "logits",
            Op::Linear,
            &[x_norm, lm_head_w],
            vec![n, vocab_size],
            DType::F32,
        );
        g.output(logits);

        (g, w)
    }
}
