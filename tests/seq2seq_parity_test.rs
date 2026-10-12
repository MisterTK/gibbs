use gibbs_engine::graph::cpu::run_stateless;
use gibbs_engine::graph::{DType, Graph, Op, Tensor, Weights};

#[test]
fn test_t5_seq2seq_encoder_decoder_graph_parity() {
    // 1. Encoder program: static n=prompt_len
    let prompt_len = 4usize;
    let d_model = 16usize;
    let heads = 2usize;
    let buckets = 16usize;

    let mut enc_g = Graph::new();
    let mut enc_w = Weights::new();

    let ids = enc_g.input("token_ids", vec![prompt_len], DType::U32);
    let embed = enc_g.weight("shared.weight", vec![32, d_model], DType::F32);
    enc_w.insert("shared.weight", Tensor::F32(vec![0.1; 32 * d_model]));

    let e = enc_g.add(
        "embed",
        Op::Embedding,
        &[embed, ids],
        vec![prompt_len, d_model],
        DType::F32,
    );

    let norm_w = enc_g.weight("encoder.norm.weight", vec![d_model], DType::F32);
    enc_w.insert("encoder.norm.weight", Tensor::F32(vec![1.0; d_model]));
    let norm = enc_g.add(
        "norm",
        Op::RmsNorm { eps: 1e-6 },
        &[e, norm_w],
        vec![prompt_len, d_model],
        DType::F32,
    );

    let rpb_w = enc_g.weight("encoder.rpb.weight", vec![buckets, heads], DType::F32);
    enc_w.insert(
        "encoder.rpb.weight",
        Tensor::F32(vec![0.05; buckets * heads]),
    );
    let q_pos = enc_g.input("q_pos", vec![prompt_len], DType::U32);
    let k_pos = enc_g.input("k_pos", vec![prompt_len], DType::U32);

    let bias = enc_g.add(
        "bias",
        Op::RelPosBias {
            buckets,
            max_distance: 32,
            bidirectional: true,
            heads,
        },
        &[rpb_w, q_pos, k_pos],
        vec![heads, prompt_len, prompt_len],
        DType::F32,
    );

    enc_g.output(norm);
    enc_g.output(bias);

    let enc_inputs = vec![
        Tensor::U32(vec![1, 5, 12, 3]),
        Tensor::U32(vec![0, 1, 2, 3]),
        Tensor::U32(vec![0, 1, 2, 3]),
    ];
    let enc_res = run_stateless(&enc_g, &enc_w, enc_inputs).expect("encoder run");
    assert_eq!(enc_res.len(), 2);
    assert_eq!(enc_res[0].len(), prompt_len * d_model);
    assert_eq!(enc_res[1].len(), heads * prompt_len * prompt_len);

    // 2. Decoder program: n=1 decode step with StateReorder
    let mut dec_g = Graph::new();
    let mut dec_w = Weights::new();

    let dec_in = dec_g.input("dec_token", vec![1], DType::U32);
    let dec_embed = dec_g.weight("shared.weight", vec![32, d_model], DType::F32);
    let dec_e = dec_g.add(
        "dec_embed",
        Op::Embedding,
        &[dec_embed, dec_in],
        vec![1, d_model],
        DType::F32,
    );
    dec_w.insert("shared.weight", Tensor::F32(vec![0.1; 32 * d_model]));

    // State reorder for beam search: 4 beams of 4 floats each = 16 floats
    let state_buf = dec_g.input("beam_states", vec![16], DType::F32);
    let parent = dec_g.input("parent_indices", vec![4], DType::U32);
    let reordered = dec_g.add(
        "reorder",
        Op::StateReorder,
        &[state_buf, parent],
        vec![16],
        DType::F32,
    );

    dec_g.output(dec_e);
    dec_g.output(reordered);

    let dec_inputs = vec![
        Tensor::U32(vec![2]),
        Tensor::F32(vec![
            0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0,
            33.0,
        ]),
        Tensor::U32(vec![3, 2, 1, 0]),
    ];
    let dec_res = run_stateless(&dec_g, &dec_w, dec_inputs).expect("decoder run");
    assert_eq!(dec_res.len(), 2);
    // Check reordering: row 0 should be old row 3 ([30.0, 31.0, 32.0, 33.0])
    let out_slice = dec_res[1].as_f32().unwrap();
    assert_eq!(&out_slice[0..4], &[30.0, 31.0, 32.0, 33.0]);
}
