//! Real-weight check of every GGUF block decoder: each quantized tensor of a published quant file
//! is dequantized and compared with the same tensor of the model's f16/bf16 GGUF. A wrong bit layout
//! gives relative RMS error ≈ 1.0+; correct decoders land in ggml's known quality ordering.
use gibbs_engine::model::loaders::gguf::GgufWeightReader;
use gibbs_engine::model::loaders::WeightReader;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn cached(repo: &str, file: &str) -> PathBuf {
    let (org, name) = repo.split_once('/').unwrap();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("disk/models--{org}--{name}/snapshots/main/{file}"))
}

/// Upper bound on relative RMS error per ggml tensor type (calibrated on correct decoders).
fn bound(dtype: &str) -> f32 {
    match dtype {
        "Q8_0" => 0.01,
        "Q6_K" => 0.03,
        "Q5_0" | "Q5_1" | "Q5_K" => 0.05,
        "Q4_0" | "Q4_1" | "Q4_K" | "IQ4_NL" | "IQ4_XS" => 0.12,
        "MXFP4" => 0.16, // power-of-two block scale only: coarser than Q4_0
        "Q3_K" | "IQ3_S" | "IQ3_XXS" => 0.2,
        "Q2_K" | "IQ2_S" | "IQ2_XS" | "IQ2_XXS" => 0.45,
        "IQ1_S" | "IQ1_M" => 0.6,
        "TQ1_0" | "TQ2_0" => 0.05,
        _ => 0.0,
    }
}

#[test]
fn gguf_quant_decoders_match_float_reference() {
    const SMOL_B: &str = "bartowski/SmolLM2-135M-Instruct-GGUF";
    const SMOL_M: &str = "mradermacher/SmolLM2-135M-Instruct-i1-GGUF";
    const TRILM: &str = "Mungert/TriLM_1.1B_Unpacked-GGUF";
    let smol_ref = cached(SMOL_B, "SmolLM2-135M-Instruct-f16.gguf");
    let trilm_ref = cached(TRILM, "TriLM_1.1B_Unpacked-bf16.gguf");
    // No small MXFP4 file with a float twin exists; the Q8_0 twin (rel RMS ≈ 0.006) stands in.
    let thinker_ref = cached(
        "mradermacher/SmallThinker-4BA0.6B-Instruct-GGUF",
        "SmallThinker-4BA0.6B-Instruct.Q8_0.gguf",
    );
    let mut cases: Vec<(PathBuf, &PathBuf)> = Vec::new();
    for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ3_XS", "IQ3_M", "IQ4_XS", "Q4_0", "Q4_K_M", "Q8_0"] {
        cases.push((cached(SMOL_B, &format!("SmolLM2-135M-Instruct-{q}.gguf")), &smol_ref));
    }
    for q in ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S"] {
        cases.push((cached(SMOL_M, &format!("SmolLM2-135M-Instruct.i1-{q}.gguf")), &smol_ref));
    }
    for q in ["tq1_0", "tq2_0", "iq4_nl"] {
        cases.push((cached(TRILM, &format!("TriLM_1.1B_Unpacked-{q}.gguf")), &trilm_ref));
    }
    cases.push((
        cached(
            "noctrex/SmallThinker-4B-A0.6B-Instruct-MXFP4_MOE-GGUF",
            "SmallThinker-4B-A0.6B-Instruct-MXFP4_MOE.gguf",
        ),
        &thinker_ref,
    ));

    let mut readers: BTreeMap<PathBuf, GgufWeightReader> = BTreeMap::new();
    let mut checked = 0;
    for (file, reference) in &cases {
        if !file.exists() || !reference.exists() {
            println!("SKIP {} (not in disk/)", file.display());
            continue;
        }
        if !readers.contains_key(*reference) {
            let r = GgufWeightReader::new(std::fs::read(reference).unwrap()).unwrap();
            readers.insert((*reference).clone(), r);
        }
        let reference = &readers[*reference];
        let quant = GgufWeightReader::new(std::fs::read(file).unwrap()).unwrap();
        // Per ggml type: (sum err², sum ref², tensors).
        let mut per_type: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
        for t in quant.list_tensors().unwrap() {
            if matches!(t.dtype.as_str(), "F32" | "F16" | "BF16") {
                continue;
            }
            let got = quant.load_tensor(&t.name).unwrap();
            let want = reference.load_tensor(&t.name).unwrap();
            assert_eq!(got.len(), want.len(), "{}", t.name);
            let e = per_type.entry(t.dtype.clone()).or_default();
            for (g, w) in got.iter().zip(&want) {
                e.0 += ((g - w) as f64).powi(2);
                e.1 += (*w as f64).powi(2);
            }
            e.2 += 1;
        }
        let name = file.file_name().unwrap().to_string_lossy();
        for (dtype, (err, norm, n)) in per_type {
            let rel = (err / norm).sqrt() as f32;
            println!("{name:<60} {dtype:<8} tensors={n:<4} rel_rms={rel:.4}");
            assert!(rel < bound(&dtype), "{name} {dtype}: rel RMS {rel} >= {}", bound(&dtype));
            checked += 1;
        }
    }
    println!("checked {checked} (file, type) pairs");
}

#[test]
fn undecodable_ggml_types_are_named_errors() {
    use gibbs_engine::model::UniversalModelLoader;
    for (id, name) in [(15, "Q8_K"), (31, "Q4_0_x_y")] {
        let err = UniversalModelLoader::ggml_type_info(id).unwrap_err();
        assert!(err.contains(name), "{err}");
    }
    assert!(UniversalModelLoader::ggml_type_info(19).is_ok()); // IQ1_S decodes
    // NVFP4 (type 40): 64 elements per 36-byte block, layout confirmed from ggml-common.h
    let (_, _, elems, bytes) = UniversalModelLoader::ggml_type_info(40).unwrap();
    assert_eq!((elems, bytes), (64, 36));
}
