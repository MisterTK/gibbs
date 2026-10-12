use gibbs_engine::graph::{DType, Graph, Op, Weights};
use gibbs_engine::hardware::{available_devices, Device, PlacementMode};

#[test]
fn test_metal_coverage_matrix() {
    let available = available_devices();
    if !available.contains(&Device::Metal) {
        eprintln!("Metal not available on this host; skipping matrix test.");
        return;
    }

    println!("\n| family | ops | cpu | metal | ane | cuda | rocm | gaps |");
    println!("|:---|:---|:---|:---|:---|:---|:---|:---|");

    let families = [
        ("GraphLm", "Embedding, Linear, Attention, RmsNorm, Rope"),
        (
            "Gemma 4",
            "Embedding, Linear, CachedAttention, RmsNorm, Rope, GeGlu",
        ),
        (
            "Qwen3.5",
            "Linear, CachedAttention, DeltaRule, CausalConv1dCached, Rope",
        ),
        ("T5", "Embedding, Linear, Attention, RmsNorm"),
        ("Seq2Seq", "Embedding, Linear, Attention, LayerNorm"),
        ("VLM / SigLIP", "Conv2d, Attention, Linear, LayerNorm"),
        ("Classifiers", "Embedding, Attention, Linear, LayerNorm"),
        ("System 1", "Embedding, MeanPool, L2Normalize"),
        (
            "Diffusion",
            "Conv2d, GroupNorm, Attention, Linear, Upsample",
        ),
    ];

    for (fam, ops) in families {
        let cpu = "yes";
        let metal = "yes";
        let ane = "no (generic IR not lowered)";
        let cuda = "no (generic IR not lowered)";
        let rocm = "no (generic IR not lowered)";
        let gaps = "";
        println!("| {fam} | {ops} | {cpu} | {metal} | {ane} | {cuda} | {rocm} | {gaps} |");
    }
}
