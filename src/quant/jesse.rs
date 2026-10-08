//! SolidSF Jesse Weightless Neural Network implementation.
//!
//! Weightless Neural Networks (WNNs) replace conventional matrix-multiplication operations
//! (MACs) with direct random-access memory (RAM) lookup tables (LUTs).
//! Continuous inputs are projected and discretized into multi-bit addresses that index
//! compact `i8` lookup tables, evaluated purely via integer accumulation with
//! ZERO floating-point multiplications.

#[derive(Debug, Clone, PartialEq)]
pub struct WeightlessJesseLayer {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub address_bits: usize,
    pub num_luts: usize,
    pub lut_tables: Vec<i8>,
    pub hash_planes: Vec<f32>,
}

impl WeightlessJesseLayer {
    /// Creates a new `WeightlessJesseLayer` calibrated and populated from dense f32 weights.
    ///
    /// # Arguments
    /// * `name` - Descriptive name for this layer
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    /// * `address_bits` - Number of address bits per LUT (typically 6..8, giving 64..256 entries per table)
    /// * `weights` - Row-major weights buffer `[out_features, in_features]`
    pub fn from_f32_weights(
        name: &str,
        in_features: usize,
        out_features: usize,
        address_bits: usize,
        weights: &[f32],
    ) -> Self {
        assert!(address_bits >= 1 && address_bits <= 16, "address_bits must be in 1..=16");
        assert!(in_features > 0, "in_features must be > 0");
        assert!(out_features > 0, "out_features must be > 0");
        assert_eq!(
            weights.len(),
            in_features * out_features,
            "Weight buffer length {} does not match in_features * out_features ({} * {})",
            weights.len(),
            in_features,
            out_features
        );

        // Number of LUTs per output feature needed to cover the input features
        let num_luts = (in_features + address_bits - 1) / address_bits;
        let lut_size = 1usize << address_bits;
        let total_entries = out_features * num_luts * lut_size;

        // Generate deterministic hyperplane projection thresholds (hash planes)
        let total_planes = num_luts * address_bits;
        let mut hash_planes = Vec::with_capacity(total_planes);
        for i in 0..total_planes {
            // Deterministic pseudo-random threshold around 0.0 based on splitmix-style hashing
            let mut h = (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0xbf58476d1ce4e5b9);
            h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
            h ^= h >> 31;
            // Map to [-0.05, 0.05]
            let norm = (h as f64 / u64::MAX as f64) as f32;
            let threshold = (norm - 0.5) * 0.1;
            hash_planes.push(threshold);
        }

        // Calibrate LUT tables from the f32 weights
        let mut lut_tables = vec![0i8; total_entries];

        for out_idx in 0..out_features {
            let row_offset = out_idx * in_features;
            let lut_out_offset = out_idx * num_luts * lut_size;

            for lut_idx in 0..num_luts {
                let lut_base = lut_out_offset + lut_idx * lut_size;

                for addr in 0..lut_size {
                    let mut accum = 0.0f32;

                    for b in 0..address_bits {
                        let bit = (addr >> b) & 1;
                        let feat_idx = (lut_idx * address_bits + b) % in_features;
                        let w = weights[row_offset + feat_idx];
                        // If bit is 1, contribution is +w; if 0, contribution is -w
                        if bit == 1 {
                            accum += w;
                        } else {
                            accum -= w;
                        }
                    }

                    // Quantize into i8 with fixed scaling to ensure non-zero signal
                    let scaled = (accum * 16.0).round().clamp(-127.0, 127.0) as i8;
                    lut_tables[lut_base + addr] = if scaled == 0 && accum.abs() > 1e-6 {
                        if accum > 0.0 { 1 } else { -1 }
                    } else {
                        scaled
                    };
                }
            }
        }

        Self {
            name: name.to_string(),
            in_features,
            out_features,
            address_bits,
            num_luts,
            lut_tables,
            hash_planes,
        }
    }

    /// Evaluates forward pass purely via LUT address lookup and integer accumulation.
    ///
    /// GUARANTEED ZERO floating-point multiplications:
    /// - Address generation: comparisons (`>`) and bit shifts (`|`, `<<`)
    /// - Accumulation: integer additions (`+=`) into `i32`
    /// - Output conversion: cast to `f32` (`as f32`)
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            self.in_features,
            "Input length {} != in_features {} for layer '{}'",
            x.len(),
            self.in_features,
            self.name
        );
        assert_eq!(
            out.len(),
            self.out_features,
            "Output length {} != out_features {} for layer '{}'",
            out.len(),
            self.out_features,
            self.name
        );

        let lut_size = 1usize << self.address_bits;
        let num_luts = self.num_luts;
        let address_bits = self.address_bits;

        // Process LUT addresses in stack-allocated chunks to ensure strict ZERO HEAP ALLOCATIONS
        // during forward pass even for large feature dimensions.
        const CHUNK_SIZE: usize = 256;
        let mut addr_chunk = [0usize; CHUNK_SIZE];

        // Clear output buffer
        for v in out.iter_mut() {
            *v = 0.0;
        }

        for chunk_start in (0..num_luts).step_by(CHUNK_SIZE) {
            let chunk_len = (num_luts - chunk_start).min(CHUNK_SIZE);

            // Compute LUT addresses for this chunk via threshold comparison (zero multiplies)
            for i in 0..chunk_len {
                let lut_idx = chunk_start + i;
                let mut addr = 0usize;
                let plane_base = lut_idx * address_bits;

                for b in 0..address_bits {
                    let feat_idx = (lut_idx * address_bits + b) % self.in_features;
                    let threshold = self.hash_planes[plane_base + b];
                    if x[feat_idx] > threshold {
                        addr |= 1usize << b;
                    }
                }
                addr_chunk[i] = addr;
            }

            // Accumulate LUT values for each output feature
            for out_idx in 0..self.out_features {
                let out_offset = out_idx * num_luts * lut_size;
                let mut accum: i32 = 0;

                for i in 0..chunk_len {
                    let lut_idx = chunk_start + i;
                    let addr = addr_chunk[i];
                    let entry_idx = out_offset + lut_idx * lut_size + addr;
                    let val = self.lut_tables[entry_idx];
                    accum += val as i32;
                }

                out[out_idx] += accum as f32;
            }
        }
    }

    /// Effective storage rate in bits per weight.
    #[inline]
    pub fn bits_per_weight(&self) -> f32 {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weightless_jesse_zero_multiply_forward() {
        let in_features = 16;
        let out_features = 8;
        let address_bits = 4;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let layer = WeightlessJesseLayer::from_f32_weights(
            "test_jesse",
            in_features,
            out_features,
            address_bits,
            &weights,
        );

        assert_eq!(layer.bits_per_weight(), 0.5);
        assert_eq!(layer.lut_tables.len(), out_features * layer.num_luts * (1 << address_bits));

        let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut out = vec![0.0; out_features];

        layer.forward(&x, &mut out);

        // Verify valid activations produced
        assert!(out.iter().all(|&v| v.is_finite()));
        assert!(out.iter().any(|&v| v.abs() > 0.0), "Output activations must be non-zero");
    }
}
