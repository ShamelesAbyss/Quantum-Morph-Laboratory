//! Deterministic pseudorandom number generation for reproducible QML experiments.
//!
//! Uses SplitMix64 for portable, cross-platform deterministic sequences across
//! Windows and Linux without external dependencies.

/// Portable deterministic SplitMix64 PRNG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for exact IEEE-754 double precision mantissa.
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Standard normal sample via Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Uniform integer in `0..upper` (returns 0 if `upper == 0`).
    pub fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }
}

#[cfg(test)]
mod tests {
    use super::DeterministicRng;

    #[test]
    fn identical_seeds_produce_identical_streams() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn normal_samples_are_finite_with_reasonable_moments() {
        let mut rng = DeterministicRng::new(1337);
        let n = 4000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        assert!(samples.iter().all(|v| v.is_finite()));
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.08, "mean={mean}");
        assert!((var - 1.0).abs() < 0.12, "var={var}");
    }
}
