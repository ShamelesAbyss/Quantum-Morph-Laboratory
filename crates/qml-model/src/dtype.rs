//! Exact, safe numerical data-type conversion for Safetensors tensor payloads.
//!
//! All conversions are pure safe Rust with bit-accurate IEEE-754 half-precision
//! (`F16`), brain-float (`BF16`), OCP FP8 (`F8_E4M3`, `F8_E5M2`), standard
//! floating-point, integer, and boolean support. Unknown or unimplemented
//! dtypes fail explicitly rather than being guessed.

use std::fmt;

/// Known Safetensors tensor element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    F8E4M3Fnuz,
    F8E5M2Fnuz,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
    C64,
    F4,
    F6E2M3,
    F6E3M2,
}

impl DType {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "F8_E4M3FNUZ" => Self::F8E4M3Fnuz,
            "F8_E5M2FNUZ" => Self::F8E5M2Fnuz,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F64" => Self::F64,
            "C64" => Self::C64,
            "F4" => Self::F4,
            "F6_E2M3" => Self::F6E2M3,
            "F6_E3M2" => Self::F6E3M2,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::F8E4M3Fnuz => "F8_E4M3FNUZ",
            Self::F8E5M2Fnuz => "F8_E5M2FNUZ",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::F4 => "F4",
            Self::F6E2M3 => "F6_E2M3",
            Self::F6E3M2 => "F6_E3M2",
        }
    }

    pub fn bit_width(self) -> u64 {
        match self {
            Self::F4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E5M2
            | Self::F8E4M3
            | Self::F8E8M0
            | Self::F8E4M3Fnuz
            | Self::F8E5M2Fnuz => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 64,
        }
    }

    /// Byte size per element for byte-aligned dtypes (`None` for sub-byte dtypes).
    pub fn byte_width(self) -> Option<usize> {
        let bits = self.bit_width();
        if bits.is_multiple_of(8) {
            Some((bits / 8) as usize)
        } else {
            None
        }
    }

    /// Whether this dtype supports scalar real-valued decoding to `f64`.
    pub fn supports_f64_decode(self) -> bool {
        matches!(
            self,
            Self::Bool
                | Self::U8
                | Self::I8
                | Self::F8E5M2
                | Self::F8E4M3
                | Self::I16
                | Self::U16
                | Self::F16
                | Self::BF16
                | Self::I32
                | Self::U32
                | Self::F32
                | Self::I64
                | Self::U64
                | Self::F64
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Decode IEEE-754 binary16 (`F16`) bits to `f32`.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x03ff);

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal binary16: value = (-1)^s * 2^-14 * (frac / 1024)
            let mut m = frac;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f32_exp = ((e + 127) as u32) << 23;
            sign | f32_exp | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN (preserve quiet NaN payload if non-zero)
        let nan_payload = if frac != 0 {
            (frac << 13) | 0x0040_0000
        } else {
            0
        };
        sign | 0x7f80_0000 | nan_payload
    } else {
        let f32_exp = (exp + (127 - 15)) << 23;
        sign | f32_exp | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// Encode `f32` to IEEE-754 binary16 (`F16`) using round-to-nearest-even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN
        if frac == 0 {
            return sign | 0x7c00;
        }
        let payload = ((frac >> 13) as u16) | 0x0200;
        return sign | 0x7c00 | payload;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow to +/- Inf
        return sign | 0x7c00;
    }
    if unbiased < -24 {
        // Underflow to +/- 0
        return sign;
    }
    if unbiased < -14 {
        // Subnormal f16
        let full_mantissa = frac | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let half = 1_u32 << (shift - 1);
        let remainder = full_mantissa & ((1_u32 << shift) - 1);
        let mut mant = full_mantissa >> shift;
        if remainder > half || (remainder == half && (mant & 1) == 1) {
            mant += 1;
        }
        return sign | (mant as u16);
    }

    // Normal f16
    let h_exp = (unbiased + 15) as u32;
    let round_bit = 0x0000_1000_u32;
    let remainder = frac & 0x0000_1fff;
    let mut h_mant = frac >> 13;
    if remainder > round_bit || (remainder == round_bit && (h_mant & 1) == 1) {
        h_mant += 1;
        if h_mant == 0x0400 {
            let next_exp = h_exp + 1;
            if next_exp >= 0x1f {
                return sign | 0x7c00;
            }
            return sign | ((next_exp as u16) << 10);
        }
    }
    sign | ((h_exp as u16) << 10) | (h_mant as u16)
}

/// Decode `BF16` bits to `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encode `f32` to `BF16` using round-to-nearest-even.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Decode OCP FP8 `F8_E4M3` (E4M3FN: bias 7, NaN=0x7f/0xff, no infinity) to `f32`.
pub fn f8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if (bits & 0x80) != 0 {
        -1.0_f32
    } else {
        1.0_f32
    };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        if mant == 0 {
            return if (bits & 0x80) != 0 { -0.0 } else { 0.0 };
        }
        // Subnormal: (-1)^s * 2^-6 * (mant / 8)
        return sign * (f32::from(mant) / 8.0) * 2.0_f32.powi(-6);
    }
    let unbiased = i32::from(exp) - 7;
    sign * (1.0 + f32::from(mant) / 8.0) * 2.0_f32.powi(unbiased)
}

/// Decode IEEE-like FP8 `F8_E5M2` (bias 15, Inf/NaN when exp==31) to `f32`.
pub fn f8_e5m2_to_f32(bits: u8) -> f32 {
    let sign = if (bits & 0x80) != 0 {
        -1.0_f32
    } else {
        1.0_f32
    };
    let exp = (bits >> 2) & 0x1f;
    let mant = bits & 0x03;
    if exp == 0x1f {
        return if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        if mant == 0 {
            return if (bits & 0x80) != 0 { -0.0 } else { 0.0 };
        }
        return sign * (f32::from(mant) / 4.0) * 2.0_f32.powi(-14);
    }
    let unbiased = i32::from(exp) - 15;
    sign * (1.0 + f32::from(mant) / 4.0) * 2.0_f32.powi(unbiased)
}

/// Decode a byte slice of little-endian elements of `dtype` into `out` (`f64`).
///
/// Returns `Err` if `dtype` does not support scalar `f64` decoding or if `bytes.len()`
/// is not an exact multiple of the element byte width.
pub fn decode_chunk_to_f64(dtype: DType, bytes: &[u8], out: &mut Vec<f64>) -> Result<(), String> {
    if !dtype.supports_f64_decode() {
        return Err(format!(
            "dtype {dtype} does not support scalar f64 decoding"
        ));
    }
    let elem_bytes = dtype
        .byte_width()
        .ok_or_else(|| format!("dtype {dtype} is sub-byte and not byte-aligned per element"))?;
    if !bytes.len().is_multiple_of(elem_bytes) {
        return Err(format!(
            "byte slice length {} is not a multiple of element width {} for dtype {dtype}",
            bytes.len(),
            elem_bytes
        ));
    }
    let count = bytes.len() / elem_bytes;
    out.clear();
    out.reserve(count);

    match dtype {
        DType::Bool => {
            for &b in bytes {
                out.push(if b == 0 { 0.0 } else { 1.0 });
            }
        }
        DType::U8 => {
            for &b in bytes {
                out.push(f64::from(b));
            }
        }
        DType::I8 => {
            for &b in bytes {
                out.push(f64::from(b as i8));
            }
        }
        DType::F8E4M3 => {
            for &b in bytes {
                out.push(f64::from(f8_e4m3_to_f32(b)));
            }
        }
        DType::F8E5M2 => {
            for &b in bytes {
                out.push(f64::from(f8_e5m2_to_f32(b)));
            }
        }
        DType::I16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f64::from(i16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        DType::U16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f64::from(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        DType::F16 => {
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f64::from(f16_to_f32(bits)));
            }
        }
        DType::BF16 => {
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f64::from(bf16_to_f32(bits)));
            }
        }
        DType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f64::from(i32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        DType::U32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f64::from(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        DType::F32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f64::from(f32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        DType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(i64::from_le_bytes(chunk.try_into().expect("8 bytes")) as f64);
            }
        }
        DType::U64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(u64::from_le_bytes(chunk.try_into().expect("8 bytes")) as f64);
            }
        }
        DType::F64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(f64::from_le_bytes(chunk.try_into().expect("8 bytes")));
            }
        }
        _ => unreachable!("checked supports_f64_decode above"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_special_normal_and_subnormal_values() {
        let cases: &[(u16, f32)] = &[
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x4000, 2.0),
            (0x3800, 0.5),
            (0x7bff, 65504.0),
            (0x0400, 6.1035156e-5), // smallest positive normal
            (0x0001, 5.9604645e-8), // smallest positive subnormal
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ];
        for &(bits, expected) in cases {
            let decoded = f16_to_f32(bits);
            assert_eq!(decoded.to_bits(), expected.to_bits(), "bits={bits:#06x}");
            let reencoded = f32_to_f16(expected);
            assert_eq!(reencoded, bits, "reencode expected={expected}");
        }
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_round_trips_and_rounds_to_nearest_even() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-3.5), 0xc060);
        assert_eq!(bf16_to_f32(f32_to_bf16(-3.5)), -3.5);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn fp8_decodes_known_values() {
        assert_eq!(f8_e4m3_to_f32(0x38), 1.0); // exp=7, mant=0 -> 2^0 * 1.0
        assert_eq!(f8_e4m3_to_f32(0x40), 2.0); // exp=8, mant=0
        assert_eq!(f8_e4m3_to_f32(0x7e), 448.0); // max finite E4M3FN
        assert!(f8_e4m3_to_f32(0x7f).is_nan());
        assert_eq!(f8_e5m2_to_f32(0x3c), 1.0); // exp=15, mant=0
        assert!(f8_e5m2_to_f32(0x7c).is_infinite());
        assert!(f8_e5m2_to_f32(0x7d).is_nan());
    }

    #[test]
    fn decode_chunk_rejects_unsupported_and_misaligned_inputs() {
        let mut out = Vec::new();
        assert!(decode_chunk_to_f64(DType::F4, &[0x12], &mut out).is_err());
        assert!(decode_chunk_to_f64(DType::C64, &[0; 8], &mut out).is_err());
        assert!(decode_chunk_to_f64(DType::F32, &[0; 3], &mut out).is_err());
        decode_chunk_to_f64(DType::F16, &0x3c00_u16.to_le_bytes(), &mut out).unwrap();
        assert_eq!(out, vec![1.0]);
    }
}
