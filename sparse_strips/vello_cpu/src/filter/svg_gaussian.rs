// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Gaussian blur of 8-bit premultiplied pixels, pass for pass as Chrome (Skia) blurs.
//!
//! Each axis is blurred separately and stored at 8 bits, using the three-box
//! approximation that the specification suggests, in exact integer arithmetic.
//! Chrome builds Skia without its direct Gaussian pass for small deviations, so a
//! deviation whose box is a single pixel leaves that axis unchanged.
//! See: <https://drafts.fxtf.org/filter-effects/#feGaussianBlurElement>
use alloc::vec;
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::filter::gaussian_blur::GaussianBlur;
use vello_common::filter_effects::EdgeMode;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

/// The largest deviation the three-box pass handles; Chrome rescales larger blurs.
const BOX_SIGMA: f32 = 135.0;
/// Larger deviations already spread any admitted raster to near-transparency, and
/// capping keeps the decimation plan's variance finite.
const MAX_SIGMA: f32 = 65_536.0;

/// Blur `pixels` (row-major, `width` columns) by per-axis deviations.
pub(super) fn blur(pixels: &mut [PremulRgba8], width: usize, std_deviation: [f32; 2]) {
    if width == 0 || pixels.is_empty() {
        return;
    }
    let height = pixels.len() / width;
    let mut line = Vec::new();
    if let Some(pass) = Pass::new(std_deviation[0]) {
        for row in pixels.chunks_exact_mut(width) {
            pass.apply(row, &mut line);
        }
    }
    if let Some(pass) = Pass::new(std_deviation[1]) {
        let mut column = vec![
            PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            height
        ];
        for x in 0..width {
            for (y, value) in column.iter_mut().enumerate() {
                *value = pixels[y * width + x];
            }
            pass.apply(&mut column, &mut line);
            for (y, value) in column.iter().enumerate() {
                pixels[y * width + x] = *value;
            }
        }
    }
}

enum Pass {
    /// Three running box sums of `window`, divided by a scaled divisor.
    Boxes {
        window: usize,
        factor: u64,
        half: u64,
    },
    /// Beyond the box range, the renderer's bounded decimated Gaussian.
    Decimated(GaussianBlur),
}

impl Pass {
    fn new(sigma: f32) -> Option<Self> {
        if sigma > BOX_SIGMA {
            let sigma = sigma.min(MAX_SIGMA);
            return Some(Self::Decimated(GaussianBlur::new(sigma, EdgeMode::None)));
        }
        let window = window(sigma);
        (window > 1).then(|| boxes(window))
    }

    /// Replace `pixels` with their blur; samples outside are transparent.
    fn apply(&self, pixels: &mut [PremulRgba8], scratch: &mut Vec<[f32; 4]>) {
        match self {
            Self::Boxes {
                window,
                factor,
                half,
            } => boxes_pass(pixels, *window, *factor, *half),
            Self::Decimated(plan) => {
                scratch.clear();
                scratch.extend(pixels.iter().map(|p| unit(*p)));
                let identity = GaussianBlur::new(0.0, EdgeMode::None);
                super::float_blur::blur(scratch, pixels.len(), 1, plan, &identity);
                for (out, value) in pixels.iter_mut().zip(scratch.iter()) {
                    *out = round_half_up(*value);
                }
            }
        }
    }
}

fn unit(pixel: PremulRgba8) -> [f32; 4] {
    [pixel.r, pixel.g, pixel.b, pixel.a].map(|v| f32::from(v) * (1.0 / 255.0))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Values are clamped to the 8-bit range before truncation."
)]
fn round_half_up(value: [f32; 4]) -> PremulRgba8 {
    let [r, g, b, a] = value.map(|v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
    PremulRgba8 { r, g, b, a }
}

/// The specification's box size: `floor(sigma * 3 * sqrt(2 * pi) / 4 + 0.5)`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Box deviations are at most 135, so windows are below 255."
)]
fn window(sigma: f32) -> usize {
    let size = (sigma * 3.0 * (2.0 * core::f32::consts::PI).sqrt() / 4.0 + 0.5).floor();
    (size as usize).max(1)
}

/// Skia's scaled divider for three boxes of `window`; even windows widen the last box.
fn boxes(window: usize) -> Pass {
    let last = if window % 2 == 1 { window } else { window + 1 };
    let divisor = (window * window * last) as u64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "The scaled reciprocal of a divisor above one fits in 32 bits."
    )]
    let factor = ((1.0 / divisor as f64) * (1_u64 << 32) as f64).round() as u64;
    Pass::Boxes {
        window,
        factor,
        half: (divisor + 1) >> 1,
    }
}

/// Stream three running sums, as Skia does: each sum feeds the next, and ring
/// buffers retire trailing edges. Output `i` reads the window centered on it.
fn boxes_pass(pixels: &mut [PremulRgba8], window: usize, factor: u64, half: u64) {
    let even = window.is_multiple_of(2);
    let border = if even {
        3 * (window / 2) - 1
    } else {
        3 * ((window - 1) / 2)
    };
    let lengths = [
        window - 1,
        window - 1,
        if even { window } else { window - 1 },
    ];
    let mut rings = lengths.map(|length| vec![[0_u64; 4]; length]);
    let mut cursors = [0_usize; 3];
    let (mut sum0, mut sum1, mut sum2) = ([0_u64; 4], [0_u64; 4], [half; 4]);
    let input: Vec<[u64; 4]> = pixels
        .iter()
        .map(|p| [p.r, p.g, p.b, p.a].map(u64::from))
        .collect();
    for t in 0..pixels.len() + border {
        let leading = input.get(t).copied().unwrap_or([0; 4]);
        for c in 0..4 {
            sum0[c] += leading[c];
            sum1[c] += sum0[c];
            sum2[c] += sum1[c];
        }
        if let Some(out) = t.checked_sub(border).and_then(|i| pixels.get_mut(i)) {
            // Weights sum to the divisor, so results stay within 8 bits.
            let [r, g, b, a] = sum2.map(|v| ((v * factor) >> 32).min(255) as u8);
            *out = PremulRgba8 { r, g, b, a };
        }
        for (ring, (cursor, (sum, entering))) in rings.iter_mut().zip(cursors.iter_mut().zip([
            (&mut sum2, sum1),
            (&mut sum1, sum0),
            (&mut sum0, leading),
        ])) {
            if ring.is_empty() {
                continue;
            }
            for c in 0..4 {
                sum[c] -= ring[*cursor][c];
            }
            ring[*cursor] = entering;
            *cursor = (*cursor + 1) % ring.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha(values: &[u8]) -> Vec<PremulRgba8> {
        values
            .iter()
            .map(|a| PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: *a,
            })
            .collect()
    }

    #[test]
    fn box_passes_match_the_direct_three_box_convolution() {
        assert_eq!(window(2.0), 4);
        assert_eq!(window(10.0), 19);
        for size in [4, 5] {
            let Pass::Boxes {
                window,
                factor,
                half,
            } = boxes(size)
            else {
                panic!("box pass")
            };
            let input: Vec<u8> = (0..23).map(|i| (i * 37 % 256) as u8).collect();
            let mut pixels = alpha(&input);
            boxes_pass(&mut pixels, window, factor, half);
            // Convolve three boxes directly, the last widened for even windows.
            let mut weights = vec![1_u64];
            for box_size in [size, size, if size % 2 == 0 { size + 1 } else { size }] {
                let mut next = vec![0; weights.len() + box_size - 1];
                for (i, w) in weights.iter().enumerate() {
                    for v in &mut next[i..i + box_size] {
                        *v += w;
                    }
                }
                weights = next;
            }
            let border = weights.len() / 2;
            for (i, pixel) in pixels.iter().enumerate() {
                let sum: u64 = weights
                    .iter()
                    .enumerate()
                    .filter_map(|(k, w)| {
                        let j = (i + k).checked_sub(border)?;
                        Some(w * u64::from(*input.get(j)?))
                    })
                    .sum();
                assert_eq!(
                    u64::from(pixel.a),
                    ((sum + half) * factor) >> 32,
                    "{size} {i}"
                );
            }
        }
    }

    #[test]
    fn deviations_with_single_pixel_boxes_are_identity() {
        assert!(Pass::new(0.5).is_none());
        assert!(Pass::new(0.8).is_some());
        let mut pixels = alpha(&[0, 0, 0, 255, 0, 0, 0]);
        blur(&mut pixels, 7, [0.5, 0.5]);
        assert_eq!(pixels[3].a, 255);
        blur(&mut pixels, 7, [1.0, 0.0]);
        // Window 2: boxes of 2, 2 and 3 over 12, centered on the source pixel.
        assert_eq!(
            pixels.iter().map(|p| p.a).collect::<Vec<_>>(),
            [0, 21, 64, 85, 64, 21, 0]
        );
    }

    #[test]
    fn box_blurs_preserve_energy_and_huge_deviations_stay_bounded() {
        let mut pixels = alpha(&[0; 64]);
        pixels[32].a = 255;
        blur(&mut pixels, 64, [3.0, 0.0]);
        let energy: u32 = pixels.iter().map(|p| u32::from(p.a)).sum();
        assert!((250..=260).contains(&energy), "{energy}");
        let mut wide = alpha(&[255; 9]);
        blur(&mut wide, 3, [500.0, 500.0]);
        assert!(wide.iter().all(|p| p.a < 255));
        // Squaring this deviation overflows; the pass must still terminate.
        blur(&mut wide, 3, [f32::MAX, f32::MAX]);
        assert!(wide.iter().all(|p| p.a == 0));
    }
}
