// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Gaussian blur of 8-bit premultiplied pixels, pass for pass as Chrome (Skia) blurs.
//!
//! Each axis is blurred separately and stored at 8 bits, using the three-box
//! approximation that the specification suggests, in exact integer arithmetic.
//! Chrome builds Skia without its direct Gaussian pass for small deviations, so a
//! deviation whose box is a single pixel leaves that axis unchanged. Deviations
//! beyond the box range blur a rescaled image (`svg_rescale`).
//! See: <https://drafts.fxtf.org/filter-effects/#feGaussianBlurElement>
use super::bounds::{PixelBounds, for_each_column};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::filter::graph::MAX_BLUR_DEVIATION;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

/// The largest deviation Skia's raster three-box pass handles.
pub(super) const BOX_SIGMA: f32 = 135.0;

/// Blur `pixels` (row-major, `width` columns) by per-axis deviations; `layer`,
/// the pixels the input may hold, centers any rescale.
pub(super) fn blur(
    pixels: &mut [PremulRgba8],
    width: usize,
    std_deviation: [f32; 2],
    layer: PixelBounds,
) {
    if width == 0 || pixels.is_empty() {
        return;
    }
    let sigma = std_deviation.map(|s| s.min(MAX_BLUR_DEVIATION));
    if !super::svg_rescale::try_blur(pixels, width, sigma, layer, box_blur) {
        box_blur(pixels, width, sigma.map(|s| s.min(BOX_SIGMA)));
    }
}

/// Three-box blur of each axis by a deviation of at most [`BOX_SIGMA`].
fn box_blur(pixels: &mut [PremulRgba8], width: usize, std_deviation: [f32; 2]) {
    if width == 0 || pixels.is_empty() {
        return;
    }
    if let Some(pass) = Pass::new(std_deviation[0]) {
        for row in pixels.chunks_exact_mut(width) {
            pass.apply(row);
        }
    }
    if let Some(pass) = Pass::new(std_deviation[1]) {
        for_each_column(pixels, width, |column| pass.apply(column));
    }
}

/// Three running box sums of `window`, divided by a scaled divisor.
struct Pass {
    window: usize,
    factor: u64,
    half: u64,
}

impl Pass {
    fn new(sigma: f32) -> Option<Self> {
        let window = window(sigma);
        (window > 1).then(|| boxes(window))
    }

    /// Replace `pixels` with their blur; samples outside are transparent.
    fn apply(&self, pixels: &mut [PremulRgba8]) {
        boxes_pass(pixels, self.window, self.factor, self.half);
    }
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
    Pass {
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

    fn whole(x1: usize, y1: usize) -> PixelBounds {
        PixelBounds {
            x0: 0,
            y0: 0,
            x1,
            y1,
        }
    }

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
            let Pass {
                window,
                factor,
                half,
            } = boxes(size);
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
        blur(&mut pixels, 7, [0.5, 0.5], whole(7, 1));
        assert_eq!(pixels[3].a, 255);
        blur(&mut pixels, 7, [1.0, 0.0], whole(7, 1));
        // Window 2: boxes of 2, 2 and 3 over 12, centered on the source pixel.
        assert_eq!(
            pixels.iter().map(|p| p.a).collect::<Vec<_>>(),
            [0, 21, 64, 85, 64, 21, 0]
        );
    }

    #[test]
    fn blurs_preserve_energy_and_huge_deviations_clamp() {
        let mut pixels = alpha(&[0; 64]);
        pixels[32].a = 255;
        blur(&mut pixels, 64, [3.0, 0.0], whole(64, 1));
        let energy: u32 = pixels.iter().map(|p| u32::from(p.a)).sum();
        assert!((250..=260).contains(&energy), "{energy}");
        let mut wide = alpha(&[255; 9]);
        blur(&mut wide, 3, [500.0, 500.0], whole(3, 3));
        assert!(wide.iter().all(|p| p.a < 255));
        // Deviations clamp to Skia's 532, so even this takes the bounded rescale.
        blur(&mut wide, 3, [f32::MAX, f32::MAX], whole(3, 3));
        assert!(wide.iter().all(|p| p.a == 0));
        let strip = |sigma| {
            let mut pixels = alpha(&[0; 1500]);
            pixels[600..900].iter_mut().for_each(|p| p.a = 255);
            blur(&mut pixels, 1500, [sigma, 0.0], whole(1500, 1));
            pixels
        };
        assert_eq!(strip(1000.0), strip(532.0));
        assert_ne!(strip(531.0), strip(532.0));
    }

    #[test]
    fn rescaled_blurs_match_chrome() {
        // Chrome 154 (`blurs.js` in submilli-browser): a 200-pixel strip blurred
        // by 150 on a 600-pixel row, sampled every 40 pixels (one rescale step),
        // and a 500-pixel strip by 300.5, sampled every 30 pixels from 300 (two).
        for (width, content, sigma, start, step, chrome) in [
            (600, 200..400, 150.0, 0, 40, &CHROME_ONE_STEP[..]),
            (1500, 500..1000, 300.5, 300, 30, &CHROME_TWO_STEPS[..]),
        ] {
            let mut pixels = alpha(&vec![0; width]);
            for pixel in &mut pixels[content.clone()] {
                pixel.a = 255;
            }
            let bounds = PixelBounds {
                x0: content.start,
                y0: 0,
                x1: content.end,
                y1: 1,
            };
            blur(&mut pixels, width, [sigma, 0.0], bounds);
            for (index, expected) in chrome.iter().enumerate() {
                let actual = pixels[start + index * step].a;
                assert!(actual.abs_diff(*expected) <= 1, "{sigma} {index}: {actual}");
            }
        }
    }

    /// Chrome's `blurBeyondBoxRange` alpha from x = 0 to 560.
    const CHROME_ONE_STEP: [u8; 15] = [
        21, 34, 50, 69, 89, 106, 120, 127, 127, 120, 106, 89, 69, 50, 34,
    ];
    /// Chrome's `twoSteps` alpha from x = 300 to 630.
    const CHROME_TWO_STEPS: [u8; 12] = [63, 71, 79, 87, 96, 104, 112, 120, 127, 133, 139, 144];
}
