// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feConvolveMatrix`.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feConvolveMatrixElement>
use super::bounds::PixelBounds;
use super::channels::{premultiply, store, stored_straight, working};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::filter::graph::{ColorSpace, MAX_KERNEL_ENTRIES};
use vello_common::filter_effects::{ConvolutionKernel, EdgeMode};
use vello_common::pixmap::Pixmap;

/// Convolve inside `bounds`, the input's pixels; `edge_mode` extends that rectangle.
/// The admitted kernel has `columns * rows` finite values and an in-range target.
pub(super) fn apply(
    pixels: &mut Pixmap,
    kernel: &ConvolutionKernel,
    bounds: PixelBounds,
    space: ColorSpace,
) {
    let width = usize::from(pixels.width());
    let entries = u64::from(kernel.columns) * u64::from(kernel.rows);
    // Chrome ignores kernels larger than its bound and passes the input through.
    if width == 0 || pixels.height() == 0 || entries > u64::from(MAX_KERNEL_ENTRIES) {
        return;
    }
    // With preserveAlpha, color is convolved unpremultiplied and alpha is kept.
    let input: Vec<[f32; 4]> = if kernel.preserve_alpha {
        pixels
            .data()
            .iter()
            .map(|p| stored_straight(*p, space))
            .collect()
    } else {
        working(pixels, space)
    };
    let columns = kernel.columns as usize;
    let rows = kernel.rows as usize;
    let mut output = vec![[0.0; 4]; input.len()];
    for (index, out) in output.iter_mut().enumerate() {
        let (x, y) = ((index % width) as i64, (index / width) as i64);
        let mut sum = [0.0_f32; 4];
        for i in 0..rows {
            for j in 0..columns {
                // The kernel is rotated by 180 degrees relative to its source order.
                let weight = kernel.values[(rows - 1 - i) * columns + (columns - 1 - j)];
                if weight == 0.0 {
                    continue;
                }
                let sx = x - i64::from(kernel.target_x) + j as i64;
                let sy = y - i64::from(kernel.target_y) + i as i64;
                if let Some(sample) = sample(&input, width, bounds, sx, sy, kernel.edge_mode) {
                    for c in 0..4 {
                        sum[c] += sample[c] * weight;
                    }
                }
            }
        }
        // The bias applies to every premultiplied channel, including alpha.
        *out = sum.map(|v| v / kernel.divisor + kernel.bias);
        if kernel.preserve_alpha {
            out[3] = input[index][3];
            *out = premultiply(out.map(|v| v.clamp(0.0, 1.0)));
        }
    }
    store(pixels, &output, space);
}

fn sample(
    data: &[[f32; 4]],
    width: usize,
    bounds: PixelBounds,
    x: i64,
    y: i64,
    edge: EdgeMode,
) -> Option<[f32; 4]> {
    let x = bounds.extend_x(x, edge)?;
    let y = bounds.extend_y(y, edge)?;
    Some(data[y * width + x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::PremulRgba8;

    fn kernel(columns: u32, rows: u32, values: Vec<f32>) -> ConvolutionKernel {
        ConvolutionKernel {
            columns,
            rows,
            values,
            target_x: 0,
            target_y: 0,
            divisor: 1.0,
            bias: 0.0,
            edge_mode: EdgeMode::None,
            preserve_alpha: false,
        }
    }

    #[test]
    fn kernels_shift_like_svg_and_oversized_kernels_pass_through() {
        let mut pixels = Pixmap::new(3, 1);
        pixels.data_mut()[2] = PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 1,
        };
        let mut oversized = pixels.clone();
        apply(
            &mut oversized,
            &kernel(257, 1, vec![0.0; 257]),
            bounds,
            ColorSpace::Srgb,
        );
        assert_eq!(oversized.data(), pixels.data());
        // The rotated kernel [1, 0] with target 0 reads the next pixel.
        apply(
            &mut pixels,
            &kernel(2, 1, vec![1.0, 0.0]),
            bounds,
            ColorSpace::Srgb,
        );
        assert_eq!(
            pixels.data().iter().map(|p| p.a).collect::<Vec<_>>(),
            [0, 255, 0]
        );
    }
}
