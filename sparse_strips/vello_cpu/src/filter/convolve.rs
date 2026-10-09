// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feConvolveMatrix`.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feConvolveMatrixElement>
use super::bounds::PixelBounds;
use super::channels::{premultiply, store, stored_straight, working};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::filter::graph::ColorSpace;
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
    if width == 0 || pixels.height() == 0 {
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
