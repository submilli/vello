// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Float convolution shares the renderer's bounded Gaussian decimation plan.

use alloc::vec;
use alloc::vec::Vec;
use vello_common::filter::gaussian_blur::GaussianBlur;
use vello_common::filter_effects::EdgeMode;

type Pixel = [f32; 4];

pub(super) fn blur(data: &mut Vec<Pixel>, width: usize, height: usize, plan: &GaussianBlur) {
    if width == 0 || height == 0 || plan.std_deviation <= 0.0 {
        return;
    }
    let mut dimensions = Vec::with_capacity(plan.n_decimations);
    let (mut w, mut h) = (width, height);
    for _ in 0..plan.n_decimations {
        dimensions.push((w, h));
        let next_w = w.div_ceil(2);
        let next_h = h.div_ceil(2);
        let horizontal = downsample(data, w, h, next_w, h, true, plan.edge_mode);
        *data = downsample(
            &horizontal,
            next_w,
            h,
            next_w,
            next_h,
            false,
            plan.edge_mode,
        );
        (w, h) = (next_w, next_h);
    }
    let kernel = &plan.kernel[..usize::from(plan.kernel_size)];
    let horizontal = convolve(data, w, h, kernel, true, plan.edge_mode);
    *data = convolve(&horizontal, w, h, kernel, false, plan.edge_mode);
    while let Some((next_w, next_h)) = dimensions.pop() {
        let horizontal = upsample(data, w, h, next_w, h, true, plan.edge_mode);
        *data = upsample(
            &horizontal,
            next_w,
            h,
            next_w,
            next_h,
            false,
            plan.edge_mode,
        );
        (w, h) = (next_w, next_h);
    }
}

fn convolve(
    data: &[Pixel],
    w: usize,
    h: usize,
    kernel: &[f32],
    horizontal: bool,
    edge: EdgeMode,
) -> Vec<Pixel> {
    let mut output = vec![[0.0; 4]; w * h];
    let radius = (kernel.len() / 2) as i64;
    for y in 0..h {
        for x in 0..w {
            for (offset, weight) in kernel.iter().enumerate() {
                let delta = offset as i64 - radius;
                let p = sample(
                    data,
                    w,
                    h,
                    x as i64 + if horizontal { delta } else { 0 },
                    y as i64 + if horizontal { 0 } else { delta },
                    edge,
                );
                add(&mut output[y * w + x], p, *weight);
            }
        }
    }
    output
}

fn downsample(
    data: &[Pixel],
    w: usize,
    h: usize,
    next_w: usize,
    next_h: usize,
    horizontal: bool,
    edge: EdgeMode,
) -> Vec<Pixel> {
    let mut output = vec![[0.0; 4]; next_w * next_h];
    for y in 0..next_h {
        for x in 0..next_w {
            for (offset, weight) in [0.125, 0.375, 0.375, 0.125].into_iter().enumerate() {
                let delta = offset as i64 - 1;
                let sx = if horizontal {
                    2 * x as i64 + delta
                } else {
                    x as i64
                };
                let sy = if horizontal {
                    y as i64
                } else {
                    2 * y as i64 + delta
                };
                add(
                    &mut output[y * next_w + x],
                    sample(data, w, h, sx, sy, edge),
                    weight,
                );
            }
        }
    }
    output
}

fn upsample(
    data: &[Pixel],
    w: usize,
    h: usize,
    next_w: usize,
    next_h: usize,
    horizontal: bool,
    edge: EdgeMode,
) -> Vec<Pixel> {
    let mut output = vec![[0.0; 4]; next_w * next_h];
    for y in 0..next_h {
        for x in 0..next_w {
            let coordinate = if horizontal { x } else { y };
            let center = (coordinate / 2) as i64;
            let neighbor = center + if coordinate.is_multiple_of(2) { -1 } else { 1 };
            let (cx, cy, nx, ny) = if horizontal {
                (center, y as i64, neighbor, y as i64)
            } else {
                (x as i64, center, x as i64, neighbor)
            };
            add(
                &mut output[y * next_w + x],
                sample(data, w, h, cx, cy, edge),
                0.75,
            );
            add(
                &mut output[y * next_w + x],
                sample(data, w, h, nx, ny, edge),
                0.25,
            );
        }
    }
    output
}

fn sample(data: &[Pixel], w: usize, h: usize, x: i64, y: i64, edge: EdgeMode) -> Pixel {
    let coordinate = |value: i64, size: usize| -> Option<usize> {
        if size == 0 {
            return None;
        }
        match edge {
            EdgeMode::None if value < 0 || value >= size as i64 => None,
            EdgeMode::Wrap => Some(value.rem_euclid(size as i64) as usize),
            EdgeMode::Mirror => {
                let period = 2 * size as i64;
                let reflected = value.rem_euclid(period);
                Some(if reflected < size as i64 {
                    reflected
                } else {
                    period - reflected - 1
                } as usize)
            }
            EdgeMode::Duplicate | EdgeMode::None => Some(value.clamp(0, size as i64 - 1) as usize),
        }
    };
    match (coordinate(x, w), coordinate(y, h)) {
        (Some(x), Some(y)) => data[y * w + x],
        _ => [0.0; 4],
    }
}

fn add(output: &mut Pixel, input: Pixel, weight: f32) {
    for (out, value) in output.iter_mut().zip(input) {
        *out += value * weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mirror_samples_repeat_edges_without_clamping_away_reflection() {
        let data = [[0.0; 4], [0.25; 4], [1.0; 4]];
        for (x, expected) in [(-3, 1.0), (-2, 0.25), (-1, 0.0), (3, 1.0), (4, 0.25), (5, 0.0)] {
            assert_eq!(sample(&data, 3, 1, x, 0, EdgeMode::Mirror), [expected; 4]);
        }
    }
}
