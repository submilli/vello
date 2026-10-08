// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG component transfer and separable blend operations.
use super::channels::{encode, straight};
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::TransferFunction;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::peniko::Mix;
use vello_common::pixmap::Pixmap;

pub(super) fn transfer(
    pixels: &mut Pixmap,
    functions: [&Option<TransferFunction>; 4],
    space: ColorSpace,
) {
    for pixel in pixels.data_mut() {
        let mut channels = straight(*pixel, space);
        for (channel, function) in channels.iter_mut().zip(functions) {
            if let Some(function) = function {
                *channel = evaluate(function, *channel).clamp(0.0, 1.0);
            }
        }
        *pixel = encode(channels, space);
    }
    pixels.recompute_may_have_transparency();
}

fn evaluate(function: &TransferFunction, value: f32) -> f32 {
    match function {
        TransferFunction::Identity => value,
        TransferFunction::Linear { slope, intercept } => slope * value + intercept,
        TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => amplitude * value.powf(*exponent) + offset,
        TransferFunction::Table { values } => {
            if values.is_empty() {
                return value;
            }
            let position = value.clamp(0.0, 1.0) * (values.len() - 1) as f32;
            let index = position as usize;
            let next = (index + 1).min(values.len() - 1);
            values[index] + (values[next] - values[index]) * (position - index as f32)
        }
        TransferFunction::Discrete { values } => {
            if values.is_empty() {
                return value;
            }
            values[((value.clamp(0.0, 1.0) * values.len() as f32) as usize).min(values.len() - 1)]
        }
    }
}

pub(super) fn blend(pixels: &mut Pixmap, other: &Pixmap, mode: Mix, space: ColorSpace) {
    for (pixel, other) in pixels.data_mut().iter_mut().zip(other.data()) {
        let s = straight(*pixel, space);
        let d = straight(*other, space);
        let alpha = s[3] + d[3] - s[3] * d[3];
        let mut output = [0.0, 0.0, 0.0, alpha];
        for i in 0..3 {
            let mixed = match mode {
                Mix::Normal => s[i],
                Mix::Multiply => s[i] * d[i],
                Mix::Screen => s[i] + d[i] - s[i] * d[i],
                Mix::Darken => s[i].min(d[i]),
                Mix::Lighten => s[i].max(d[i]),
                _ => s[i], // Admission restricts separable modes before execution.
            };
            output[i] = if alpha > 0.0 {
                ((1.0 - d[3]) * s[3] * s[i] + (1.0 - s[3]) * d[3] * d[i] + s[3] * d[3] * mixed)
                    / alpha
            } else {
                0.0
            };
        }
        *pixel = encode(output, space);
    }
    pixels.recompute_may_have_transparency();
}
