// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared premultiplied channel encoding and SVG working-space conversion.
use vello_common::color::PremulRgba8;
use vello_common::filter::graph::ColorSpace;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
pub(super) fn straight(pixel: PremulRgba8, space: ColorSpace) -> [f32; 4] {
    let alpha = f32::from(pixel.a) / 255.0;
    if alpha == 0.0 {
        return [0.0; 4];
    }
    let mut channels = [
        f32::from(pixel.r) / f32::from(pixel.a),
        f32::from(pixel.g) / f32::from(pixel.a),
        f32::from(pixel.b) / f32::from(pixel.a),
        alpha,
    ];
    if space == ColorSpace::LinearRgb {
        for channel in &mut channels[..3] {
            *channel = linear(*channel);
        }
    }
    channels
}

pub(super) fn encode(mut channels: [f32; 4], space: ColorSpace) -> PremulRgba8 {
    if space == ColorSpace::LinearRgb {
        for channel in &mut channels[..3] {
            *channel = srgb(*channel);
        }
    }
    let alpha = channels[3].clamp(0.0, 1.0);
    PremulRgba8 {
        r: quantize(channels[0].clamp(0.0, 1.0) * alpha),
        g: quantize(channels[1].clamp(0.0, 1.0) * alpha),
        b: quantize(channels[2].clamp(0.0, 1.0) * alpha),
        a: quantize(alpha),
    }
}

fn linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Finite normalized channels are clamped before rounding."
)]
pub(super) fn quantize(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
