// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared premultiplied channel encoding and SVG working-space conversion.
//!
//! Chrome fuses color-space conversion into color-filter primitives (matrix and
//! component transfer), so those convert in float. Primitives that read or write
//! pixels see 8-bit premultiplied surfaces in their working space: the `stored`
//! conversions round through that storage.
use alloc::vec::Vec;
use vello_common::color::{AlphaColor, PremulRgba8, Srgb};
use vello_common::filter::graph::ColorSpace;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;
/// Straight components of an encoded sRGB pixel in `space`, converted in float.
pub(super) fn straight(pixel: PremulRgba8, space: ColorSpace) -> [f32; 4] {
    let mut channels = unpremultiplied(pixel);
    if space == ColorSpace::LinearRgb {
        for channel in &mut channels[..3] {
            *channel = linear(*channel);
        }
    }
    channels
}

/// Straight components of the 8-bit surface pixel stored in `space`.
pub(super) fn stored_straight(pixel: PremulRgba8, space: ColorSpace) -> [f32; 4] {
    unpremultiplied(to_space(pixel, space))
}

/// The 8-bit premultiplied pixel Chrome stores for an sRGB pixel in `space`.
pub(super) fn to_space(pixel: PremulRgba8, space: ColorSpace) -> PremulRgba8 {
    if space == ColorSpace::Srgb || pixel.a == 0 {
        return pixel;
    }
    let mut channels = unpremultiplied(pixel);
    for channel in &mut channels[..3] {
        *channel = linear(*channel);
    }
    premultiplied8(channels)
}

/// The encoded sRGB pixel for an 8-bit premultiplied pixel stored in `space`.
pub(super) fn from_space(pixel: PremulRgba8, space: ColorSpace) -> PremulRgba8 {
    if space == ColorSpace::Srgb || pixel.a == 0 {
        return pixel;
    }
    let mut channels = unpremultiplied(pixel);
    for channel in &mut channels[..3] {
        *channel = srgb(*channel);
    }
    premultiplied8(channels)
}

fn unpremultiplied(pixel: PremulRgba8) -> [f32; 4] {
    if pixel.a == 0 {
        return [0.0; 4];
    }
    let alpha = f32::from(pixel.a);
    [
        f32::from(pixel.r) / alpha,
        f32::from(pixel.g) / alpha,
        f32::from(pixel.b) / alpha,
        alpha / 255.0,
    ]
}

/// Straight components of a color in `space`.
pub(super) fn in_space(color: AlphaColor<Srgb>, space: ColorSpace) -> [f32; 4] {
    let mut channels = color.components;
    if space == ColorSpace::LinearRgb {
        for channel in &mut channels[..3] {
            *channel = linear(channel.clamp(0.0, 1.0));
        }
    }
    channels
}

/// Encode straight components in `space` as an sRGB pixel, converting in float.
pub(super) fn encode(mut channels: [f32; 4], space: ColorSpace) -> PremulRgba8 {
    if space == ColorSpace::LinearRgb {
        for channel in &mut channels[..3] {
            *channel = srgb(channel.clamp(0.0, 1.0));
        }
    }
    premultiplied8(channels)
}

/// Encode a result stored on an 8-bit surface in `space`, then converted to sRGB.
pub(super) fn encode_stored(channels: [f32; 4], space: ColorSpace) -> PremulRgba8 {
    from_space(premultiplied8(channels), space)
}

pub(super) fn premultiplied8(channels: [f32; 4]) -> PremulRgba8 {
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
/// Round to the nearest 8-bit level, ties to even, as Chrome's raster pipeline does.
pub(super) fn quantize(value: f32) -> u8 {
    let scaled = value.clamp(0.0, 1.0) * 255.0;
    let nearest = (scaled + 0.5) as u8;
    if f32::from(nearest) - scaled == 0.5 && nearest % 2 == 1 {
        nearest - 1
    } else {
        nearest
    }
}

/// Premultiplied components of the 8-bit surface stored in `space`.
pub(super) fn working(pixels: &Pixmap, space: ColorSpace) -> Vec<[f32; 4]> {
    pixels
        .data()
        .iter()
        .map(|pixel| {
            let stored = to_space(*pixel, space);
            [stored.r, stored.g, stored.b, stored.a].map(|c| f32::from(c) / 255.0)
        })
        .collect()
}

/// Store premultiplied working components on an 8-bit surface, then as sRGB pixels.
pub(super) fn store(pixels: &mut Pixmap, data: &[[f32; 4]], space: ColorSpace) {
    for (pixel, channels) in pixels.data_mut().iter_mut().zip(data) {
        *pixel = encode_stored(unpremultiply(*channels), space);
    }
    pixels.recompute_may_have_transparency();
}

pub(super) fn premultiply(mut channels: [f32; 4]) -> [f32; 4] {
    for i in 0..3 {
        channels[i] *= channels[3];
    }
    channels
}

/// Color is clamped to alpha first, so out-of-range premultiplied results stay valid.
pub(super) fn unpremultiply(mut channels: [f32; 4]) -> [f32; 4] {
    let alpha = channels[3].clamp(0.0, 1.0);
    for channel in &mut channels[..3] {
        *channel = if alpha > 0.0 {
            channel.clamp(0.0, alpha) / alpha
        } else {
            0.0
        };
    }
    channels[3] = alpha;
    channels
}
