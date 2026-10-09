// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feDisplacementMap`.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feDisplacementMapElement>
use super::bounds::PixelBounds;
use super::channels::{TRANSPARENT, from_space, straight, to_space};
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::{ColorChannel, EdgeMode};
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

/// Displace `pixels` (the `in` image) by unpremultiplied channels of `map`, both in
/// the primitive's color space. Like Chrome, the displaced image is read from its
/// 8-bit working surface. Samples round to the nearest pixel, and samples outside
/// the input's `bounds` are transparent.
pub(super) fn apply(
    pixels: &mut Pixmap,
    map: &Pixmap,
    scale: f32,
    channels: [ColorChannel; 2],
    bounds: PixelBounds,
    space: ColorSpace,
) {
    let width = usize::from(pixels.width());
    if width == 0 {
        return;
    }
    let source: Vec<PremulRgba8> = pixels.data().iter().map(|p| to_space(*p, space)).collect();
    for (index, (pixel, displacement)) in pixels.data_mut().iter_mut().zip(map.data()).enumerate() {
        let map = straight(*displacement, space);
        let offset = |channel| f64::from(scale) * (f64::from(map[component(channel)]) - 0.5);
        let x = (index % width) as f64 + offset(channels[0]);
        let y = (index / width) as f64 + offset(channels[1]);
        *pixel = from_space(sample(&source, width, bounds, x, y), space);
    }
    pixels.recompute_may_have_transparency();
}

fn component(channel: ColorChannel) -> usize {
    match channel {
        ColorChannel::Red => 0,
        ColorChannel::Green => 1,
        ColorChannel::Blue => 2,
        ColorChannel::Alpha => 3,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Coordinates are finite and saturate to values outside the bounds."
)]
fn sample(
    source: &[PremulRgba8],
    width: usize,
    bounds: PixelBounds,
    x: f64,
    y: f64,
) -> PremulRgba8 {
    let x = bounds.extend_x((x + 0.5).floor() as i64, EdgeMode::None);
    let y = bounds.extend_y((y + 0.5).floor() as i64, EdgeMode::None);
    match (x, y) {
        (Some(x), Some(y)) => source[y * width + x],
        _ => TRANSPARENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_maps_keep_pixels_and_full_channels_shift_by_half_the_scale() {
        let mut pixels = Pixmap::new(4, 1);
        pixels.data_mut()[1] = PremulRgba8 {
            r: 9,
            g: 0,
            b: 0,
            a: 255,
        };
        let mut map = Pixmap::new(4, 1);
        let gray = PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };
        map.data_mut().fill(gray);
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 1,
        };
        let channels = [ColorChannel::Red, ColorChannel::Green];
        let mut neutral = pixels.clone();
        apply(&mut neutral, &map, 2.0, channels, bounds, ColorSpace::Srgb);
        assert_eq!(neutral.data()[1].r, 9);
        map.data_mut().fill(PremulRgba8 {
            r: 0,
            g: 128,
            b: 0,
            a: 255,
        });
        apply(&mut pixels, &map, 4.0, channels, bounds, ColorSpace::Srgb);
        // Each output reads two pixels to its left; the first ones read outside.
        assert_eq!(
            pixels.data().iter().map(|p| p.r).collect::<Vec<_>>(),
            [0, 0, 0, 9]
        );
    }
}
