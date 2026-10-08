// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Linear-light blur and shadow primitive implementations.
use super::channels::{encode, straight};
use super::{context::ScratchBuffer, filter_lowp};
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::{CompositeOperator, Filter, FilterPrimitive};
use vello_common::kurbo::Affine;
use vello_common::pixmap::Pixmap;

pub(super) fn blur(pixels: &mut Pixmap, sigma: f32, edge: vello_common::filter_effects::EdgeMode) {
    if sigma == 0.0 {
        return;
    }
    let mut data = pixels
        .data()
        .iter()
        .map(|pixel| {
            let mut channels = straight(*pixel, ColorSpace::LinearRgb);
            for i in 0..3 {
                channels[i] *= channels[3];
            }
            channels
        })
        .collect();
    let plan = vello_common::filter::gaussian_blur::GaussianBlur::new(sigma, edge);
    super::float_blur::blur(
        &mut data,
        usize::from(pixels.width()),
        usize::from(pixels.height()),
        &plan,
    );
    for (pixel, mut channels) in pixels.data_mut().iter_mut().zip(data) {
        for i in 0..3 {
            channels[i] = if channels[3] > 0.0 {
                channels[i] / channels[3]
            } else {
                0.0
            };
        }
        *pixel = encode(channels, ColorSpace::LinearRgb);
    }
    pixels.recompute_may_have_transparency();
}

pub(super) fn shadow(
    output: &mut Pixmap,
    primitive: &FilterPrimitive,
    scratch: &mut ScratchBuffer,
) {
    let (FilterPrimitive::DropShadow {
        dx,
        dy,
        std_deviation,
        color,
        edge_mode,
    }
    | FilterPrimitive::DropShadowOnly {
        dx,
        dy,
        std_deviation,
        color,
        edge_mode,
    }) = primitive
    else {
        return;
    };
    let source = output.clone();
    for pixel in output.data_mut() {
        let alpha = f32::from(pixel.a) / 255.0;
        let mut tinted = color.components;
        tinted[3] *= alpha;
        *pixel = encode(tinted, ColorSpace::Srgb);
    }
    blur(output, *std_deviation, *edge_mode);
    filter_lowp(
        &Filter::from_primitive(FilterPrimitive::Offset { dx: *dx, dy: *dy }),
        output,
        scratch,
        Affine::IDENTITY,
    );
    if matches!(primitive, FilterPrimitive::DropShadow { .. }) {
        let mut foreground = source;
        super::graph::composite(
            &mut foreground,
            output,
            CompositeOperator::Over,
            ColorSpace::LinearRgb,
        );
        *output = foreground;
    }
}
