// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG graph blur and shadow primitives on 8-bit working surfaces, as Chrome builds them.
use super::channels::{color8, from_space, premultiplied8, to_space};
use super::{context::ScratchBuffer, filter_lowp};
use alloc::vec::Vec;
use vello_common::color::{AlphaColor, PremulRgba8, Srgb};
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::{CompositeOperator, Filter, FilterPrimitive};
use vello_common::kurbo::Affine;
use vello_common::pixmap::Pixmap;

/// A shadow's geometry and paint; the blur deviations are per axis.
pub(super) struct Shadow {
    pub dx: f32,
    pub dy: f32,
    pub std_deviation: [f32; 2],
    pub color: AlphaColor<Srgb>,
    /// Composite the input over its shadow (`feDropShadow`) or keep only the shadow.
    pub with_source: bool,
}

/// Blur in `space`, storing the working surface at 8 bits like Chrome.
pub(super) fn blur(pixels: &mut Pixmap, std_deviation: [f32; 2], space: ColorSpace) {
    let width = usize::from(pixels.width());
    let mut data: Vec<PremulRgba8> = pixels.data().iter().map(|p| to_space(*p, space)).collect();
    super::svg_gaussian::blur(&mut data, width, std_deviation);
    for (pixel, value) in pixels.data_mut().iter_mut().zip(data) {
        *pixel = from_space(value, space);
    }
    pixels.recompute_may_have_transparency();
}

/// Blur the input, paint the shadow color through its alpha, offset it and,
/// for `feDropShadow`, draw the input over it.
pub(super) fn shadow(
    output: &mut Pixmap,
    shadow: &Shadow,
    space: ColorSpace,
    scratch: &mut ScratchBuffer,
) {
    let source = output.clone();
    blur(output, shadow.std_deviation, space);
    // The shadow color fills the blurred alpha (Skia's SrcIn color filter).
    let color = color8(shadow.color, space);
    for pixel in output.data_mut() {
        let alpha = f32::from(to_space(*pixel, space).a) / 255.0;
        let painted = [color[0], color[1], color[2], color[3] * alpha];
        *pixel = from_space(premultiplied8(painted), space);
    }
    filter_lowp(
        &Filter::from_primitive(FilterPrimitive::Offset {
            dx: shadow.dx,
            dy: shadow.dy,
        }),
        output,
        scratch,
        Affine::IDENTITY,
    );
    if shadow.with_source {
        let mut foreground = source;
        super::svg_composite::composite(&mut foreground, output, CompositeOperator::Over, space);
        *output = foreground;
    }
}
