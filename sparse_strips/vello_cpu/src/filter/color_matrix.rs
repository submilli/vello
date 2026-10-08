// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrices operate on unpremultiplied sRGB, with per-operation clamping.

use vello_common::pixmap::Pixmap;

use super::channels::{encode, straight};
use vello_common::filter::graph::ColorSpace;

pub(super) fn apply(pixels: &mut Pixmap, matrix: &[f32; 20]) {
    apply_in_space(pixels, matrix, ColorSpace::Srgb);
}

pub(super) fn apply_in_space(pixels: &mut Pixmap, matrix: &[f32; 20], space: ColorSpace) {
    for p in pixels.data_mut() {
        let input = straight(*p, space);
        let mut output = [0.0; 4];
        for (value, row) in output.iter_mut().zip(matrix.as_chunks::<5>().0) {
            *value =
                (input.iter().zip(row).map(|(v, k)| v * k).sum::<f32>() + row[4]).clamp(0.0, 1.0);
        }
        *p = encode(output, space);
    }
    pixels.recompute_may_have_transparency();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{context::ScratchBuffer, filter_lowp};
    use vello_common::filter_effects::{Filter, FilterFunction};
    use vello_common::kurbo::Affine;

    #[test]
    fn css_chain_clamps_each_step_and_preserves_premultiplied_alpha() {
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0].r = 64;
        pixels.data_mut()[0].g = 32;
        pixels.data_mut()[0].a = 128;
        let chain = Filter::from_chain(
            [
                FilterFunction::Brightness { amount: 4.0 },
                FilterFunction::Invert { amount: 1.0 },
                FilterFunction::Opacity { amount: 0.5 },
            ]
            .into_iter()
            .map(|function| Filter::from_function(function).graph.primitives[0].clone()),
        )
        .unwrap();
        filter_lowp(
            &chain,
            &mut pixels,
            &mut ScratchBuffer::default(),
            Affine::IDENTITY,
        );
        let pixel = pixels.data()[0];
        assert_eq!((pixel.r, pixel.g, pixel.b, pixel.a), (0, 0, 64, 64));
        assert!(pixels.may_have_transparency());
    }
}
