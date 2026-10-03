// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrices operate on unpremultiplied sRGB, with per-operation clamping.

use vello_common::pixmap::Pixmap;

pub(super) fn apply(pixmap: &mut Pixmap, matrix: &[f32; 20]) {
    for pixel in pixmap.data_mut() {
        let alpha = f32::from(pixel.a);
        let input = if pixel.a == 0 {
            [0.0; 4]
        } else {
            [
                f32::from(pixel.r) / alpha,
                f32::from(pixel.g) / alpha,
                f32::from(pixel.b) / alpha,
                alpha / 255.0,
            ]
        };
        let mut output = [0.0; 4];
        for (channel, row) in output.iter_mut().zip(matrix.as_chunks::<5>().0) {
            *channel = (input
                .iter()
                .zip(row)
                .map(|(v, factor)| v * factor)
                .sum::<f32>()
                + row[4])
                .clamp(0.0, 1.0);
        }
        let alpha = output[3] * 255.0;
        pixel.r = quantize(output[0] * alpha);
        pixel.g = quantize(output[1] * alpha);
        pixel.b = quantize(output[2] * alpha);
        pixel.a = quantize(alpha);
    }
    pixmap.recompute_may_have_transparency();
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Channels are clamped to 0..255 before rounding."
)]
fn quantize(channel: f32) -> u8 {
    (channel + 0.5) as u8
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
