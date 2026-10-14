// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image paint opacity must compose with both tint modes without a trap.

use super::RenderContext;
use crate::kurbo::Rect;
use crate::peniko::{Color, ImageBrush};
use crate::{PaintType, Pixmap, Resources};
use std::sync::Arc;
use vello_common::color::PremulRgba8;
use vello_common::paint::{ImageSource, Tint, TintMode};

#[test]
fn image_opacity_multiplies_source_and_existing_tints() {
    for mode in [None, Some(TintMode::Multiply), Some(TintMode::AlphaMask)] {
        for alpha in [0.0, 0.5, 1.0] {
            let mut pixels = Pixmap::new(1, 1);
            pixels.data_mut()[0] = PremulRgba8 {
                r: 128,
                g: 64,
                b: 32,
                a: 255,
            };
            let mut scene = RenderContext::new(1, 1);
            let image = ImageBrush {
                image: ImageSource::Pixmap(Arc::new(pixels)),
                sampler: crate::peniko::ImageSampler::default().with_alpha(alpha),
            };
            scene.set_paint(PaintType::Image(image));
            scene.set_tint(mode.map(|mode| Tint {
                color: Color::from_rgba8(255, 128, 64, 128),
                mode,
            }));
            scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
            let mut result = Pixmap::new(1, 1);
            scene.render(result.as_mut(), &mut Resources::new());
            let p = result.data()[0];
            let expected = match mode {
                None => [128.0, 64.0, 32.0, 255.0],
                Some(TintMode::Multiply) => [64.0, 16.0, 4.0, 128.0],
                Some(TintMode::AlphaMask) => [128.0, 64.0, 32.0, 128.0],
            };
            for (actual, expected) in [p.r, p.g, p.b, p.a].into_iter().zip(expected) {
                assert!(
                    (f32::from(actual) - expected * alpha).abs() <= 1.0,
                    "mode={mode:?}, alpha={alpha}"
                );
            }
        }
    }
}
