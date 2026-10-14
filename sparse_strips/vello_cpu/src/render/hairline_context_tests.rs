// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::RenderContext;
use crate::kurbo::{BezPath, Cap, Join, Rect, Stroke};
use crate::peniko::color::palette::css::{RED, YELLOW};
use crate::peniko::{BlendMode, Compose, Mix};
use crate::{Pixmap, Resources};

fn path() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((2.0, 2.0));
    path.line_to((6.0, 2.0));
    path.line_to((6.0, 6.0));
    path
}

fn pixels(mut scene: RenderContext) -> Pixmap {
    scene.flush();
    let mut pixels = Pixmap::new(8, 8);
    scene.render(pixels.as_mut(), &mut Resources::new());
    pixels
}

#[test]
fn hairline_segments_use_source_opacity_before_composition() {
    for (compose, alpha, green, opacity) in [
        (Compose::SrcOver, 0.5, 195_u8, 255_u8),
        (Compose::Xor, 1.0, 230, 160),
        (Compose::Xor, 0.5, 250, 199),
    ] {
        let mut scene = RenderContext::new(8, 8);
        scene.set_paint(YELLOW);
        scene.fill_rect(&Rect::new(0.0, 0.0, 8.0, 8.0));
        scene.set_paint(RED);
        scene.set_paint_alpha(alpha);
        scene.set_blend_mode(BlendMode::new(Mix::Normal, compose));
        scene.set_stroke(Stroke::new(0.5).with_join(Join::Miter).with_caps(Cap::Butt));
        scene.stroke_path(&path());
        let image = pixels(scene);
        let observed = image.data()[2 * 8 + 5];
        assert!(observed.a.abs_diff(opacity) <= 1);
        // Pixmaps retain premultiplied channels.
        let expected_green = ((u32::from(green) * u32::from(opacity) + 127) / 255) as u8;
        assert!(
            observed.g.abs_diff(expected_green) <= 1,
            "{compose:?} alpha={alpha}: {observed:?} expected_green={expected_green}"
        );
    }
}

#[test]
fn isolated_copy_source_does_not_convert_a_thin_stroke_to_hairlines() {
    let mut scene = RenderContext::new(8, 8);
    scene.set_paint(RED);
    scene.set_stroke(Stroke::new(0.5).with_join(Join::Miter).with_caps(Cap::Butt));
    scene.set_stroke_source_blend(Some(BlendMode::new(Mix::Normal, Compose::Copy)));
    scene.stroke_path(&path());
    let image = pixels(scene);
    assert!(image.data()[8 + 6].a.abs_diff(16) <= 1);
    assert!(image.data()[2 * 8 + 5].a.abs_diff(112) <= 1);
}

#[test]
fn paint_opacity_is_bounded_and_reset_with_the_context() {
    for (alpha, expected) in [(f32::NAN, 255), (f32::INFINITY, 255), (-1.0, 0), (2.0, 255)] {
        let mut scene = RenderContext::new(8, 8);
        scene.set_paint(RED);
        scene.set_paint_alpha(alpha);
        scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(pixels(scene).data()[0].a, expected);
    }
    let mut scene = RenderContext::new(8, 8);
    scene.set_paint_alpha(0.0);
    scene.reset();
    scene.set_paint(RED);
    scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
    assert_eq!(pixels(scene).data()[0].a, 255);
}

#[test]
fn unit_hairline_keeps_effective_source_composition_inside_a_layer() {
    let mut scene = RenderContext::new(8, 8);
    scene.set_paint(RED);
    scene.set_paint_alpha(0.5);
    scene.set_stroke(Stroke::new(1.0).with_join(Join::Miter).with_caps(Cap::Butt));
    scene.push_blend_layer(BlendMode::new(Mix::Normal, Compose::Copy));
    scene.set_stroke_source_blend(Some(BlendMode::new(Mix::Normal, Compose::Copy)));
    scene.stroke_path(&path());
    scene.pop_layer();
    let image = pixels(scene);
    assert!(image.data()[2 * 8 + 5].a.abs_diff(96) <= 1);
}
