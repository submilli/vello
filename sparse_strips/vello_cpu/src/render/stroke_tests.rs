// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Browser-observed stroke pixels, independent of outline implementation.

use super::RenderContext;
use crate::Resources;
use crate::kurbo::{Affine, Join, Rect, Shape, Stroke};
use crate::peniko::color::palette::css::RED;
use vello_common::pixmap::Pixmap;

fn pixels(mut scene: RenderContext) -> Pixmap {
    scene.flush();
    let mut pixels = Pixmap::new(16, 16);
    scene.render(pixels.as_mut(), &mut Resources::new());
    pixels
}

#[test]
fn rectangle_miter_inner_corner_retains_partial_coverage() {
    let mut scene = RenderContext::new(16, 16);
    scene.set_paint(RED);
    scene.set_stroke(Stroke::new(1.0).with_join(Join::Miter));
    scene.stroke_rect(&Rect::new(2.0, 2.0, 6.0, 5.0));
    let image = pixels(scene);
    assert!(image.data()[2 * 16 + 2].a.abs_diff(191) <= 1);
    assert_eq!(image.data()[3 * 16 + 3].a, 0);
}

#[test]
fn nonuniform_rect_and_path_take_their_distinct_browser_paths() {
    for (primitive, expected) in [(true, 255), (false, 184)] {
        let mut scene = RenderContext::new(16, 16);
        scene.set_paint(RED);
        scene.set_transform(Affine::scale_non_uniform(8.0 / 300.0, 8.0 / 150.0));
        scene.set_stroke(Stroke::new(23.717).with_join(Join::Miter));
        let rect = Rect::new(60.0, 30.0, 210.0, 105.0);
        if primitive {
            scene.stroke_rect(&rect);
        } else {
            scene.stroke_path(&rect.to_path(0.1));
        }
        let image = pixels(scene);
        assert!(
            image.data()[16 + 1].a.abs_diff(expected) <= 1,
            "primitive={primitive}"
        );
    }
}

#[test]
fn aliased_rectangle_keeps_unsnapped_geometry() {
    let mut scene = RenderContext::new(16, 16);
    scene.set_paint(RED);
    scene.set_stroke(Stroke::new(0.49).with_join(Join::Miter));
    scene.set_aliasing_threshold(Some(128));
    scene.stroke_path(&Rect::new(2.0, 2.5, 6.0, 5.5).to_path(0.1));
    assert_eq!(pixels(scene).data()[2 * 16 + 3].a, 0);
}

#[test]
fn open_miter_inner_corner_has_union_coverage() {
    let mut path = crate::kurbo::BezPath::new();
    path.move_to((2.0, 5.0));
    path.line_to((2.0, 2.0));
    path.line_to((6.0, 2.0));
    for transform in [
        Affine::IDENTITY,
        Affine::new([-1.0, 0.0, 0.0, 1.0, 8.0, 0.0]),
    ] {
        let mut scene = RenderContext::new(16, 16);
        scene.set_paint(RED);
        scene.set_transform(transform);
        scene.set_stroke(Stroke::new(1.5).with_join(Join::Miter));
        scene.stroke_path(&path);
        let x = if transform == Affine::IDENTITY { 2 } else { 5 };
        // The two arms cover .75 each; their .5625 overlap is counted once.
        assert_eq!(pixels(scene).data()[2 * 16 + x].a, 239);
    }
}

#[test]
fn fractional_horizontal_stroke_keeps_coverage_across_sparse_tile_gaps() {
    let mut path = crate::kurbo::BezPath::new();
    path.move_to((2.0, 3.0));
    path.line_to((14.0, 3.0));
    let mut scene = RenderContext::new(16, 16);
    scene.set_paint(RED);
    scene.set_stroke(Stroke::new(1.5));
    scene.stroke_path(&path);
    let image = pixels(scene);
    for y in [2, 3] {
        for x in 2..14 {
            assert_eq!(image.data()[y * 16 + x].a, 191, "pixel=({x},{y})");
        }
    }
}
