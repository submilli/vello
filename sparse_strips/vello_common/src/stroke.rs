// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Device-space geometry for the axis-aligned rectangle stroke fast path.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, BezPath, Join, Rect, Shape, Stroke};

/// Build the quantized frame used for a solid miter-joined rectangle stroke.
/// Returns `None` when the stroke needs the general path rasterizer.
///
/// Skia's `AntiFrameRect` uses 24.8 edge positions and aligns both edges of
/// a subpixel band to the same pixel. Keeping this in device space preserves
/// distinct horizontal and vertical widths under non-uniform transforms.
pub fn rectangle_frame(rect: Rect, stroke: &Stroke, transform: Affine) -> Option<BezPath> {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    if !((b == 0.0 && c == 0.0) || (a == 0.0 && d == 0.0))
        || stroke.join != Join::Miter
        || stroke.miter_limit < core::f64::consts::SQRT_2
        || !stroke.dash_pattern.is_empty()
        || !stroke.width.is_finite()
        || stroke.width <= 0.0
        || rect.width() == 0.0
        || rect.height() == 0.0
    {
        return None;
    }
    let rect = transform.transform_rect_bbox(rect);
    let sx = ((stroke.width as f32 * a as f32) + (stroke.width as f32 * c as f32)).abs();
    let sy = ((stroke.width as f32 * b as f32) + (stroke.width as f32 * d as f32)).abs();
    let rx = sx * 0.5;
    let ry = sy * 0.5;
    // Refuse coordinates outside Skia's fixed-point domain before conversion.
    let fixed = |value: f32| {
        if value.is_finite() && value.abs() < 32767.0 {
            Some((value * 256.0) as i64)
        } else {
            None
        }
    };
    let mut outer = [
        fixed(rect.x0 as f32 - rx)?,
        fixed(rect.y0 as f32 - ry)?,
        fixed(rect.x1 as f32 + rx)?,
        fixed(rect.y1 as f32 + ry)?,
    ];
    let mut inner = [
        fixed(rect.x0 as f32 + (sx - rx))?,
        fixed(rect.y0 as f32 + (sy - ry))?,
        fixed(rect.x1 as f32 - (sx - rx))?,
        fixed(rect.y1 as f32 - (sy - ry))?,
    ];
    if sx < 1.0 || sy < 1.0 {
        for axis in 0..2 {
            align(&mut outer[axis], &mut inner[axis]);
            align(&mut inner[axis + 2], &mut outer[axis + 2]);
        }
    }
    let rect = |p: [i64; 4]| {
        Rect::new(
            p[0] as f64 / 256.0,
            p[1] as f64 / 256.0,
            p[2] as f64 / 256.0,
            p[3] as f64 / 256.0,
        )
    };
    let mut path = rect(outer).to_path(0.01);
    if inner[0] < inner[2] && inner[1] < inner[3] {
        path.extend(rect(inner).to_path(0.01).reverse_subpaths());
    }
    Some(path)
}

fn align(first: &mut i64, second: &mut i64) {
    if *first >> 8 == *second >> 8 {
        *second -= *first & 255;
        *first &= !255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonuniform_frame_aligns_thin_bands_before_rasterization() {
        let path = rectangle_frame(
            Rect::new(60.0, 30.0, 210.0, 105.0),
            &Stroke::new(23.717).with_join(Join::Miter),
            Affine::scale_non_uniform(8.0 / 300.0, 8.0 / 150.0),
        )
        .unwrap();
        let bounds = path.bounding_box();
        assert_eq!(bounds.x0, 1.0);
        assert_eq!(bounds.x1, 5.6328125);
        assert_eq!(path.winding(crate::kurbo::Point::new(1.25, 1.5)), 1);
        assert_eq!(path.winding(crate::kurbo::Point::new(3.0, 3.0)), 0);
    }

    #[test]
    fn frame_rejects_general_stroke_states_and_unrepresentable_coordinates() {
        let r = Rect::new(2.0, 2.0, 6.0, 5.0);
        let s = Stroke::new(1.0).with_join(Join::Miter);
        assert!(rectangle_frame(r, &s, Affine::rotate(0.3)).is_none());
        assert!(rectangle_frame(r, &s.clone().with_join(Join::Round), Affine::IDENTITY).is_none());
        assert!(
            rectangle_frame(r, &s.clone().with_dashes(0.0, [1.0, 1.0]), Affine::IDENTITY).is_none()
        );
        assert!(rectangle_frame(r, &s, Affine::scale(1e10)).is_none());
    }
}
