// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Closed rectangular stroke contours without overlapping inner joins.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{BezPath, Join, PathEl, Point, Rect, RoundedRect, Shape, Stroke};

/// Recognize a single closed axis-aligned rectangle, including explicit closure.
/// This bounded lookahead does not consume arbitrary page paths.
pub(crate) fn rectangle(elements: &[PathEl]) -> Option<Rect> {
    let elements = match elements.last() {
        Some(PathEl::MoveTo(_)) if elements.len() > 1 => &elements[..elements.len() - 1],
        _ => elements,
    };
    let [
        PathEl::MoveTo(a),
        PathEl::LineTo(b),
        PathEl::LineTo(c),
        PathEl::LineTo(d),
        rest @ ..,
    ] = elements
    else {
        return None;
    };
    if !matches!(rest, [PathEl::ClosePath])
        && !matches!(rest, [PathEl::LineTo(p), PathEl::ClosePath] if p == a)
    {
        return None;
    }
    let horizontal = a.y == b.y && b.x == c.x && c.y == d.y && d.x == a.x;
    let vertical = a.x == b.x && b.y == c.y && c.x == d.x && d.y == a.y;
    if !(horizontal || vertical) || ![a, b, c, d].iter().all(|p| p.is_finite()) {
        return None;
    }
    let rect = Rect::from_points(*a, *c);
    (rect.width() > 0.0 && rect.height() > 0.0).then_some(rect)
}

/// Skia's rectangular stroker uses a simple outer contour and a reversed inner
/// rectangle. General offset paths may overlap at inner joins; integrating their
/// winding before clamping would overcover those corners.
pub(crate) fn outline(rect: Rect, style: &Stroke, tolerance: f64) -> Option<BezPath> {
    if !style.dash_pattern.is_empty() || !style.width.is_finite() || style.width <= 0.0 {
        return None;
    }
    let radius = style.width * 0.5;
    let outer = rect.inflate(radius, radius);
    if !outer.is_finite() {
        return None;
    }
    let join = if style.join == Join::Miter && style.miter_limit < core::f64::consts::SQRT_2 {
        Join::Bevel
    } else {
        style.join
    };
    let mut path = match join {
        Join::Miter => outer.to_path(tolerance),
        Join::Round => {
            // Keep arc work bounded under a large user-space radius and tiny
            // transform, just as the general stroker's normalized round joins do.
            RoundedRect::from_rect(outer, radius).to_path(tolerance.min(0.001).max(radius * 1e-3))
        }
        Join::Bevel => {
            let mut p = BezPath::new();
            p.move_to((rect.x0, outer.y0));
            for point in [
                Point::new(rect.x1, outer.y0),
                Point::new(outer.x1, rect.y0),
                Point::new(outer.x1, rect.y1),
                Point::new(rect.x1, outer.y1),
                Point::new(rect.x0, outer.y1),
                Point::new(outer.x0, rect.y1),
                Point::new(outer.x0, rect.y0),
            ] {
                p.line_to(point);
            }
            p.close_path();
            p
        }
    };
    if style.width < rect.width().min(rect.height()) {
        let inner = rect
            .inflate(-radius, -radius)
            .to_path(tolerance)
            .reverse_subpaths();
        path.extend(inner);
    }
    Some(path)
}

/// Skia analytic line edges start on a 1/256 X grid and a quarter-pixel Y
/// grid. Apply this only to admitted axis-aligned, antialiased straight
/// rectangle contours; aliased paths retain their original geometry.
pub(crate) fn snap_edges(lines: &mut [crate::flatten::Line]) {
    let snap = |p: &mut crate::flatten::Point| {
        p.x = (p.x * 256.0).trunc() / 256.0;
        p.y = ((p.y * 256.0).trunc() / 64.0 + 0.5).floor() / 4.0;
    };
    for line in lines {
        snap(&mut line.p0);
        snap(&mut line.p1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::{Affine, Shape};

    #[test]
    fn rectangle_recognition_preserves_shape_and_closure() {
        let r = Rect::new(2.0, 2.0, 6.0, 5.0);
        let path = r.to_path(0.01);
        assert_eq!(rectangle(path.elements()), Some(r));
        assert_eq!(rectangle(path.reverse_subpaths().elements()), Some(r));
        assert_eq!(rectangle(&path.elements()[..4]), None);
        assert_eq!(rectangle((Affine::skew(0.2, 0.0) * path).elements()), None);
    }

    #[test]
    fn shrinking_a_huge_round_rectangle_bounds_arc_expansion() {
        let r = Rect::new(0.0, 0.0, 1e30, 1e30);
        let path = outline(r, &Stroke::new(1e29).with_join(Join::Round), 0.001).unwrap();
        assert!(path.elements().len() < 64);
    }

    #[test]
    fn rectangle_inner_contour_has_no_overlap_winding() {
        let r = Rect::new(2.0, 2.0, 6.0, 5.0);
        for join in [Join::Miter, Join::Round, Join::Bevel] {
            let p = outline(r, &Stroke::new(1.0).with_join(join), 0.001).unwrap();
            assert_eq!(p.winding(Point::new(2.25, 2.25)).abs(), 1);
            assert_eq!(p.winding(Point::new(2.75, 2.75)), 0);
            assert_eq!(p.winding(Point::new(3.0, 3.0)), 0);
        }
        let p = outline(r, &Stroke::new(4.0), 0.001).unwrap();
        assert_eq!(p.winding(Point::new(4.0, 3.0)).abs(), 1);
    }
}
