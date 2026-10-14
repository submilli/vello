// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use crate::kurbo::{Join, PathEl};

fn winding(lines: &[Line], p: (f64, f64)) -> i32 {
    lines
        .iter()
        .copied()
        .filter_map(|line| Edge::from_line(line).unwrap())
        .filter(|edge| edge.top.1 <= p.1 && p.1 < edge.bottom.1 && edge.x(p.1) < p.0)
        .map(|edge| edge.winding)
        .sum()
}

fn outline(points: &[(f64, f64)], transform: Affine) -> (Vec<Line>, Vec<Line>) {
    let mut path = BezPath::new();
    path.move_to(points[0]);
    for point in &points[1..] {
        path.line_to(*point);
    }
    let style = Stroke::new(1.5).with_join(Join::Miter);
    let mut ctx = StrokeCtx::default();
    flatten::expand_stroke(&path, &style, flatten::TOL, &mut ctx);
    let mut original = Vec::new();
    flatten::fill(
        Level::new(),
        ctx.output(),
        transform,
        &mut original,
        &mut FlattenCtx::default(),
        RectU16::new(0, 0, 16, 16),
    );
    let mut union = original.clone();
    normalize(&mut union, &mut Cost::default()).unwrap();
    (original, union)
}

#[test]
fn miter_inner_overlap_is_covered_once() {
    let (original, union) = outline(&[(2.0, 5.0), (2.0, 2.0), (6.0, 2.0)], Affine::IDENTITY);
    assert_eq!(winding(&original, (2.25, 2.25)).abs(), 2);
    assert_eq!(winding(&union, (2.25, 2.25)), 1);
    assert_eq!(winding(&union, (2.875, 2.875)), 0);
}

#[test]
fn short_edges_crossings_and_reflection_keep_the_nonzero_region() {
    for points in [
        &[(2.0, 5.0), (2.0, 2.0), (2.25, 2.0), (2.25, 5.0)][..],
        &[(2.0, 2.0), (6.0, 6.0), (2.0, 6.0), (6.0, 2.0)][..],
        &[(2.0, 2.0), (6.0, 2.0), (2.0, 2.0), (2.0, 5.0)][..],
    ] {
        for transform in [
            Affine::IDENTITY,
            Affine::new([-1.0, 0.0, 0.0, 1.0, 10.0, 0.0]),
        ] {
            let (original, union) = outline(points, transform);
            for y in 0..64 {
                for x in 0..80 {
                    // Avoid placing every probe on horizontal/vertical boundaries.
                    let p = ((f64::from(x) + 0.31) / 8.0, (f64::from(y) + 0.27) / 8.0);
                    let result = winding(&union, p);
                    assert!(result == 0 || result == 1);
                    assert_eq!(result != 0, winding(&original, p) != 0, "point={p:?}");
                }
            }
        }
    }
}

#[test]
fn coincident_opposite_contours_cancel_before_area_integration() {
    let path: BezPath = [
        PathEl::MoveTo((1.0, 1.0).into()),
        PathEl::LineTo((4.0, 1.0).into()),
        PathEl::LineTo((4.0, 4.0).into()),
        PathEl::LineTo((1.0, 4.0).into()),
        PathEl::ClosePath,
    ]
    .into_iter()
    .collect();
    let mut duplicate = path.clone();
    duplicate.extend(path.reverse_subpaths());
    let mut lines = Vec::new();
    flatten::fill(
        Level::new(),
        &duplicate,
        Affine::IDENTITY,
        &mut lines,
        &mut FlattenCtx::default(),
        RectU16::new(0, 0, 16, 16),
    );
    normalize(&mut lines, &mut Cost::default()).unwrap();
    assert!(lines.is_empty());
}

#[test]
fn pair_work_is_refused_before_intersection_enumeration() {
    let mut lines = Vec::new();
    for x in 0..1500 {
        lines.push(Line::new(
            Point::new(x as f32, 0.0),
            Point::new(x as f32, 1.0),
        ));
    }
    assert!(matches!(
        normalize(&mut lines, &mut Cost::default()),
        Err(GeometryLimit)
    ));
    assert_eq!(lines.len(), 1500);
}

#[test]
fn union_cost_is_cumulative_and_overflow_safe() {
    let mut cost = Cost {
        pairs: MAX_PAIRS,
        ..Cost::default()
    };
    assert!(
        cost.add(Cost {
            pairs: 1,
            ..Cost::default()
        })
        .is_err()
    );
    let mut cost = Cost {
        sweep: usize::MAX,
        ..Cost::default()
    };
    assert!(
        cost.add(Cost {
            sweep: 1,
            ..Cost::default()
        })
        .is_err()
    );
}

#[test]
fn refusal_is_reported_before_paint_and_raw_render_emits_no_partial_stroke() {
    let mut path = BezPath::new();
    path.move_to((2.0, 2.0));
    for index in 0..900 {
        path.line_to((2.0 + f64::from(index % 2), 2.0 + f64::from(index) / 1000.0));
    }
    let style = Stroke::new(1.5);
    assert!(
        StrokeBudget::default()
            .charge(&path, &style, Affine::IDENTITY)
            .is_err()
    );
    let mut lines = Vec::new();
    flatten::stroke(
        Level::new(),
        &path,
        &style,
        Affine::IDENTITY,
        &mut lines,
        &mut FlattenCtx::default(),
        &mut StrokeCtx::default(),
        RectU16::new(0, 0, 16, 16),
    );
    assert!(lines.is_empty());
}

#[test]
fn preparation_is_independent_of_the_later_clip_bounds() {
    let mut path = BezPath::new();
    path.move_to((-2.0, 8.0));
    path.line_to((-2.0, -2.0));
    path.line_to((10.0, -2.0));
    let mut outputs = [Vec::new(), Vec::new()];
    for (output, clip) in outputs
        .iter_mut()
        .zip([RectU16::new(0, 0, 16, 16), RectU16::new(2, 2, 6, 6)])
    {
        flatten::stroke(
            Level::new(),
            &path,
            &Stroke::new(6.0),
            Affine::IDENTITY,
            output,
            &mut FlattenCtx::default(),
            &mut StrokeCtx::default(),
            clip,
        );
    }
    assert!(!outputs[0].is_empty());
    assert_eq!(outputs[0].len(), outputs[1].len());
    for (a, b) in outputs[0].iter().zip(&outputs[1]) {
        assert_eq!(a.p0, b.p0);
        assert_eq!(a.p1, b.p1);
    }
}

#[test]
fn internal_slabs_do_not_multiply_horizontal_tiling_work() {
    let mut path = BezPath::new();
    for index in 0..16 {
        let y = 2.0 + f64::from(index) * 0.5;
        path.move_to((2.0, y));
        path.line_to((14.0, y));
        path.line_to((14.0, y + 4.0));
        path.line_to((2.0, y + 4.0));
        path.close_path();
    }
    let mut lines = Vec::new();
    flatten::fill(
        Level::new(),
        &path,
        Affine::IDENTITY,
        &mut lines,
        &mut FlattenCtx::default(),
        RectU16::new(0, 0, 16, 16),
    );
    normalize(&mut lines, &mut Cost::default()).unwrap();
    let horizontal: Vec<_> = lines.iter().filter(|line| line.p0.y == line.p1.y).collect();
    assert_eq!(horizontal.len(), 2);
    let length: f32 = horizontal
        .iter()
        .map(|line| (line.p1.x - line.p0.x).abs())
        .sum();
    assert_eq!(length, 24.0);
}
