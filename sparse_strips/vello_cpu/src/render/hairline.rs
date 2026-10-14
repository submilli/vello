// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent axis-aligned hairline segments with Skia's 26.6 endpoint grid.

use super::RenderContext;
use crate::kurbo::{Affine, BezPath, Cap, PathEl, Point, Rect, Shape};
use crate::peniko::Fill;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

pub(super) fn draw(ctx: &mut RenderContext, path: &BezPath, transform: Affine) -> bool {
    let stroke = &ctx.state.stroke;
    if ctx.aliasing_threshold.is_some()
        || !stroke.dash_pattern.is_empty()
        || stroke.start_cap != Cap::Butt
        || stroke.end_cap != Cap::Butt
    {
        return false;
    }
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let fast_len = |x: f32, y: f32| x.abs().max(y.abs()) + 0.5 * x.abs().min(y.abs());
    let width = stroke.width as f32;
    let (sx, sy) = (
        fast_len(width * a as f32, width * b as f32),
        fast_len(width * c as f32, width * d as f32),
    );
    if !(sx > 0.0 && sy > 0.0 && sx <= 1.0 && sy <= 1.0) {
        return false;
    }
    let coverage = (sx + sy) * 0.5;
    let blend = ctx.stroke_source_blend.unwrap_or(ctx.state.blend_mode);
    if coverage != 1.0 && !RenderContext::supports_stroke_coverage_as_alpha(blend) {
        return false;
    }
    // Preflight the entire contour before emitting any commands.
    if !segments(path, transform, |_, _| {}) {
        return false;
    }
    // Skia scales the original 8-bit paint alpha before independently
    // compositing segments. Folding it into the fixed-point span coverage
    // retains the blitter's truncation; shader stops/texels keep their alpha.
    let supports_alpha = RenderContext::supports_stroke_coverage_as_alpha(blend);
    let (paint, coverage_alpha) = if supports_alpha {
        let source_alpha = match &ctx.state.paint {
            crate::PaintType::Solid(color) => color.components[3],
            _ => 1.0,
        };
        let original = ((source_alpha * ctx.paint_alpha).clamp(0.0, 1.0) * 255.0).round() as u32;
        let alpha = (original * (coverage * 256.0) as u32) >> 8;
        let multiplier = if original > 0 && source_alpha.is_finite() {
            1.0 / source_alpha
        } else {
            0.0
        };
        (ctx.encode_paint_alpha(multiplier), alpha)
    } else {
        (ctx.encode_current_paint(), 255)
    };
    segments(path, transform, |start, end| {
        coverage_spans(start, end, ctx.width, ctx.height, coverage_alpha, |rect| {
            ctx.dispatcher.fill_path(
                &rect.to_path(0.1),
                Fill::NonZero,
                Affine::IDENTITY,
                paint.clone(),
                blend,
                None,
                ctx.mask.clone(),
            );
        });
    });
    true
}

// SkScan_Antihair uses a 26.6 endpoint grid and complementary 8-bit
// cross-axis weights, with endpoint coverage truncated before alpha scaling.
const GRID: i64 = 64;
const FRACTION_MASK: i64 = GRID - 1;
const HALF_PIXEL: i64 = GRID / 2;
const MAX_ALPHA: u32 = 255;

fn coverage_spans(
    a: Point,
    b: Point,
    width: u16,
    height: u16,
    alpha: u32,
    mut emit: impl FnMut(Rect),
) {
    let horizontal = a.y == b.y;
    let (u0, u1, v) = if horizontal {
        (a.x.min(b.x), a.x.max(b.x), a.y)
    } else {
        (a.y.min(b.y), a.y.max(b.y), a.x)
    };
    let (u0, u1, v) = (
        (u0 * GRID as f64) as i64,
        (u1 * GRID as f64) as i64,
        (v * GRID as f64) as i64,
    );
    let shifted = v + HALF_PIXEL;
    let cross = ((shifted & FRACTION_MASK) * 4) as u32;
    let major_limit = if horizontal { width } else { height };
    let cross_limit = if horizontal { height } else { width };
    for (row, coverage) in [
        (shifted >> 6, cross),
        ((shifted >> 6) - 1, MAX_ALPHA - cross),
    ] {
        if row < 0 || row >= i64::from(cross_limit) {
            continue;
        }
        let start = (u0 >> 6).max(0);
        let end = ((u1 + FRACTION_MASK) >> 6).min(i64::from(major_limit));
        let first_end = (start + 1).min(end);
        let spans = [
            (
                start,
                first_end,
                (u1.min(first_end * GRID) - u0.max(start * GRID)).max(0) as u32,
            ),
            (first_end, (u1 >> 6).min(end), GRID as u32),
            ((u1 >> 6).max(first_end), end, (u1 & FRACTION_MASK) as u32),
        ];
        for (left, right, length) in spans {
            let coverage = ((coverage * length) >> 6) * alpha / MAX_ALPHA;
            if left >= right || coverage == 0 {
                continue;
            }
            // A fractional area inside each covered pixel encodes the
            // constant run coverage without an intermediate mask surface.
            let fraction = f64::from(coverage) / f64::from(MAX_ALPHA);
            let rect = if horizontal {
                Rect::new(left as f64, row as f64, right as f64, row as f64 + fraction)
            } else {
                Rect::new(row as f64, left as f64, row as f64 + fraction, right as f64)
            };
            emit(rect);
        }
    }
}

fn segments(path: &BezPath, transform: Affine, mut emit: impl FnMut(Point, Point)) -> bool {
    let mut current = Point::ORIGIN;
    let mut start = current;
    for el in path.elements() {
        let next = match *el {
            PathEl::MoveTo(p) => {
                start = transform * p;
                current = start;
                continue;
            }
            PathEl::LineTo(p) => transform * p,
            PathEl::ClosePath => start,
            _ => return false,
        };
        if !next.is_finite()
            || !current.is_finite()
            || [next.x, next.y, current.x, current.y]
                .iter()
                .any(|v| v.abs() > 32766.0)
            || (next.x != current.x && next.y != current.y)
        {
            return false;
        }
        if next != current {
            emit(current, next);
        }
        current = next;
    }
    true
}
