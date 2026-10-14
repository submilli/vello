// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Preflight work accounting before the renderer expands strokes.
//!
//! Input work is bounded before expansion; the union pass then admits its actual
//! flattened edge count before intersection tests and slab processing. The
//! pinned Kurbo offset algorithm bounds each cubic side at 2^9 leaves: error
//! refinement stops at depth 8; a cusp split gives same-sign endpoints to each
//! child, preventing another cusp split without intervening error refinement.
//! Charging 4096 outline elements per curved piece covers both sides, caps,
//! joins and cusp handling. Linear pieces need at most 32. Dash cuts multiply
//! those costs before any outline allocation occurs.

use super::GeometryLimit;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, BezPath, PathEl, Point, Stroke};

const MAX_OUTLINE_ELEMENTS: f64 = 1_048_576.0;
const MAX_DEVICE_LENGTH: f64 = 262_144.0;

/// Cumulative admission before renderer geometry expansion.
#[derive(Default, Debug)]
pub struct StrokeBudget {
    elements: f64,
    device_length: f64,
    union: crate::stroke_union::Cost,
}

impl StrokeBudget {
    /// Charge the complete transformed path before flattening or stroking.
    pub fn charge(
        &mut self,
        path: &BezPath,
        style: &Stroke,
        transform: Affine,
    ) -> Result<(), GeometryLimit> {
        self.charge_expansion(path, style, transform)?;
        crate::stroke_union::admit(path, style, transform, &mut self.union)
    }

    pub(crate) fn charge_expansion(
        &mut self,
        path: &BezPath,
        style: &Stroke,
        transform: Affine,
    ) -> Result<(), GeometryLimit> {
        super::raster::RasterBudget::default().charge(path, transform)?;
        let scale = transform.as_coeffs()[..4]
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        if !scale.is_finite()
            || !style.width.is_finite()
            || !style.miter_limit.is_finite()
            || style.width * style.miter_limit.max(1.0) * scale > MAX_DEVICE_LENGTH
        {
            return Err(GeometryLimit);
        }
        let dash_min = style
            .dash_pattern
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .reduce(f64::min);
        let mut current = Point::ORIGIN;
        let mut start = current;
        for element in path.elements() {
            let (points, curved): (&[Point], bool) = match element {
                PathEl::MoveTo(point) => {
                    current = *point;
                    start = *point;
                    self.elements += 32.0 + style.dash_pattern.len() as f64;
                    continue;
                }
                PathEl::LineTo(point) => (core::slice::from_ref(point), false),
                PathEl::QuadTo(a, b) => {
                    self.charge_segment(&[current, *a, *b], true, style, transform, dash_min)?;
                    current = *b;
                    continue;
                }
                PathEl::CurveTo(a, b, c) => {
                    self.charge_segment(&[current, *a, *b, *c], true, style, transform, dash_min)?;
                    current = *c;
                    continue;
                }
                PathEl::ClosePath => (core::slice::from_ref(&start), false),
            };
            for point in points {
                self.charge_segment(&[current, *point], curved, style, transform, dash_min)?;
                current = *point;
            }
        }
        self.check()
    }

    fn charge_segment(
        &mut self,
        polygon: &[Point],
        curved: bool,
        style: &Stroke,
        transform: Affine,
        dash_min: Option<f64>,
    ) -> Result<(), GeometryLimit> {
        // A Bézier's control polygon bounds its arc length, including loops
        // whose endpoints coincide. Measure dashes in user space, flattening in
        // device space, so shrinking or singular transforms cannot hide work.
        let length = polygon
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum::<f64>();
        let device = polygon
            .windows(2)
            .map(|pair| (transform * pair[0]).distance(transform * pair[1]))
            .sum::<f64>();
        let pieces = dash_min.map_or(1.0, |minimum| {
            2.0 + (length / minimum).ceil() * (style.dash_pattern.len() as f64 + 1.0)
        });
        let per_piece = if curved { 4096.0 } else { 32.0 };
        let cap_join = if matches!(style.start_cap, crate::kurbo::Cap::Round)
            || matches!(style.end_cap, crate::kurbo::Cap::Round)
            || matches!(style.join, crate::kurbo::Join::Round)
        {
            32.0
        } else {
            16.0
        };
        self.elements += pieces * (per_piece + cap_join);
        let scale = transform.as_coeffs()[..4]
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        // Charge cap circumference, both offsets and the longest permitted
        // miter for every piece, including zero-length dash transitions.
        let extension = 8.0 * style.width * style.miter_limit.max(1.0) * scale;
        self.device_length += device + pieces * extension;
        self.check()
    }

    fn check(&self) -> Result<(), GeometryLimit> {
        if !self.elements.is_finite()
            || self.elements > MAX_OUTLINE_ELEMENTS
            || !self.device_length.is_finite()
            || self.device_length > MAX_DEVICE_LENGTH
        {
            return Err(GeometryLimit);
        }
        Ok(())
    }
}
