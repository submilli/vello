// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bound device-space geometry before renderer curve flattening.

use super::GeometryLimit;
use crate::kurbo::{Affine, BezPath, PathEl, Point};

/// Cumulative admission before renderer geometry expansion.
#[derive(Default, Debug)]
pub struct RasterBudget {
    length: f64,
    elements: usize,
}

impl RasterBudget {
    /// Charge the complete transformed path before flattening or stroking.
    pub fn charge(&mut self, path: &BezPath, transform: Affine) -> Result<(), GeometryLimit> {
        self.elements += path.elements().len();
        if self.elements > 65_536 {
            return Err(GeometryLimit);
        }
        let mut current = Point::ORIGIN;
        let mut start = current;
        for element in path.elements() {
            match *element {
                PathEl::MoveTo(point) => {
                    self.length += current.distance(start);
                    current = device_point(point, transform)?;
                    start = current;
                }
                PathEl::LineTo(point) => self.edge(&mut current, point, transform)?,
                PathEl::QuadTo(a, b) => {
                    self.edge(&mut current, a, transform)?;
                    self.edge(&mut current, b, transform)?;
                }
                PathEl::CurveTo(a, b, c) => {
                    self.edge(&mut current, a, transform)?;
                    self.edge(&mut current, b, transform)?;
                    self.edge(&mut current, c, transform)?;
                }
                PathEl::ClosePath => {
                    self.length += current.distance(start);
                    current = start;
                }
            }
            if !self.length.is_finite() || self.length > 262_144.0 {
                return Err(GeometryLimit);
            }
        }
        self.length += current.distance(start);
        if !self.length.is_finite() || self.length > 262_144.0 {
            return Err(GeometryLimit);
        }
        Ok(())
    }

    fn edge(
        &mut self,
        current: &mut Point,
        point: Point,
        transform: Affine,
    ) -> Result<(), GeometryLimit> {
        let next = device_point(point, transform)?;
        // Control-polygon length bounds curve length even for closed loops.
        self.length += current.distance(next);
        *current = next;
        Ok(())
    }
}

fn device_point(point: Point, transform: Affine) -> Result<Point, GeometryLimit> {
    let point = transform * point;
    if !point.x.is_finite()
        || !point.y.is_finite()
        || point.x.abs() > 1_000_000.0
        || point.y.abs() > 1_000_000.0
    {
        return Err(GeometryLimit);
    }
    Ok(point)
}
