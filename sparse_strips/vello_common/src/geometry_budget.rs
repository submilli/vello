// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cumulative, pre-allocation admission for vector flattening and stroke expansion.
//! Pixel clipping cannot bound geometry work performed before clipping.

mod raster;
mod stroke;

pub use raster::RasterBudget;
pub use stroke::StrokeBudget;

/// Geometry exceeds the coordinate, segment, length or stroke-work admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometryLimit;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::{Affine, BezPath, Stroke};

    #[test]
    fn distant_curve_controls_are_refused_before_flattening() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.curve_to((1e20, 1e20), (-1e20, 1e20), (1.0, 0.0));
        assert!(
            RasterBudget::default()
                .charge(&path, Affine::IDENTITY)
                .is_err()
        );
    }

    #[test]
    fn dash_expansion_is_admitted_before_outline_allocation() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((100.0, 0.0));
        let stroke = Stroke::new(1.0).with_dashes(0.0, [1e-12, 1e-12]);
        assert!(
            StrokeBudget::default()
                .charge(&path, &stroke, Affine::IDENTITY)
                .is_err()
        );
    }

    #[test]
    fn geometry_length_is_cumulative_across_paths() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((65536.0, 0.0));
        let mut budget = RasterBudget::default();
        budget.charge(&path, Affine::IDENTITY).unwrap();
        budget.charge(&path, Affine::IDENTITY).unwrap();
        assert!(budget.charge(&path, Affine::IDENTITY).is_err());
    }
}
