// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The offset filter.

use crate::kurbo::Vec2;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;

/// Translation/shift filter.
///
/// This shifts the input image by `(dx, dy)` in device pixel space.
#[derive(Clone, Copy, Debug)]
pub struct Offset {
    /// The x-offset that should be applied.
    pub dx: f32,
    /// The y-offset that should be applied.
    pub dy: f32,
}

impl Offset {
    /// Create a new offset filter.
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }
}

/// The whole-pixel translation an SVG graph offset or shadow applies: offsets
/// round to whole pixels, as Chrome's do on an integer-translated layer.
pub fn whole_pixels(dx: f32, dy: f32) -> Vec2 {
    Vec2::new(f64::from(dx.round()), f64::from(dy.round()))
}
