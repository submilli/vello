// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Pixel rectangles of graph inputs, which neighborhood primitives extend.
use vello_common::filter_effects::EdgeMode;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};

/// A half-open pixel rectangle inside one raster; it may be empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PixelBounds {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl PixelBounds {
    /// The raster pixels a region covers, rounded outward as region clipping does.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Values are clamped to the raster extent before conversion."
    )]
    pub(super) fn of(region: Rect, origin: Point, width: u16, height: u16) -> Self {
        let clamp = |v: f64, max: u16| v.clamp(0.0, f64::from(max)) as usize;
        let x0 = clamp((region.x0 - origin.x).floor(), width);
        let y0 = clamp((region.y0 - origin.y).floor(), height);
        Self {
            x0,
            y0,
            x1: clamp((region.x1 - origin.x).ceil(), width).max(x0),
            y1: clamp((region.y1 - origin.y).ceil(), height).max(y0),
        }
    }

    pub(super) fn intersect(&self, other: Self) -> Self {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        Self {
            x0,
            y0,
            x1: self.x1.min(other.x1).max(x0),
            y1: self.y1.min(other.y1).max(y0),
        }
    }

    pub(super) fn extend_x(&self, x: i64, edge: EdgeMode) -> Option<usize> {
        extend(x, self.x0, self.x1, edge)
    }

    pub(super) fn extend_y(&self, y: i64, edge: EdgeMode) -> Option<usize> {
        extend(y, self.y0, self.y1, edge)
    }
}

/// Map a coordinate outside `[start, end)` according to the edge mode.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Raster coordinates fit i64; mapped results lie inside the bounds."
)]
fn extend(value: i64, start: usize, end: usize, edge: EdgeMode) -> Option<usize> {
    if start >= end {
        return None;
    }
    let (start, end) = (start as i64, end as i64);
    if (start..end).contains(&value) {
        return Some(value as usize);
    }
    let size = end - start;
    // Displaced samples saturate at the i64 range, so this must not overflow.
    let offset = value.saturating_sub(start);
    Some(match edge {
        EdgeMode::None => return None,
        EdgeMode::Duplicate => value.clamp(start, end - 1) as usize,
        EdgeMode::Wrap => (start + offset.rem_euclid(size)) as usize,
        EdgeMode::Mirror => {
            let reflected = offset.rem_euclid(2 * size);
            let inside = if reflected < size {
                reflected
            } else {
                2 * size - reflected - 1
            };
            (start + inside) as usize
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_extend_relative_to_the_input_rectangle() {
        let bounds = PixelBounds {
            x0: 2,
            y0: 0,
            x1: 5,
            y1: 1,
        };
        assert_eq!(bounds.extend_x(1, EdgeMode::None), None);
        assert_eq!(bounds.extend_x(1, EdgeMode::Duplicate), Some(2));
        assert_eq!(bounds.extend_x(7, EdgeMode::Duplicate), Some(4));
        assert_eq!(bounds.extend_x(1, EdgeMode::Wrap), Some(4));
        assert_eq!(bounds.extend_x(5, EdgeMode::Wrap), Some(2));
        assert_eq!(bounds.extend_x(1, EdgeMode::Mirror), Some(2));
        for edge in [
            EdgeMode::None,
            EdgeMode::Duplicate,
            EdgeMode::Wrap,
            EdgeMode::Mirror,
        ] {
            for extreme in [i64::MIN, i64::MAX] {
                assert!(
                    bounds
                        .extend_x(extreme, edge)
                        .is_none_or(|x| (2..5).contains(&x))
                );
            }
        }
        let empty = PixelBounds {
            x0: 3,
            y0: 0,
            x1: 3,
            y1: 1,
        };
        assert_eq!(empty.extend_x(3, EdgeMode::Duplicate), None);
    }

    #[test]
    fn regions_round_outward_and_clamp_to_the_raster() {
        let bounds = PixelBounds::of(Rect::new(1.5, -9.0, 3.2, 2.0), Point::new(1.0, 0.0), 4, 4);
        assert_eq!(
            bounds,
            PixelBounds {
                x0: 0,
                y0: 0,
                x1: 3,
                y1: 2
            }
        );
    }
}
