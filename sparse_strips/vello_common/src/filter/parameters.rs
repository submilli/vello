// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Finite parameter admission for shared unary CPU filter operations.
use crate::filter_effects::FilterPrimitive;
pub(crate) fn valid_unary_parameters(primitive: &FilterPrimitive) -> bool {
    let finite = |v: f32| v.is_finite() && v.abs() <= 1e6;
    match primitive {
        FilterPrimitive::ColorMatrix { matrix } => matrix.iter().copied().all(finite),
        FilterPrimitive::GaussianBlur { std_deviation, .. } => {
            finite(*std_deviation) && *std_deviation >= 0.0
        }
        FilterPrimitive::DropShadow {
            dx,
            dy,
            std_deviation,
            color,
            ..
        }
        | FilterPrimitive::DropShadowOnly {
            dx,
            dy,
            std_deviation,
            color,
            ..
        } => {
            [*dx, *dy, *std_deviation].into_iter().all(finite)
                && *std_deviation >= 0.0
                && color.components.iter().copied().all(finite)
        }
        FilterPrimitive::Offset { dx, dy } => finite(*dx) && finite(*dy),
        FilterPrimitive::Flood { color } => color.components.iter().copied().all(finite),
        _ => false,
    }
}
