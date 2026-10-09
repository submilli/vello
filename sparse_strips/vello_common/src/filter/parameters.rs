// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Parameter admission and per-pixel work estimates for SVG graph primitives.
use crate::color::{AlphaColor, Srgb};
use crate::filter_effects::{CompositeOperator, FilterPrimitive, LightSource, TransferFunction};

/// Largest executed convolution kernel, in entries. Like Chrome, larger kernels pass
/// their input through.
pub const MAX_KERNEL_ENTRIES: u32 = 256;
/// Largest admitted kernel, bounding the retained values of a pass-through kernel.
const MAX_ADMITTED_KERNEL_ENTRIES: u32 = 1 << 16;
/// Morphology radii are capped at this many pixels, as Chrome does (crbug.com/1123035).
pub const MAX_MORPHOLOGY_RADIUS: u32 = 256;
/// Octaves past this are below 8-bit resolution; Chrome caps noise at the same count.
pub const MAX_TURBULENCE_OCTAVES: u32 = 9;

/// Whether the renderer can execute `primitive` with these parameters.
pub(crate) fn valid(primitive: &FilterPrimitive) -> bool {
    match primitive {
        FilterPrimitive::ColorMatrix { matrix } => matrix.iter().copied().all(finite),
        // CSS chains pad their raster by the blur radius, so it stays bounded.
        FilterPrimitive::GaussianBlur { std_deviation, .. } => deviation(*std_deviation),
        // Work does not grow with these magnitudes: SVG blurs past the box range
        // are decimated, and SVG nodes are clipped to admitted regions.
        FilterPrimitive::AxisGaussianBlur {
            std_deviation_x,
            std_deviation_y,
            ..
        } => magnitude(*std_deviation_x) && magnitude(*std_deviation_y),
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
        } => finite(*dx) && finite(*dy) && deviation(*std_deviation) && valid_color(*color),
        FilterPrimitive::AxisDropShadow {
            dx,
            dy,
            std_deviation_x,
            std_deviation_y,
            color,
        } => {
            finite(*dx)
                && finite(*dy)
                && magnitude(*std_deviation_x)
                && magnitude(*std_deviation_y)
                && valid_color(*color)
        }
        FilterPrimitive::Offset { dx, dy } => finite(*dx) && finite(*dy),
        FilterPrimitive::Flood { color } => valid_color(*color),
        FilterPrimitive::Composite { operator } => match operator {
            CompositeOperator::Arithmetic { k1, k2, k3, k4 } => {
                [*k1, *k2, *k3, *k4].into_iter().all(finite)
            }
            _ => true,
        },
        // Every peniko mix is an SVG blend mode.
        FilterPrimitive::Blend { .. } | FilterPrimitive::Tile => true,
        FilterPrimitive::ComponentTransfer {
            red_function,
            green_function,
            blue_function,
            alpha_function,
        } => [red_function, green_function, blue_function, alpha_function]
            .into_iter()
            .flatten()
            .all(valid_transfer),
        // Execution caps radii at `MAX_MORPHOLOGY_RADIUS`.
        FilterPrimitive::Morphology {
            radius_x, radius_y, ..
        } => magnitude(*radius_x) && magnitude(*radius_y),
        FilterPrimitive::ConvolveMatrix { kernel } => {
            let entries = kernel.columns.checked_mul(kernel.rows);
            entries.is_some_and(|n| n > 0 && n <= MAX_ADMITTED_KERNEL_ENTRIES)
                && kernel.values.len() == entries.unwrap_or(0) as usize
                && kernel.target_x < kernel.columns
                && kernel.target_y < kernel.rows
                && kernel.values.iter().all(|v| v.is_finite())
                && kernel.divisor.is_finite()
                && kernel.divisor != 0.0
                && kernel.bias.is_finite()
        }
        // The noise generator clamps any finite seed into its range.
        FilterPrimitive::Turbulence {
            base_frequency_x,
            base_frequency_y,
            seed,
            ..
        } => magnitude(*base_frequency_x) && magnitude(*base_frequency_y) && seed.is_finite(),
        FilterPrimitive::DisplacementMap { scale, .. } => scale.is_finite(),
        FilterPrimitive::DiffuseLighting {
            surface_scale,
            diffuse_constant,
            color,
            light_source,
        } => {
            surface_scale.is_finite()
                && diffuse_constant.is_finite()
                && valid_color(*color)
                && valid_light(light_source)
        }
        FilterPrimitive::SpecularLighting {
            surface_scale,
            specular_constant,
            specular_exponent,
            color,
            light_source,
        } => {
            surface_scale.is_finite()
                && specular_constant.is_finite()
                && specular_exponent.is_finite()
                && valid_color(*color)
                && valid_light(light_source)
        }
        // Image inputs need resources the graph does not own.
        FilterPrimitive::Image { .. } => false,
    }
}

/// Per-pixel work, in units of roughly one four-channel multiply-add, used to
/// admit a graph's total work. Blurs and shadows run up to four box/offset passes
/// with conversions; morphology runs prefix/suffix scans per axis; lighting
/// samples nine heights and evaluates vectors and a power; noise evaluates four
/// lattice gradients per octave.
pub(crate) fn cost(primitive: &FilterPrimitive) -> u64 {
    match primitive {
        FilterPrimitive::GaussianBlur { .. }
        | FilterPrimitive::AxisGaussianBlur { .. }
        | FilterPrimitive::DropShadow { .. }
        | FilterPrimitive::DropShadowOnly { .. }
        | FilterPrimitive::AxisDropShadow { .. } => 64,
        FilterPrimitive::Morphology { .. }
        | FilterPrimitive::DiffuseLighting { .. }
        | FilterPrimitive::SpecularLighting { .. } => 32,
        FilterPrimitive::ConvolveMatrix { kernel } => {
            let entries = u64::from(kernel.columns) * u64::from(kernel.rows);
            // Oversized kernels pass their input through.
            if entries > u64::from(MAX_KERNEL_ENTRIES) {
                1
            } else {
                entries
            }
        }
        FilterPrimitive::Turbulence { num_octaves, .. } => {
            16 * u64::from((*num_octaves).clamp(1, MAX_TURBULENCE_OCTAVES))
        }
        _ => 4,
    }
}

/// Extra per-pixel work of converting a linearRGB node's inputs and output, which
/// evaluates several powers per pixel.
pub(crate) const LINEAR_CONVERSION_COST: u64 = 24;

fn finite(v: f32) -> bool {
    v.is_finite() && v.abs() <= 1e6
}

fn deviation(v: f32) -> bool {
    finite(v) && v >= 0.0
}

/// A nonnegative finite value whose size cannot increase work.
fn magnitude(v: f32) -> bool {
    v.is_finite() && v >= 0.0
}

fn valid_color(color: AlphaColor<Srgb>) -> bool {
    color.components.iter().copied().all(finite)
}

fn valid_transfer(function: &TransferFunction) -> bool {
    match function {
        TransferFunction::Identity => true,
        TransferFunction::Table { values } | TransferFunction::Discrete { values } => {
            values.len() <= 256 && values.iter().copied().all(finite)
        }
        TransferFunction::Linear { slope, intercept } => finite(*slope) && finite(*intercept),
        TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => finite(*amplitude) && finite(*exponent) && *exponent >= 0.0 && finite(*offset),
    }
}

/// Light geometry only enters per-pixel vector math, so any finite value is safe.
fn valid_light(light: &LightSource) -> bool {
    match light {
        LightSource::Distant { azimuth, elevation } => azimuth.is_finite() && elevation.is_finite(),
        LightSource::Point { x, y, z } => [*x, *y, *z].iter().all(|v| v.is_finite()),
        LightSource::Spot {
            x,
            y,
            z,
            points_at_x,
            points_at_y,
            points_at_z,
            specular_exponent,
            limiting_cone_angle,
        } => {
            [
                *x,
                *y,
                *z,
                *points_at_x,
                *points_at_y,
                *points_at_z,
                *specular_exponent,
            ]
            .iter()
            .all(|v| v.is_finite())
                && limiting_cone_angle.is_none_or(f32::is_finite)
        }
    }
}
