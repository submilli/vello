// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Execution of one admitted SVG graph node over its input rasters.
use super::bounds::PixelBounds;
use super::channels::{TRANSPARENT, encode};
use super::context::ScratchBuffer;
use super::filter_lowp;
use super::lighting::{Lighting, Reflection};
use super::svg_blur::Shadow;
use super::turbulence::Turbulence;
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::pixmap::Pixmap;

/// Where a node runs: its clipped region and that region's pixels, its primary
/// input's content pixels, the raster origin in filter coordinates and its
/// color-interpolation-filters space.
pub(super) struct Context {
    pub region: Rect,
    pub crop: PixelBounds,
    pub input: PixelBounds,
    pub origin: Point,
    pub space: ColorSpace,
}

/// Replace `output`, initially the primary input, with the primitive's result.
/// Graph admission guarantees parameters, arity (`other` for binary primitives)
/// and supported primitive kinds; anything else leaves the input unchanged.
pub(super) fn execute(
    primitive: &FilterPrimitive,
    output: &mut Pixmap,
    other: Option<&Pixmap>,
    context: &Context,
    scratch: &mut ScratchBuffer,
) {
    let space = context.space;
    match primitive {
        FilterPrimitive::ColorMatrix { matrix } => {
            super::color_matrix::apply_in_space(output, matrix, space);
        }
        FilterPrimitive::Flood { color } => {
            output
                .data_mut()
                .fill(encode(color.components, ColorSpace::Srgb));
            output.recompute_may_have_transparency();
        }
        FilterPrimitive::Composite { operator } => {
            if let Some(other) = other {
                super::svg_composite::composite(output, other, *operator, space);
            }
        }
        FilterPrimitive::Blend { mode } => {
            if let Some(other) = other {
                super::svg_blend::blend(output, other, *mode, space);
            }
        }
        FilterPrimitive::DisplacementMap {
            scale,
            x_channel,
            y_channel,
        } => {
            if let Some(map) = other {
                super::displacement::apply(
                    output,
                    map,
                    *scale,
                    [*x_channel, *y_channel],
                    context.input,
                    space,
                );
            }
        }
        FilterPrimitive::ComponentTransfer {
            red_function,
            green_function,
            blue_function,
            alpha_function,
        } => super::svg_channels::transfer(
            output,
            [red_function, green_function, blue_function, alpha_function],
            space,
        ),
        // Graph blurs are decal, like Chrome's filter blurs.
        FilterPrimitive::GaussianBlur { std_deviation, .. } => {
            super::svg_blur::blur(output, [*std_deviation; 2], space);
        }
        FilterPrimitive::AxisGaussianBlur {
            std_deviation_x,
            std_deviation_y,
            ..
        } => super::svg_blur::blur(output, [*std_deviation_x, *std_deviation_y], space),
        FilterPrimitive::DropShadow { .. }
        | FilterPrimitive::DropShadowOnly { .. }
        | FilterPrimitive::AxisDropShadow { .. } => {
            if let Some(shadow) = shadow(primitive) {
                super::svg_blur::shadow(output, &shadow, space, scratch);
            }
        }
        FilterPrimitive::Morphology {
            operator,
            radius_x,
            radius_y,
        } => super::morphology::apply(output, *operator, [*radius_x, *radius_y], space),
        FilterPrimitive::ConvolveMatrix { kernel } => {
            // As in Chrome, duplicate and wrap extend the input inside the node's crop,
            // while `none` reads the uncropped input and treats its outside as transparent.
            let bounds = match kernel.edge_mode {
                EdgeMode::None => context.input,
                _ => context.crop.intersect(context.input),
            };
            super::convolve::apply(output, kernel, bounds, space);
        }
        FilterPrimitive::Turbulence {
            base_frequency_x,
            base_frequency_y,
            num_octaves,
            seed,
            stitch_tiles,
            turbulence_type,
        } => {
            let turbulence = Turbulence {
                base_frequency: [*base_frequency_x, *base_frequency_y],
                num_octaves: *num_octaves,
                seed: *seed,
                stitch_tiles: *stitch_tiles,
                kind: *turbulence_type,
            };
            super::turbulence::apply(output, &turbulence, context.region, context.origin, space);
        }
        FilterPrimitive::Tile => tile(output, context),
        FilterPrimitive::DiffuseLighting {
            surface_scale,
            diffuse_constant,
            color,
            light_source,
        } => {
            let lighting = Lighting {
                reflection: Reflection::Diffuse {
                    constant: *diffuse_constant,
                },
                surface_scale: *surface_scale,
                color: *color,
                light: light_source,
            };
            super::lighting::apply(output, &lighting, context.crop, context.origin, space);
        }
        FilterPrimitive::SpecularLighting {
            surface_scale,
            specular_constant,
            specular_exponent,
            color,
            light_source,
        } => {
            let lighting = Lighting {
                reflection: Reflection::Specular {
                    constant: *specular_constant,
                    exponent: *specular_exponent,
                },
                surface_scale: *surface_scale,
                color: *color,
                light: light_source,
            };
            super::lighting::apply(output, &lighting, context.crop, context.origin, space);
        }
        FilterPrimitive::Image {
            image,
            source,
            destination,
        } => {
            let placement = super::image::Placement {
                image: image.pixmap(),
                source: *source,
                destination: *destination,
            };
            super::image::draw(output, &placement, context.crop, context.origin);
        }
        FilterPrimitive::Offset { .. } => {
            filter_lowp(
                &Filter::from_primitive(primitive.clone()),
                output,
                scratch,
                Affine::IDENTITY,
            );
        }
    }
}

fn shadow(primitive: &FilterPrimitive) -> Option<Shadow> {
    Some(match *primitive {
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
        } => Shadow {
            dx,
            dy,
            std_deviation: [std_deviation; 2],
            color,
            with_source: matches!(primitive, FilterPrimitive::DropShadow { .. }),
        },
        FilterPrimitive::AxisDropShadow {
            dx,
            dy,
            std_deviation_x,
            std_deviation_y,
            color,
        } => Shadow {
            dx,
            dy,
            std_deviation: [std_deviation_x, std_deviation_y],
            color,
            with_source: true,
        },
        _ => return None,
    })
}

/// Repeat the input's pixel rectangle across the node's region.
fn tile(output: &mut Pixmap, context: &Context) {
    let (tile, target) = (context.input, context.crop);
    let width = usize::from(output.width());
    let source = output.clone();
    for y in target.y0..target.y1 {
        for x in target.x0..target.x1 {
            let pixel = match (
                tile.extend_x(x as i64, EdgeMode::Wrap),
                tile.extend_y(y as i64, EdgeMode::Wrap),
            ) {
                (Some(sx), Some(sy)) => source.data()[sy * width + sx],
                _ => TRANSPARENT,
            };
            output.data_mut()[y * width + x] = pixel;
        }
    }
    output.recompute_may_have_transparency();
}
