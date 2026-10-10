// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! A raster sized by a graph's reach reproduces the destination exactly.

use super::graph::apply_svg_graph;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::color::palette::css::{BLUE, WHITE};
use vello_common::filter::graph::{ColorSpace, Input, Node, SvgGraph};
use vello_common::filter_effects::{
    ColorChannel, CompositeOperator, ConvolutionKernel, EdgeMode, FilterPrimitive, LightSource,
    MorphologyOperator, TurbulenceType,
};
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

/// Every region lies inside this raster, so executing over it is exact.
const FULL: Rect = Rect::new(-160.0, -160.0, 160.0, 160.0);
const DESTINATION: Rect = Rect::new(0.0, 0.0, 24.0, 16.0);

#[test]
fn reach_sized_rasters_reproduce_the_destination() {
    for (name, nodes) in graphs() {
        let output = nodes.len() - 1;
        let graph = SvgGraph::new(nodes, FULL, output)
            .unwrap()
            .with_source_region(Rect::new(-40.0, -30.0, 50.0, 40.0))
            .unwrap();
        let reach = graph.reach(DESTINATION);
        let mut extent = reach.extent(DESTINATION);
        if let Some(source) = reach.source {
            extent = extent.union(source);
        }
        assert!(
            extent.area() < FULL.area() / 4.0,
            "{name}: reach {extent:?}"
        );
        let expected = destination(&graph, FULL);
        assert!(expected.iter().any(|pixel| pixel.a != 0), "{name}: blank");
        assert_eq!(destination(&graph, extent), expected, "{name}");
    }
}

/// Execute over a raster covering `extent` with the source pattern, and read
/// back the destination.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Test rasters are small and integer-aligned."
)]
fn destination(graph: &SvgGraph, extent: Rect) -> Vec<PremulRgba8> {
    let (width, height) = (extent.width() as u16, extent.height() as u16);
    let mut pixels = Pixmap::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let point = (
                extent.x0 as i64 + i64::from(x),
                extent.y0 as i64 + i64::from(y),
            );
            pixels.set_pixel(x, y, source(point));
        }
    }
    apply_svg_graph(graph, &mut pixels, Point::new(extent.x0, extent.y0)).unwrap();
    let mut out = Vec::new();
    for y in DESTINATION.y0 as i64..DESTINATION.y1 as i64 {
        for x in DESTINATION.x0 as i64..DESTINATION.x1 as i64 {
            let column = (x - extent.x0 as i64) as u16;
            let row = (y - extent.y0 as i64) as u16;
            out.push(pixels.sample(column, row));
        }
    }
    out
}

/// Opaque stripes over part of the canvas, transparent elsewhere.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Channel values stay within 0..=255."
)]
fn source((x, y): (i64, i64)) -> PremulRgba8 {
    if !(-40..50).contains(&x) || !(-30..40).contains(&y) || (x + 2 * y).rem_euclid(7) < 3 {
        return PremulRgba8::from_u32(0);
    }
    let shade = (x * 5 + y * 3).rem_euclid(200) as u8;
    PremulRgba8 {
        r: shade,
        g: 255 - shade,
        b: 40,
        a: 255,
    }
}

fn graphs() -> Vec<(&'static str, Vec<Node>)> {
    let edge = Rect::new(-90.0, -5.0, 20.0, 12.0);
    vec![
        (
            "offset turbulence",
            vec![noise(0.07), offset(70.0, -40.0, 0)],
        ),
        (
            "tiled flood",
            vec![
                flood(Rect::new(-150.0, 30.0, -120.0, 47.0)),
                node(FilterPrimitive::Tile, Input::Result(0)),
                offset(5.0, 3.0, 1),
            ],
        ),
        (
            "blurred turbulence",
            vec![noise(0.1), unary(blur(4.0), 0), offset(-50.0, 20.0, 1)],
        ),
        (
            "lit morphology",
            vec![
                noise(0.05),
                unary(
                    FilterPrimitive::Morphology {
                        operator: MorphologyOperator::Dilate,
                        radius_x: 3.0,
                        radius_y: 2.0,
                    },
                    0,
                ),
                unary(
                    lighting(LightSource::Point {
                        x: -60.0,
                        y: 40.0,
                        z: 30.0,
                    }),
                    1,
                ),
                offset(30.0, 30.0, 2),
            ],
        ),
        (
            "displaced turbulence",
            vec![
                noise(0.06),
                noise(0.02),
                binary(
                    FilterPrimitive::DisplacementMap {
                        scale: 30.0,
                        x_channel: ColorChannel::Red,
                        y_channel: ColorChannel::Green,
                    },
                    0,
                    1,
                ),
                offset(40.0, 0.0, 2),
            ],
        ),
        (
            "flood-mapped displacement",
            vec![
                noise(0.06),
                node(
                    FilterPrimitive::Flood {
                        color: vello_common::color::palette::css::RED,
                    },
                    Input::SourceGraphic,
                ),
                binary(
                    FilterPrimitive::DisplacementMap {
                        scale: 80.0,
                        x_channel: ColorChannel::Red,
                        y_channel: ColorChannel::Green,
                    },
                    0,
                    1,
                ),
            ],
        ),
        (
            "lit edge",
            vec![
                Node {
                    region: Rect::new(-100.0, -100.0, 24.0, 100.0),
                    ..noise(0.05)
                },
                unary(
                    lighting(LightSource::Point {
                        x: 30.0,
                        y: 8.0,
                        z: 20.0,
                    }),
                    0,
                ),
            ],
        ),
        (
            "source shadow",
            vec![
                Node {
                    primitive: FilterPrimitive::DropShadow {
                        dx: 20.0,
                        dy: -10.0,
                        std_deviation: 3.0,
                        color: BLUE,
                        edge_mode: EdgeMode::None,
                    },
                    ..node(FilterPrimitive::Tile, Input::SourceGraphic)
                },
                offset(-15.0, 12.0, 0),
            ],
        ),
        (
            "shadowed offset",
            vec![
                flood(Rect::new(-150.0, -20.0, -110.0, 30.0)),
                offset(120.0, 0.0, 0),
                Node {
                    primitive: FilterPrimitive::DropShadow {
                        dx: -40.0,
                        dy: 6.0,
                        std_deviation: 2.0,
                        color: WHITE,
                        edge_mode: EdgeMode::None,
                    },
                    ..unary(FilterPrimitive::Tile, 1)
                },
            ],
        ),
        (
            "wrapped kernel",
            vec![
                noise(0.08),
                Node {
                    region: edge,
                    ..unary(convolve(EdgeMode::Wrap), 0)
                },
            ],
        ),
        (
            "duplicated kernel",
            vec![
                noise(0.08),
                Node {
                    region: edge,
                    ..unary(convolve(EdgeMode::Duplicate), 0)
                },
            ],
        ),
        (
            "duplicated far edge",
            vec![
                Node {
                    region: Rect::new(-150.0, -20.0, -50.0, 40.0),
                    ..noise(0.08)
                },
                unary(convolve(EdgeMode::Duplicate), 0),
            ],
        ),
        (
            "mirrored far edge",
            vec![
                Node {
                    region: Rect::new(-150.0, -20.0, -50.0, 40.0),
                    ..noise(0.08)
                },
                unary(convolve(EdgeMode::Mirror), 0),
            ],
        ),
        (
            "spot lit offset",
            vec![
                noise(0.04),
                offset(-70.0, 0.0, 0),
                Node {
                    region: edge,
                    ..unary(
                        lighting(LightSource::Spot {
                            x: 10.0,
                            y: 10.0,
                            z: 40.0,
                            points_at_x: -20.0,
                            points_at_y: 0.0,
                            points_at_z: 0.0,
                            specular_exponent: 2.0,
                            limiting_cone_angle: None,
                        }),
                        1,
                    )
                },
            ],
        ),
        (
            "arithmetic composite",
            vec![
                flood(Rect::new(-120.0, -100.0, -60.0, 100.0)),
                offset(90.0, 0.0, 0),
                noise(0.03),
                binary(
                    FilterPrimitive::Composite {
                        operator: CompositeOperator::Arithmetic {
                            k1: 0.5,
                            k2: 0.5,
                            k3: 0.5,
                            k4: 0.1,
                        },
                    },
                    1,
                    2,
                ),
            ],
        ),
    ]
}

fn node(primitive: FilterPrimitive, input: Input) -> Node {
    Node {
        primitive,
        input,
        input2: None,
        region: FULL,
        color_space: ColorSpace::Srgb,
    }
}

fn unary(primitive: FilterPrimitive, input: usize) -> Node {
    node(primitive, Input::Result(input))
}

fn binary(primitive: FilterPrimitive, input: usize, input2: usize) -> Node {
    Node {
        input2: Some(Input::Result(input2)),
        ..unary(primitive, input)
    }
}

fn flood(region: Rect) -> Node {
    Node {
        region,
        ..node(FilterPrimitive::Flood { color: BLUE }, Input::SourceGraphic)
    }
}

fn offset(dx: f32, dy: f32, input: usize) -> Node {
    unary(FilterPrimitive::Offset { dx, dy }, input)
}

fn noise(frequency: f32) -> Node {
    node(
        FilterPrimitive::Turbulence {
            base_frequency_x: frequency,
            base_frequency_y: frequency,
            num_octaves: 2,
            seed: 3.0,
            stitch_tiles: false,
            turbulence_type: TurbulenceType::Turbulence,
        },
        Input::SourceGraphic,
    )
}

fn blur(std_deviation: f32) -> FilterPrimitive {
    FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    }
}

fn lighting(light_source: LightSource) -> FilterPrimitive {
    FilterPrimitive::DiffuseLighting {
        surface_scale: 4.0,
        diffuse_constant: 1.0,
        color: WHITE,
        light_source,
    }
}

fn convolve(edge_mode: EdgeMode) -> FilterPrimitive {
    FilterPrimitive::ConvolveMatrix {
        kernel: ConvolutionKernel {
            columns: 3,
            rows: 3,
            values: vec![1.0, 0.0, -1.0, 2.0, 0.5, -2.0, 1.0, 0.0, -1.0],
            target_x: 1,
            target_y: 1,
            divisor: 1.0,
            bias: 0.5,
            edge_mode,
            preserve_alpha: false,
        },
    }
}
