// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bounded SVG DAG execution over caller-supplied, already-rasterized source pixels.

use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::filter::graph::{GraphError, Input, SvgGraph};
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

use super::bounds::PixelBounds;
use super::context::ScratchBuffer;
use super::svg_node::Context;

/// Execute an admitted SVG graph without changing `pixels` on admission failure.
/// `origin` locates the first pixel in the resolved graph coordinate space. The
/// caller supplies all source/output padding and admits the complete drawing
/// transaction before compositing this result into its destination.
pub fn apply_svg_graph(
    graph: &SvgGraph,
    pixels: &mut Pixmap,
    origin: Point,
) -> Result<(), GraphError> {
    graph.admit_pixels(pixels.width(), pixels.height())?;
    if !origin.x.is_finite() || !origin.y.is_finite() {
        return Err(GraphError::Parameter);
    }
    let source = pixels.clone();
    let mut results: Vec<Pixmap> = Vec::with_capacity(graph.nodes().len());
    let mut scratch = ScratchBuffer::default();
    let (width, height) = (pixels.width(), pixels.height());
    let bounds = |region: Rect| PixelBounds::of(region, origin, width, height);
    for node in graph.nodes() {
        let mut output = input(node.input, &source, &results);
        let other = node.input2.map(|other| input(other, &source, &results));
        let region = node.region.intersect(graph.region());
        let context = Context {
            region,
            crop: bounds(region),
            input: bounds(input_region(graph, node.input)),
            origin,
            space: node.color_space,
        };
        super::svg_node::execute(
            &node.primitive,
            &mut output,
            other.as_ref(),
            &context,
            &mut scratch,
        );
        clip(&mut output, context.region, origin);
        results.push(output);
    }
    // The output index was checked before this execution began.
    let output = results.swap_remove(graph.output());
    *pixels = output;
    Ok(())
}

/// Where an input's content lies: the source's content or an earlier node's region.
fn input_region(graph: &SvgGraph, input: Input) -> Rect {
    let region = match input {
        Input::SourceGraphic | Input::SourceAlpha => graph.source_region(),
        Input::Result(index) => graph.nodes()[index].region,
    };
    region.intersect(graph.region())
}

fn input(input: Input, source: &Pixmap, results: &[Pixmap]) -> Pixmap {
    match input {
        Input::SourceGraphic => source.clone(),
        Input::Result(index) => results[index].clone(),
        Input::SourceAlpha => {
            let mut alpha = source.clone();
            for p in alpha.data_mut() {
                p.r = 0;
                p.g = 0;
                p.b = 0;
            }
            alpha
        }
    }
}

fn clip(pixels: &mut Pixmap, region: Rect, origin: Point) {
    if region.width() <= 0.0 || region.height() <= 0.0 {
        pixels.data_mut().fill(PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        });
        pixels.recompute_may_have_transparency();
        return;
    }
    let width = usize::from(pixels.width());
    if width == 0 {
        return;
    }
    for (index, p) in pixels.data_mut().iter_mut().enumerate() {
        let x = origin.x + (index % width) as f64;
        let y = origin.y + (index / width) as f64;
        // Chrome clips filter/subregions to outward integer pixel bounds.
        // Applying fractional coverage repeatedly would attenuate shared edges.
        if x < region.x0.floor()
            || x >= region.x1.ceil()
            || y < region.y0.floor()
            || y >= region.y1.ceil()
        {
            *p = PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
        }
    }
    pixels.recompute_may_have_transparency();
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::filter::graph::ColorSpace;
    use vello_common::filter::graph::Node;
    use vello_common::filter_effects::matrices;
    use vello_common::filter_effects::{CompositeOperator, FilterPrimitive};
    fn node(matrix: [f32; 20], input: Input) -> Node {
        Node {
            primitive: FilterPrimitive::ColorMatrix { matrix },
            input,
            input2: None,
            region: Rect::new(0.0, 0.0, 2.0, 1.0),
            color_space: ColorSpace::Srgb,
        }
    }
    #[test]
    fn blur_and_shadow_accept_tiny_odd_thin_and_empty_rasters() {
        use vello_common::filter_effects::EdgeMode;
        for (width, height) in [(0, 0), (0, 1), (1, 0), (1, 1), (3, 5), (1, 127), (65535, 1)] {
            for edge_mode in [EdgeMode::None, EdgeMode::Duplicate, EdgeMode::Wrap] {
                for primitive in [
                    FilterPrimitive::GaussianBlur {
                        std_deviation: 3.0,
                        edge_mode,
                    },
                    FilterPrimitive::DropShadow {
                        dx: 0.0,
                        dy: 0.0,
                        std_deviation: 3.0,
                        color: vello_common::color::palette::css::RED,
                        edge_mode,
                    },
                ] {
                    let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
                    n.primitive = primitive;
                    let graph = SvgGraph::new(
                        [n],
                        Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
                        0,
                    )
                    .unwrap();
                    let mut pixels = Pixmap::new(width, height);
                    pixels.data_mut().fill(PremulRgba8 {
                        r: 255,
                        g: 0,
                        b: 0,
                        a: 255,
                    });
                    apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
                    assert_eq!((pixels.width(), pixels.height()), (width, height));
                }
            }
        }
    }
    #[test]
    fn linear_surfaces_round_dark_values_like_chrome_but_color_filters_do_not() {
        // Chrome 154: a linearRGB blur stores 8-bit linear pixels, while a color
        // matrix converts in float. Gray levels 3, 6, 12 and 20 come back as below.
        let mut blur = node(matrices::IDENTITY, Input::SourceGraphic);
        blur.color_space = ColorSpace::LinearRgb;
        blur.primitive = FilterPrimitive::GaussianBlur {
            std_deviation: 0.5,
            edge_mode: vello_common::filter_effects::EdgeMode::None,
        };
        let mut matrix = node(matrices::IDENTITY, Input::SourceGraphic);
        matrix.color_space = ColorSpace::LinearRgb;
        for (primitive, expected) in [(blur, [0, 0, 13, 22]), (matrix, [3, 6, 12, 20])] {
            let graph = SvgGraph::new([primitive], Rect::new(0.0, 0.0, 4.0, 1.0), 0).unwrap();
            for (gray, expected) in [3, 6, 12, 20].into_iter().zip(expected) {
                let mut pixels = Pixmap::new(4, 1);
                pixels.data_mut().fill(PremulRgba8 {
                    r: gray,
                    g: gray,
                    b: gray,
                    a: 255,
                });
                apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
                assert_eq!(pixels.data()[1].r, expected, "{gray}");
            }
        }
    }
    #[test]
    fn linear_shadow_composites_in_linear_light_and_preserves_alpha_metadata() {
        use vello_common::filter_effects::EdgeMode;
        let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
        n.color_space = ColorSpace::LinearRgb;
        n.primitive = FilterPrimitive::DropShadow {
            dx: 0.0,
            dy: 0.0,
            std_deviation: 0.0,
            color: vello_common::color::palette::css::BLUE,
            edge_mode: EdgeMode::None,
        };
        let graph = SvgGraph::new([n.clone()], Rect::new(0.0, 0.0, 1.0, 1.0), 0).unwrap();
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert!((159..=161).contains(&pixels.data()[0].r));
        assert!((116..=118).contains(&pixels.data()[0].b));
        assert_eq!(pixels.data()[0].a, 192);
        n.primitive = FilterPrimitive::DropShadowOnly {
            dx: 0.0,
            dy: 0.0,
            std_deviation: 0.0,
            color: vello_common::color::AlphaColor::new([0.0, 0.0, 1.0, 0.5]),
            edge_mode: EdgeMode::None,
        };
        let graph = SvgGraph::new([n], Rect::new(0.0, 0.0, 1.0, 1.0), 0).unwrap();
        pixels.data_mut()[0] = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(
            pixels.data()[0],
            PremulRgba8 {
                r: 0,
                g: 0,
                b: 64,
                a: 64
            }
        );
        assert!(pixels.may_have_transparency());
    }
    #[test]
    fn neutral_linear_offsets_preserve_dark_colors_and_chrome_rounding() {
        let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
        n.color_space = ColorSpace::LinearRgb;
        n.primitive = FilterPrimitive::Offset { dx: 0.0, dy: 0.0 };
        let graph = SvgGraph::new([n.clone()], Rect::new(0.0, 0.0, 2.0, 1.0), 0).unwrap();
        let mut pixels = Pixmap::new(2, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 6,
            g: 6,
            b: 6,
            a: 255,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(pixels.data()[0].r, 6);
        n.primitive = FilterPrimitive::Offset { dx: 0.6, dy: 0.0 };
        let graph = SvgGraph::new([n], Rect::new(0.0, 0.0, 2.0, 1.0), 0).unwrap();
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(pixels.data()[0].a, 0);
        assert_eq!(pixels.data()[1].a, 255);
        assert_eq!(pixels.data()[1].r, 6);
    }
    #[test]
    fn repeated_fractional_regions_clip_outward_without_attenuating_edges() {
        let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
        n.region.x1 = 0.5;
        let mut second = n.clone();
        second.input = Input::Result(0);
        let graph = SvgGraph::new([n, second], Rect::new(0.0, 0.0, 0.5, 1.0), 1).unwrap();
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(pixels.data()[0].a, 255);
    }
    #[test]
    fn disjoint_and_empty_fractional_regions_do_not_round_into_visible_pixels() {
        for region in [
            Rect::new(0.6, 0.0, 1.0, 1.0),
            Rect::new(0.2, 0.0, 0.2, 1.0),
            Rect::new(0.0, 0.2, 0.4, 0.2),
        ] {
            let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
            n.region = region;
            let graph = SvgGraph::new([n], Rect::new(0.0, 0.0, 0.4, 1.0), 0).unwrap();
            let mut pixels = Pixmap::new(1, 1);
            pixels.data_mut()[0] = PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            };
            apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
            assert_eq!(pixels.data()[0].a, 0);
        }
    }
    #[test]
    fn invalid_origin_leaves_caller_pixels_unchanged() {
        let graph = SvgGraph::new(
            [node(matrices::IDENTITY, Input::SourceGraphic)],
            Rect::new(0.0, 0.0, 1.0, 1.0),
            0,
        )
        .unwrap();
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 7,
            g: 0,
            b: 0,
            a: 255,
        };
        let before = pixels.data().to_vec();
        assert_eq!(
            apply_svg_graph(&graph, &mut pixels, Point::new(f64::INFINITY, 0.0)),
            Err(GraphError::Parameter)
        );
        assert_eq!(pixels.data(), before);
    }
    #[test]
    fn branches_reuse_original_source_and_named_outputs() {
        let mut invert = matrices::IDENTITY;
        invert[0] = -1.0;
        invert[4] = 1.0;
        let mut half = matrices::IDENTITY;
        half[18] = 0.5;
        let graph = SvgGraph::new(
            [
                node(invert, Input::SourceGraphic),
                node(half, Input::SourceAlpha),
                node(matrices::IDENTITY, Input::Result(0)),
            ],
            Rect::new(0.0, 0.0, 2.0, 1.0),
            2,
        )
        .unwrap();
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(
            pixels.data()[0],
            PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }
        );
    }
    #[test]
    fn matrix_color_space_and_primitive_region_affect_observable_pixels() {
        let mut half = matrices::IDENTITY;
        half[0] = 0.5;
        let mut n = node(half, Input::SourceGraphic);
        n.color_space = ColorSpace::LinearRgb;
        n.region.x1 = 1.0;
        let graph = SvgGraph::new([n], Rect::new(0.0, 0.0, 2.0, 1.0), 0).unwrap();
        let mut pixels = Pixmap::new(2, 1);
        pixels.data_mut().fill(PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(pixels.data()[0].r, 188);
        assert_eq!(pixels.data()[1].a, 0);
    }
    #[test]
    fn source_alpha_binary_input_does_not_observe_previous_node_mutation() {
        let first = node(matrices::IDENTITY, Input::SourceAlpha);
        let second = Node {
            primitive: FilterPrimitive::Composite {
                operator: CompositeOperator::In,
            },
            input: Input::SourceGraphic,
            input2: Some(Input::Result(0)),
            ..node(matrices::IDENTITY, Input::SourceGraphic)
        };
        let graph = SvgGraph::new([first, second], Rect::new(0.0, 0.0, 2.0, 1.0), 1).unwrap();
        let mut pixels = Pixmap::new(1, 1);
        pixels.data_mut()[0] = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert_eq!(
            pixels.data()[0],
            PremulRgba8 {
                r: 64,
                g: 0,
                b: 0,
                a: 64
            }
        );
    }
}
