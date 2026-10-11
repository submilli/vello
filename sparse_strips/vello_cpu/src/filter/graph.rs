// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bounded SVG DAG execution over caller-supplied, already-rasterized source
//! pixels, each node over its own frame (see `vello_common`'s `filter::frames`).

use alloc::vec::Vec;
use vello_common::filter::frames::Frame;
use vello_common::filter::graph::{GraphError, Input, MAX_COORDINATE, Node, SvgGraph};
use vello_common::filter::offset::whole_pixels;
use vello_common::filter_effects::FilterPrimitive;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

use super::bounds::PixelBounds;
use super::channels::TRANSPARENT;
use super::context::ScratchBuffer;
use super::svg_frame::{Framed, alpha_only, raster};
use super::svg_node::Context;

/// Execute an admitted SVG graph over the whole of `pixels` without changing
/// them on admission failure; see [`apply_svg_graph_to`].
pub fn apply_svg_graph(
    graph: &SvgGraph,
    pixels: &mut Pixmap,
    origin: Point,
) -> Result<(), GraphError> {
    let size = (f64::from(pixels.width()), f64::from(pixels.height()));
    apply_svg_graph_to(graph, pixels, origin, Rect::from_origin_size(origin, size))
}

/// Execute an admitted SVG graph without changing `pixels` on admission
/// failure, producing its output over `destination` (in graph coordinates) and
/// leaving the rest of `pixels` transparent. `origin`, a whole-pixel position,
/// locates the first pixel in the resolved graph coordinate space. The caller
/// supplies all source/output padding and admits the complete drawing
/// transaction before compositing this result into its destination.
pub fn apply_svg_graph_to(
    graph: &SvgGraph,
    pixels: &mut Pixmap,
    origin: Point,
    destination: Rect,
) -> Result<(), GraphError> {
    // Within the graph's coordinate range (plus a raster's side), frame
    // coordinates stay exact in f64, so frames match their rasters' sizes.
    let range = MAX_COORDINATE + f64::from(u16::MAX);
    let within = |v: f64| v.is_finite() && v.abs() <= range;
    if !within(origin.x) || !within(origin.y) || origin != origin.round() {
        return Err(GraphError::Parameter);
    }
    let size = (f64::from(pixels.width()), f64::from(pixels.height()));
    let frames = graph.admit_raster(Rect::from_origin_size(origin, size), destination)?;
    let empty = Pixmap::new(pixels.width(), pixels.height());
    let source = Framed {
        pixels: core::mem::replace(pixels, empty),
        frame: frames.raster(),
    };
    let mut execution = Execution {
        graph,
        source,
        results: Vec::with_capacity(graph.nodes().len()),
        scratch: ScratchBuffer::default(),
    };
    for (index, node) in graph.nodes().iter().enumerate() {
        let result = frames.node(index).map(|frame| execution.node(node, frame));
        execution.results.push(result);
    }
    // The output index was checked before this execution began.
    if let Some(output) = &execution.results[graph.output()] {
        output.copy_into(pixels, frames.raster());
    }
    Ok(())
}

/// One graph execution: the source, each node's result so far and shared
/// scratch storage.
struct Execution<'a> {
    graph: &'a SvgGraph,
    source: Framed,
    results: Vec<Option<Framed>>,
    scratch: ScratchBuffer,
}

impl Execution<'_> {
    /// Compute `node` over its frame and keep what its consumers read.
    fn node(&mut self, node: &Node, frame: &Frame) -> Framed {
        let region = node.region.intersect(self.graph.region());
        let mut result = match node.primitive {
            FilterPrimitive::Offset { dx, dy } => self.offset(node.input, frame.kept, [dx, dy]),
            FilterPrimitive::Tile => self.tile(node, frame.kept),
            _ => self.computed(node, frame, region),
        };
        clip(&mut result.pixels, region, result.frame.origin());
        result
    }

    /// Execute a primitive over its working frame, then keep what is read.
    fn computed(&mut self, node: &Node, frame: &Frame, region: Rect) -> Framed {
        // Generators read no input.
        let mut output = match node.primitive {
            FilterPrimitive::Flood { .. }
            | FilterPrimitive::Turbulence { .. }
            | FilterPrimitive::Image { .. } => Framed {
                pixels: raster(frame.work),
                frame: frame.work,
            },
            _ => self.input(node.input, frame.work),
        };
        let other = node.input2.map(|input| self.input(input, frame.work));
        let origin = frame.work.origin();
        let (width, height) = (output.pixels.width(), output.pixels.height());
        let bounds = |rect: Rect| PixelBounds::of(rect, origin, width, height);
        let context = Context {
            region,
            crop: bounds(region),
            input: bounds(input_region(self.graph, node.input)),
            layer: bounds(frame.layer),
            origin,
            space: node.color_space,
        };
        super::svg_node::execute(
            &node.primitive,
            &mut output.pixels,
            other.as_ref().map(|other| &other.pixels),
            &context,
            &mut self.scratch,
        );
        if frame.kept == frame.work {
            return output;
        }
        output.reframe(frame.kept)
    }

    /// Offsets move whole pixels (as the graph's reach assumes); reading from
    /// the input's frame moves paint from any distance.
    fn offset(&self, input: Input, kept: Rect, [dx, dy]: [f32; 2]) -> Framed {
        let shift = whole_pixels(dx, dy);
        self.mapped(input, kept, |x, y| (x - shift.x, y - shift.y))
    }

    /// Repeat the input's bounds across the node's frame, or draw nothing
    /// where Chrome cannot tile.
    fn tile(&self, node: &Node, kept: Rect) -> Framed {
        let Some(period) = self.graph.tile_period(node, kept) else {
            return Framed {
                pixels: raster(kept),
                frame: kept,
            };
        };
        let wrap =
            |value: f64, start: f64, end: f64| start + (value - start).rem_euclid(end - start);
        self.mapped(node.input, kept, |x, y| {
            (wrap(x, period.x0, period.x1), wrap(y, period.y0, period.y1))
        })
    }

    /// A raster over `frame` whose pixel at each position is the input's at
    /// `map(position)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "Raster coordinates are below 2^16."
    )]
    fn mapped(&self, input: Input, frame: Rect, map: impl Fn(f64, f64) -> (f64, f64)) -> Framed {
        let mut pixels = raster(frame);
        if let Some(held) = self.held(input) {
            let width = usize::from(pixels.width());
            for (index, pixel) in pixels.data_mut().iter_mut().enumerate() {
                let x = frame.x0 + (index % width) as f64;
                let y = frame.y0 + (index / width) as f64;
                let (sx, sy) = map(x, y);
                *pixel = alpha_only(held.get(sx, sy), input);
            }
            pixels.recompute_may_have_transparency();
        }
        Framed { pixels, frame }
    }

    /// `input` over `frame`: transparent wherever it holds nothing.
    fn input(&self, input: Input, frame: Rect) -> Framed {
        let mut framed = match self.held(input) {
            Some(held) => held.reframe(frame),
            None => Framed {
                pixels: raster(frame),
                frame,
            },
        };
        if input == Input::SourceAlpha {
            for pixel in framed.pixels.data_mut() {
                *pixel = alpha_only(*pixel, input);
            }
        }
        framed
    }

    fn held(&self, input: Input) -> Option<&Framed> {
        match input {
            Input::SourceGraphic | Input::SourceAlpha => Some(&self.source),
            Input::Result(index) => self.results.get(index)?.as_ref(),
        }
    }
}

/// Where an input's content lies: the source's content or an earlier node's region.
fn input_region(graph: &SvgGraph, input: Input) -> Rect {
    let region = match input {
        Input::SourceGraphic | Input::SourceAlpha => graph.source_region(),
        Input::Result(index) => graph.nodes()[index].region,
    };
    region.intersect(graph.region())
}

fn clip(pixels: &mut Pixmap, region: Rect, origin: Point) {
    if region.width() <= 0.0 || region.height() <= 0.0 {
        pixels.data_mut().fill(TRANSPARENT);
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
            *p = TRANSPARENT;
        }
    }
    pixels.recompute_may_have_transparency();
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::PremulRgba8;
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
    fn rescaled_blurs_keep_source_paint_beyond_the_source_region() {
        // Chrome 154 (`blurs.js` `offCanvasRegion`): a filter region from -300 to
        // 900 over a 600-pixel canvas, red paint at -200..0 off the canvas, blurred
        // by 150; the canvas reads these alphas every 40 pixels from 0.
        let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
        n.region = Rect::new(-300.0, 0.0, 900.0, 1.0);
        n.primitive = FilterPrimitive::AxisGaussianBlur {
            std_deviation_x: 150.0,
            std_deviation_y: 0.0,
            edge_mode: vello_common::filter_effects::EdgeMode::None,
        };
        let graph = SvgGraph::new([n], Rect::new(-300.0, 0.0, 900.0, 1.0), 0)
            .unwrap()
            .with_source_region(Rect::new(0.0, 0.0, 600.0, 1.0))
            .unwrap();
        let mut pixels = Pixmap::new(1200, 1);
        for pixel in &mut pixels.data_mut()[100..300] {
            *pixel = PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            };
        }
        apply_svg_graph(&graph, &mut pixels, Point::new(-300.0, 0.0)).unwrap();
        for (index, expected) in [106_u8, 89, 69, 50, 34, 21, 12, 6, 2]
            .into_iter()
            .enumerate()
        {
            let actual = pixels.data()[300 + index * 40].a;
            assert!(actual.abs_diff(expected) <= 1, "{index}: {actual}");
        }
    }
    #[test]
    fn unextended_convolutions_read_source_paint_beyond_the_source_region() {
        use vello_common::filter_effects::{ConvolutionKernel, EdgeMode};
        // Chrome 154 (`sources.js` `convolveEdgeNone` and `convolveEdgeDuplicate`): a
        // 5×1 box over a 300-pixel canvas whose filter region starts at 0, red paint
        // at -10..0 off the canvas. `none` reads that paint; `duplicate` extends the
        // canvas's own edge, which is transparent.
        let convolve = |edge_mode| {
            let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
            n.region = Rect::new(0.0, 0.0, 300.0, 1.0);
            n.primitive = FilterPrimitive::ConvolveMatrix {
                kernel: ConvolutionKernel {
                    columns: 5,
                    rows: 1,
                    values: alloc::vec![1.0; 5],
                    target_x: 2,
                    target_y: 0,
                    divisor: 5.0,
                    bias: 0.0,
                    edge_mode,
                    preserve_alpha: false,
                },
            };
            let graph = SvgGraph::new([n], Rect::new(-1e6, -1e6, 1e6, 1e6), 0)
                .unwrap()
                .with_source_region(Rect::new(0.0, 0.0, 300.0, 1.0))
                .unwrap();
            let mut pixels = Pixmap::new(320, 1);
            for pixel in &mut pixels.data_mut()[10..20] {
                *pixel = PremulRgba8 {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255,
                };
            }
            apply_svg_graph(&graph, &mut pixels, Point::new(-20.0, 0.0)).unwrap();
            [20, 21, 22].map(|index| pixels.data()[index].a)
        };
        assert_eq!(convolve(EdgeMode::None), [102, 51, 0]);
        assert_eq!(convolve(EdgeMode::Duplicate), [0, 0, 0]);
    }
    #[test]
    fn displacements_read_source_paint_beyond_the_source_region() {
        use vello_common::color::AlphaColor;
        use vello_common::filter_effects::ColorChannel;
        // Chrome 154 (`sources.js` `displacement`): a filter region from 100 to 400
        // over a 600-pixel canvas, red paint at -150..-50 off the canvas, displaced
        // 200 pixels right (and less than one down) by a map whose red channel is 0.
        let mut map = node(matrices::IDENTITY, Input::SourceGraphic);
        map.region = Rect::new(100.0, 0.0, 400.0, 2.0);
        map.primitive = FilterPrimitive::Flood {
            color: AlphaColor::from_rgba8(0, 128, 0, 255),
        };
        let mut displaced = node(matrices::IDENTITY, Input::SourceGraphic);
        displaced.region = map.region;
        displaced.input2 = Some(Input::Result(0));
        displaced.primitive = FilterPrimitive::DisplacementMap {
            scale: 400.0,
            x_channel: ColorChannel::Red,
            y_channel: ColorChannel::Green,
        };
        let graph = SvgGraph::new([map, displaced], Rect::new(-1e6, -1e6, 1e6, 1e6), 1)
            .unwrap()
            .with_source_region(Rect::new(0.0, 0.0, 600.0, 2.0))
            .unwrap();
        let mut pixels = Pixmap::new(800, 2);
        for pixel in (0..2).flat_map(|row| row * 800 + 50..row * 800 + 150) {
            let pixel = &mut pixels.data_mut()[pixel];
            *pixel = PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            };
        }
        apply_svg_graph(&graph, &mut pixels, Point::new(-200.0, 0.0)).unwrap();
        let alpha = [80, 100, 140, 149, 150, 160].map(|x| pixels.data()[200 + x].a);
        assert_eq!(alpha, [0, 255, 255, 255, 0, 0]);
    }
    #[test]
    fn far_apart_reads_draw_like_near_ones() {
        // Chrome 154 (`frames.js` and `regions.js` `offsetBeyondBudget`): paint a
        // flood generates 100,000 pixels away, offset into view, and a tile whose
        // period lies 500,000 pixels away cover the canvas.
        use vello_common::color::palette::css::BLUE;
        let huge = Rect::new(-1e6, -1e6, 1e6, 1e6);
        let flood = |region| Node {
            primitive: FilterPrimitive::Flood { color: BLUE },
            region,
            ..node(matrices::IDENTITY, Input::SourceGraphic)
        };
        let offset = Node {
            primitive: FilterPrimitive::Offset {
                dx: 100_000.0,
                dy: 0.0,
            },
            region: huge,
            ..node(matrices::IDENTITY, Input::Result(0))
        };
        let tile = Node {
            primitive: FilterPrimitive::Tile,
            region: huge,
            ..node(matrices::IDENTITY, Input::Result(0))
        };
        let period = Rect::new(-500_000.0, 10.0, -499_963.0, 33.0);
        for nodes in [[flood(huge), offset], [flood(period), tile]] {
            let graph = SvgGraph::new(nodes, huge, 1).unwrap();
            let mut pixels = Pixmap::new(300, 150);
            apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
            assert!(pixels.data().iter().all(|pixel| pixel.a == 255));
        }
    }
    #[test]
    fn tiles_beyond_chromes_largest_allocation_draw_nothing() {
        use vello_common::color::palette::css::BLUE;
        let huge = Rect::new(-1e6, -1e6, 1e6, 1e6);
        let tiled = |width: f64| {
            let flood = Node {
                primitive: FilterPrimitive::Flood { color: BLUE },
                region: Rect::from_center_size((2.0, 1.0), (width, 20_000.0)),
                ..node(matrices::IDENTITY, Input::SourceGraphic)
            };
            let tile = Node {
                primitive: FilterPrimitive::Tile,
                region: huge,
                ..node(matrices::IDENTITY, Input::Result(0))
            };
            let graph = SvgGraph::new([flood, tile], huge, 1).unwrap();
            let mut pixels = Pixmap::new(4, 2);
            apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
            pixels.data()[0].a
        };
        assert_eq!(tiled(26_812.0), 255);
        assert_eq!(tiled(26_814.0), 0);
    }
    #[test]
    fn outputs_beyond_the_destination_leave_the_raster_transparent() {
        let mut n = node(matrices::IDENTITY, Input::SourceGraphic);
        n.region = Rect::new(5.0, 0.0, 6.0, 1.0);
        let graph = SvgGraph::new([n], Rect::new(0.0, 0.0, 6.0, 1.0), 0).unwrap();
        let mut pixels = Pixmap::new(4, 1);
        pixels.data_mut().fill(PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        apply_svg_graph(&graph, &mut pixels, Point::ORIGIN).unwrap();
        assert!(pixels.data().iter().all(|pixel| pixel.a == 0));
        // Only the destination is produced.
        let whole = SvgGraph::new(
            [node(matrices::IDENTITY, Input::SourceGraphic)],
            Rect::new(0.0, 0.0, 4.0, 1.0),
            0,
        )
        .unwrap();
        pixels.data_mut().fill(PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        let destination = Rect::new(1.0, 0.0, 2.0, 1.0);
        apply_svg_graph_to(&whole, &mut pixels, Point::ORIGIN, destination).unwrap();
        let alpha: Vec<u8> = pixels.data().iter().map(|pixel| pixel.a).collect();
        assert_eq!(alpha, [0, 255, 0, 0]);
    }
    #[test]
    fn fractional_and_far_raster_origins_are_rejected() {
        let graph = SvgGraph::new(
            [node(matrices::IDENTITY, Input::SourceGraphic)],
            Rect::new(0.0, 0.0, 2.0, 1.0),
            0,
        )
        .unwrap();
        let mut pixels = Pixmap::new(2, 1);
        for origin in [Point::new(0.5, 0.0), Point::new(0.0, 1e16)] {
            assert_eq!(
                apply_svg_graph(&graph, &mut pixels, origin),
                Err(GraphError::Parameter)
            );
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
