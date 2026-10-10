// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The pixels an SVG graph reads to produce its output over a destination, as
//! Skia maps a desired output back through image filters
//! (`SkImageFilter_Base::getInputBounds`).
//!
//! A raster covering the destination and every node's reach computes the
//! destination exactly, however large the regions are: paint that a generator
//! creates far away and an offset, tile or neighborhood brings in is computed
//! where it is read. Each node also reaches the pixels its own crop-dependent
//! reads touch, so clipping its crop to the raster changes nothing it produces.
//! The exception is a blur above the box range, whose rescale centers on its
//! input's layer as the raster clips it.

use super::graph::{Input, MAX_BLUR_DEVIATION, Node, SvgGraph};
use crate::filter_effects::{EdgeMode, FilterPrimitive};
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Rect, Vec2};
use alloc::vec;
use alloc::vec::Vec;

/// What a graph reads, on the pixel grid at integer coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Reach {
    /// By node index: the pixels the node must hold, or `None` when the
    /// output does not depend on it.
    pub nodes: Vec<Option<Rect>>,
    /// The source graphic's pixels the graph reads, if any.
    pub source: Option<Rect>,
}

impl Reach {
    /// The smallest rectangle holding `destination` and every node's pixels.
    pub fn extent(&self, destination: Rect) -> Rect {
        self.nodes
            .iter()
            .flatten()
            .fold(outward(destination), |extent, rect| extent.union(*rect))
    }
}

impl SvgGraph {
    /// What the graph reads to produce its output over `destination`.
    pub fn reach(&self, destination: Rect) -> Reach {
        let nodes = self.nodes();
        let mut needs: Vec<Option<Rect>> = vec![None; nodes.len()];
        let mut reach = Reach {
            nodes: vec![None; nodes.len()],
            source: None,
        };
        needs[self.output()] = pixels_within(destination, self.node_region(self.output()));
        for (index, node) in nodes.iter().enumerate().rev() {
            let Some(need) = needs[index] else {
                continue;
            };
            let reads = self.reads(node, need);
            let own = reads
                .own
                .and_then(|own| pixels_within(own, self.node_region(index)));
            reach.nodes[index] = union(Some(need), own);
            for (input, rect) in [(Some(node.input), reads.input), (node.input2, reads.input2)] {
                let (Some(input), Some(rect)) = (input, rect) else {
                    continue;
                };
                match input {
                    Input::SourceGraphic | Input::SourceAlpha => {
                        reach.source = union(reach.source, pixels_within(rect, self.region()));
                    }
                    Input::Result(earlier) => {
                        let rect = pixels_within(rect, self.node_region(earlier));
                        needs[earlier] = union(needs[earlier], rect);
                    }
                }
            }
        }
        reach
    }

    /// What `node` reads to produce `need`: from each input, and within its own crop.
    fn reads(&self, node: &Node, need: Rect) -> Reads {
        let around = |margin: Vec2| Some(inflate(need, margin));
        let pointwise = Reads {
            input: Some(need),
            input2: node.input2.map(|_| need),
            own: None,
        };
        match &node.primitive {
            FilterPrimitive::Flood { .. }
            | FilterPrimitive::Turbulence { .. }
            | FilterPrimitive::Image { .. } => Reads::default(),
            FilterPrimitive::ColorMatrix { .. }
            | FilterPrimitive::ComponentTransfer { .. }
            | FilterPrimitive::Composite { .. }
            | FilterPrimitive::Blend { .. } => pointwise,
            // Offsets move whole pixels (see `vello_cpu`'s `offset_pixels`).
            FilterPrimitive::Offset { dx, dy } => Reads {
                input: Some(need - Vec2::new(f64::from(dx.round()), f64::from(dy.round()))),
                ..Reads::default()
            },
            FilterPrimitive::Tile => Reads {
                input: tile(need, self.input_bounds(node.input)),
                ..Reads::default()
            },
            FilterPrimitive::GaussianBlur { std_deviation, .. } => {
                Reads::around(around(blur(*std_deviation, *std_deviation)))
            }
            FilterPrimitive::AxisGaussianBlur {
                std_deviation_x,
                std_deviation_y,
                ..
            } => Reads::around(around(blur(*std_deviation_x, *std_deviation_y))),
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                ..
            } => shadow(need, [*dx, *dy], blur(*std_deviation, *std_deviation), true),
            FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                ..
            } => shadow(
                need,
                [*dx, *dy],
                blur(*std_deviation, *std_deviation),
                false,
            ),
            FilterPrimitive::AxisDropShadow {
                dx,
                dy,
                std_deviation_x,
                std_deviation_y,
                ..
            } => shadow(
                need,
                [*dx, *dy],
                blur(*std_deviation_x, *std_deviation_y),
                true,
            ),
            // Radii are capped as execution caps them.
            FilterPrimitive::Morphology { .. } => {
                let expansion = node.primitive.filter_expansion();
                Reads::around(around(Vec2::new(expansion.x1, expansion.y1)))
            }
            // Surface normals read one neighboring pixel.
            FilterPrimitive::DiffuseLighting { .. } | FilterPrimitive::SpecularLighting { .. } => {
                Reads::around(around(Vec2::new(1.0, 1.0)))
            }
            FilterPrimitive::ConvolveMatrix { kernel } => {
                let margin = Vec2::new(f64::from(kernel.columns), f64::from(kernel.rows));
                let mut reads = inflate(need, margin);
                // Edge modes other than `none` map reads past the input's bounds
                // within the crop back into them: to the nearest edge, however far,
                // or across to the far side.
                if kernel.edge_mode != EdgeMode::None {
                    let crop = self.node_region_of(node);
                    if let Some(bounds) = pixels_within(self.input_bounds(node.input), crop) {
                        reads = extend(reads, bounds);
                    }
                }
                Reads::around(Some(reads))
            }
            // Samples round to the nearest pixel within half the scale; samples outside
            // the input's bounds are transparent whatever the crop.
            FilterPrimitive::DisplacementMap { scale, .. } => {
                let reach = (f64::from(*scale).abs() / 2.0).ceil() + 1.0;
                Reads {
                    input: around(Vec2::new(reach, reach)),
                    input2: Some(need),
                    own: None,
                }
            }
        }
    }

    /// Where a node's result can be nonzero, rounded outward to whole pixels.
    fn node_region(&self, index: usize) -> Rect {
        self.node_region_of(&self.nodes()[index])
    }

    fn node_region_of(&self, node: &Node) -> Rect {
        outward(node.region.intersect(self.region()))
    }

    /// The pixels an input holds content in, which tiles repeat and edge modes extend.
    fn input_bounds(&self, input: Input) -> Rect {
        let region = match input {
            Input::SourceGraphic | Input::SourceAlpha => self.source_region(),
            Input::Result(index) => self.nodes()[index].region,
        };
        outward(region.intersect(self.region()))
    }
}

/// One node's reads: from its primary and secondary inputs, and positions
/// within its own crop that crop-dependent edges test.
#[derive(Default)]
struct Reads {
    input: Option<Rect>,
    input2: Option<Rect>,
    own: Option<Rect>,
}

impl Reads {
    /// A unary neighborhood: the input and its own crop around the output.
    fn around(rect: Option<Rect>) -> Self {
        Self {
            input: rect,
            input2: None,
            own: rect,
        }
    }
}

/// Reach of a blur by these deviations, clamped as execution clamps them:
/// three boxes stay within 3σ, plus rounding, and a rescaled blur's bilinear
/// steps add up to four texels of its low-resolution image, each at most one
/// pixel per 128 of deviation.
fn blur(x: f32, y: f32) -> Vec2 {
    let reach = |sigma: f32| {
        let sigma = f64::from(sigma.clamp(0.0, MAX_BLUR_DEVIATION));
        (3.0 * sigma).ceil() + 2.0 + 4.0 * (sigma / 128.0).ceil()
    };
    Vec2::new(reach(x), reach(y))
}

/// A shadow reads its blurred input from behind the offset, and a shadow
/// drawn with its source reads the source where it lands.
fn shadow(need: Rect, [dx, dy]: [f32; 2], blur: Vec2, with_source: bool) -> Reads {
    let behind = need - Vec2::new(f64::from(dx.round()), f64::from(dy.round()));
    let mut reads = inflate(behind, blur);
    if with_source {
        reads = reads.union(need);
    }
    // The blur's decal edges and the offset stay inside the reads.
    Reads::around(Some(reads))
}

/// A tile reads its input's bounds where `need` wraps around them and the
/// same pixels where it does not.
fn tile(need: Rect, bounds: Rect) -> Option<Rect> {
    if bounds.width() <= 0.0 || bounds.height() <= 0.0 {
        return None;
    }
    let (x0, x1) = span((need.x0, need.x1), (bounds.x0, bounds.x1));
    let (y0, y1) = span((need.y0, need.y1), (bounds.y0, bounds.y1));
    Some(Rect::new(x0, y0, x1, y1))
}

/// Reads past `bounds` that an edge mode maps back into them reach the whole
/// span of the bounds on that axis.
fn extend(reads: Rect, bounds: Rect) -> Rect {
    let (x0, x1) = span((reads.x0, reads.x1), (bounds.x0, bounds.x1));
    let (y0, y1) = span((reads.y0, reads.y1), (bounds.y0, bounds.y1));
    Rect::new(x0, y0, x1, y1).union(reads)
}

/// `reads` when it stays within `bounds`, else the whole of `bounds`.
fn span(reads: (f64, f64), bounds: (f64, f64)) -> (f64, f64) {
    if reads.0 >= bounds.0 && reads.1 <= bounds.1 {
        reads
    } else {
        bounds
    }
}

fn inflate(rect: Rect, margin: Vec2) -> Rect {
    rect.inflate(margin.x, margin.y)
}

/// The pixels a region covers, rounded outward as region clipping rounds.
fn outward(rect: Rect) -> Rect {
    Rect::new(
        rect.x0.floor(),
        rect.y0.floor(),
        rect.x1.ceil(),
        rect.y1.ceil(),
    )
}

/// `rect` within `bounds`, or `None` when no pixel remains.
fn pixels_within(rect: Rect, bounds: Rect) -> Option<Rect> {
    let rect = outward(rect).intersect(bounds);
    (rect.width() > 0.0 && rect.height() > 0.0).then_some(rect)
}

fn union(a: Option<Rect>, b: Option<Rect>) -> Option<Rect> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.union(b)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::palette::css::BLUE;
    use crate::filter::graph::ColorSpace;
    use crate::filter_effects::{ConvolutionKernel, TurbulenceType};

    const HUGE: Rect = Rect::new(-1e6, -1e6, 1e6, 1e6);
    const BITMAP: Rect = Rect::new(0.0, 0.0, 300.0, 150.0);

    fn node(primitive: FilterPrimitive, input: Input, region: Rect) -> Node {
        Node {
            primitive,
            input,
            input2: None,
            region,
            color_space: ColorSpace::Srgb,
        }
    }

    fn flood(region: Rect) -> Node {
        node(
            FilterPrimitive::Flood { color: BLUE },
            Input::SourceGraphic,
            region,
        )
    }

    fn offset(dx: f32, dy: f32, input: usize) -> Node {
        node(
            FilterPrimitive::Offset { dx, dy },
            Input::Result(input),
            HUGE,
        )
    }

    fn reach_of(nodes: Vec<Node>) -> Reach {
        let output = nodes.len() - 1;
        let graph = SvgGraph::new(nodes, HUGE, output).unwrap();
        graph.with_source_region(BITMAP).unwrap().reach(BITMAP)
    }

    #[test]
    fn offsets_reach_generated_paint_wherever_it_lies() {
        let reach = reach_of(vec![flood(HUGE), offset(250.0, 0.0, 0)]);
        assert_eq!(reach.nodes[0], Some(Rect::new(-250.0, 0.0, 50.0, 150.0)));
        assert_eq!(reach.extent(BITMAP), Rect::new(-250.0, 0.0, 300.0, 150.0));
        // Far offsets read far paint without widening what the output holds.
        let far = reach_of(vec![flood(HUGE), offset(1e5, 0.0, 0)]);
        assert_eq!(
            far.nodes[0],
            Some(Rect::new(-1e5, 0.0, -1e5 + 300.0, 150.0))
        );
        assert_eq!(far.nodes[1], Some(BITMAP));
        assert_eq!(far.source, None);
    }

    #[test]
    fn regions_and_unread_nodes_bound_the_reach() {
        let region = Rect::new(-100.0, 0.0, 100.0, 150.0);
        let unread = node(
            FilterPrimitive::Offset { dx: 0.0, dy: 0.0 },
            Input::SourceGraphic,
            HUGE,
        );
        let reach = reach_of(vec![unread, flood(region), offset(250.0, 0.0, 1)]);
        assert_eq!(reach.nodes[0], None);
        assert_eq!(reach.nodes[1], Some(Rect::new(-100.0, 0.0, 50.0, 150.0)));
        // Paint moved out of view reads nothing.
        let gone = reach_of(vec![flood(region), offset(-500.0, 0.0, 0)]);
        assert_eq!(gone.nodes[0], None);
    }

    #[test]
    fn tiles_read_whole_periods_only_where_they_wrap() {
        let source = Rect::new(-700.0, 10.0, -663.0, 33.0);
        let tiled = reach_of(vec![
            flood(source),
            node(FilterPrimitive::Tile, Input::Result(0), HUGE),
        ]);
        assert_eq!(tiled.nodes[0], Some(source));
        let large = Rect::new(-1000.0, -1000.0, 1000.0, 1000.0);
        let identity = reach_of(vec![
            flood(large),
            node(FilterPrimitive::Tile, Input::Result(0), HUGE),
        ]);
        assert_eq!(identity.nodes[0], Some(BITMAP));
    }

    #[test]
    fn neighborhoods_reach_inputs_and_their_own_crop() {
        let blurred = reach_of(vec![node(
            FilterPrimitive::GaussianBlur {
                std_deviation: 10.0,
                edge_mode: EdgeMode::None,
            },
            Input::SourceGraphic,
            HUGE,
        )]);
        assert_eq!(blurred.source, Some(BITMAP.inflate(36.0, 36.0)));
        assert_eq!(blurred.nodes[0], Some(BITMAP.inflate(36.0, 36.0)));
        let noise = node(
            FilterPrimitive::Turbulence {
                base_frequency_x: 0.05,
                base_frequency_y: 0.05,
                num_octaves: 1,
                seed: 0.0,
                stitch_tiles: false,
                turbulence_type: TurbulenceType::Turbulence,
            },
            Input::SourceGraphic,
            HUGE,
        );
        let mut displaced = node(
            FilterPrimitive::DisplacementMap {
                scale: 40.0,
                x_channel: crate::filter_effects::ColorChannel::Red,
                y_channel: crate::filter_effects::ColorChannel::Green,
            },
            Input::Result(0),
            HUGE,
        );
        displaced.input2 = Some(Input::Result(1));
        let reach = reach_of(vec![noise.clone(), noise, displaced]);
        assert_eq!(reach.nodes[0], Some(BITMAP.inflate(21.0, 21.0)));
        assert_eq!(reach.nodes[1], Some(BITMAP));
    }

    #[test]
    fn edge_modes_reach_the_whole_span_of_their_bounds() {
        let bounds = Rect::new(-500.0, 0.0, 302.0, 150.0);
        let kernel = |edge_mode| ConvolutionKernel {
            columns: 3,
            rows: 1,
            values: vec![1.0; 3],
            target_x: 1,
            target_y: 0,
            divisor: 3.0,
            bias: 0.0,
            edge_mode,
            preserve_alpha: false,
        };
        let convolve = |edge_mode| {
            reach_of(vec![
                flood(bounds),
                node(
                    FilterPrimitive::ConvolveMatrix {
                        kernel: kernel(edge_mode),
                    },
                    Input::Result(0),
                    HUGE,
                ),
            ])
        };
        for edge_mode in [EdgeMode::Wrap, EdgeMode::Duplicate, EdgeMode::Mirror] {
            assert_eq!(
                convolve(edge_mode).nodes[0],
                Some(Rect::new(-500.0, 0.0, 302.0, 150.0))
            );
        }
        assert_eq!(
            convolve(EdgeMode::None).nodes[0],
            Some(Rect::new(-3.0, 0.0, 302.0, 150.0))
        );
    }
}
