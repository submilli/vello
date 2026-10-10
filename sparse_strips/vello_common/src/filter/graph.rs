// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Validated SVG graph plans. Parsing and resource authority belong to the caller.

pub use super::parameters::{MAX_KERNEL_ENTRIES, MAX_MORPHOLOGY_RADIUS, MAX_TURBULENCE_OCTAVES};
use crate::filter_effects::FilterPrimitive;
use crate::kurbo::Rect;
use alloc::vec::Vec;

/// Maximum retained primitives in one SVG graph.
pub const MAX_NODES: usize = 32;
/// Aggregate pixels retained by a graph, including source and working storage.
pub const MAX_INTERMEDIATE_PIXELS: usize = 16 * 1024 * 1024;
/// Rasters admitted per graph besides one per node: the source, alpha extraction
/// and working copies such as a blur's rescaled images.
pub const FIXED_SURFACES: usize = 16;
/// Aggregate estimated per-pixel operations over a graph's raster, bounding
/// neighborhood kernels and octave counts that pages control.
pub const MAX_WORK: u64 = 1 << 28;

/// A source or a previously computed result. Forward edges cannot be admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// Original source graphic, before any primitive changes it.
    SourceGraphic,
    /// Alpha channel of the original source, with zero RGB.
    SourceAlpha,
    /// An earlier node's output, addressed by its insertion index.
    Result(usize),
}

/// Color interpolation used by one SVG primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    /// Encoded sRGB components.
    Srgb,
    /// Linear-light sRGB components (SVG's default).
    LinearRgb,
}

/// A primitive plus its input connections and already-resolved user-space region.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// A supported renderer operation.
    pub primitive: FilterPrimitive,
    /// Primary source; generators ignore it.
    pub input: Input,
    /// Required for binary operations and forbidden for unary operations.
    pub input2: Option<Input>,
    /// Per-primitive clipping rectangle in the graph's coordinate space.
    pub region: Rect,
    /// SVG color-interpolation-filters value for this node.
    pub color_space: ColorSpace,
}

/// A graph whose size, topology, parameters and regions were validated.
/// Fields stay private so execution cannot encounter a caller-mutated invalid plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SvgGraph {
    nodes: Vec<Node>,
    region: Rect,
    source_region: Rect,
    output: usize,
}

/// Errors returned before a graph or intermediate allocation is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// Empty graph, excessive nodes or invalid output index.
    Size,
    /// An edge is forward, cyclic, absent or has an inappropriate arity.
    Input,
    /// Nonfinite, inverted or excessively large coordinate/parameter.
    Parameter,
    /// The renderer has no executor for this operation.
    Unsupported,
    /// Aggregate intermediate pixels exceed the renderer's bound.
    Memory,
    /// Aggregate estimated per-pixel work exceeds [`MAX_WORK`].
    Work,
}

impl SvgGraph {
    /// Admit an iterator incrementally; an unbounded iterator cannot grow storage.
    pub fn new(
        nodes: impl IntoIterator<Item = Node>,
        region: Rect,
        output: usize,
    ) -> Result<Self, GraphError> {
        validate_region(region)?;
        let mut admitted = Vec::new();
        for node in nodes {
            if admitted.len() == MAX_NODES {
                return Err(GraphError::Size);
            }
            validate_node(&node, admitted.len())?;
            admitted.push(node);
        }
        if admitted.is_empty() || output >= admitted.len() {
            return Err(GraphError::Size);
        }
        Ok(Self {
            nodes: admitted,
            region,
            source_region: region,
            output,
        })
    }

    /// Limit the source graphic's content, for example to a canvas bitmap. Edge modes
    /// extend inputs from their content, so sources stop at this rectangle.
    pub fn with_source_region(mut self, source_region: Rect) -> Result<Self, GraphError> {
        validate_region(source_region)?;
        self.source_region = source_region;
        Ok(self)
    }

    /// The largest raster area `admit_pixels` accepts for a graph of `nodes` nodes.
    pub fn max_admitted_area(nodes: usize) -> usize {
        MAX_INTERMEDIATE_PIXELS / (nodes + FIXED_SURFACES)
    }

    /// Immutable nodes in topological order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Final filter clip in resolved graph coordinates.
    pub fn region(&self) -> Rect {
        self.region
    }
    /// Content of `SourceGraphic` and `SourceAlpha` in graph coordinates.
    pub fn source_region(&self) -> Rect {
        self.source_region
    }
    /// The node whose pixels are the filter's output.
    pub fn output(&self) -> usize {
        self.output
    }

    /// Conservatively admit source, all results, alpha extraction and float convolution scratch
    /// surfaces, the image inputs the graph retains, and its total per-pixel work. Admission
    /// happens before allocations or destination modification.
    pub fn admit_pixels(&self, width: u16, height: u16) -> Result<(), GraphError> {
        let area = usize::from(width)
            .checked_mul(usize::from(height))
            .ok_or(GraphError::Memory)?;
        let surfaces = area
            .checked_mul(self.nodes.len() + FIXED_SURFACES)
            .ok_or(GraphError::Memory)?;
        if surfaces.saturating_add(self.image_pixels()) > MAX_INTERMEDIATE_PIXELS {
            return Err(GraphError::Memory);
        }
        let work: u64 = self
            .nodes
            .iter()
            .map(|node| {
                let conversion = match node.color_space {
                    ColorSpace::Srgb => 0,
                    ColorSpace::LinearRgb => super::parameters::LINEAR_CONVERSION_COST,
                };
                super::parameters::cost(&node.primitive) + conversion
            })
            .sum();
        if work.saturating_mul(area as u64) > MAX_WORK {
            return Err(GraphError::Work);
        }
        Ok(())
    }
}

impl SvgGraph {
    /// Pixels of the distinct image inputs, which the graph keeps alive.
    fn image_pixels(&self) -> usize {
        let mut total = 0_usize;
        for (index, node) in self.nodes.iter().enumerate() {
            let FilterPrimitive::Image { image, .. } = &node.primitive else {
                continue;
            };
            let shared = self.nodes[..index].iter().any(|earlier| {
                matches!(&earlier.primitive, FilterPrimitive::Image { image: other, .. } if other == image)
            });
            if !shared {
                let pixmap = image.pixmap();
                total = total
                    .saturating_add(usize::from(pixmap.width()) * usize::from(pixmap.height()));
            }
        }
        total
    }
}

/// Largest coordinate magnitude in a graph's regions and image placements.
pub const MAX_COORDINATE: f64 = 1e6;

fn validate_region(rect: Rect) -> Result<(), GraphError> {
    valid_rect(rect).then_some(()).ok_or(GraphError::Parameter)
}

/// A finite, ordered rectangle within the graph's coordinate range.
pub(crate) fn valid_rect(rect: Rect) -> bool {
    [rect.x0, rect.y0, rect.x1, rect.y1]
        .into_iter()
        .all(|v| v.is_finite() && v.abs() <= MAX_COORDINATE)
        && rect.x1 >= rect.x0
        && rect.y1 >= rect.y0
}

fn valid_input(input: Input, before: usize) -> bool {
    match input {
        Input::SourceGraphic | Input::SourceAlpha => true,
        Input::Result(index) => index < before,
    }
}

fn validate_node(node: &Node, before: usize) -> Result<(), GraphError> {
    validate_region(node.region)?;
    if !valid_input(node.input, before)
        || node.input2.is_some_and(|input| !valid_input(input, before))
    {
        return Err(GraphError::Input);
    }
    let binary = matches!(
        node.primitive,
        FilterPrimitive::Composite { .. }
            | FilterPrimitive::Blend { .. }
            | FilterPrimitive::DisplacementMap { .. }
    );
    if binary != node.input2.is_some() {
        return Err(GraphError::Input);
    }
    if !super::parameters::valid(&node.primitive) {
        return Err(GraphError::Parameter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::matrices;

    fn node(input: Input) -> Node {
        Node {
            primitive: FilterPrimitive::ColorMatrix {
                matrix: matrices::IDENTITY,
            },
            input,
            input2: None,
            region: Rect::new(0.0, 0.0, 10.0, 10.0),
            color_space: ColorSpace::Srgb,
        }
    }
    #[test]
    fn graph_retains_source_and_branch_edges_without_accepting_forward_edges() {
        let nodes = [
            node(Input::SourceGraphic),
            node(Input::SourceAlpha),
            node(Input::Result(0)),
        ];
        let graph = SvgGraph::new(nodes, Rect::new(0.0, 0.0, 10.0, 10.0), 2).unwrap();
        assert_eq!(graph.nodes()[2].input, Input::Result(0));
        assert_eq!(
            SvgGraph::new([node(Input::Result(0))], graph.region(), 0),
            Err(GraphError::Input)
        );
        assert_eq!(
            SvgGraph::new([node(Input::Result(1))], graph.region(), 0),
            Err(GraphError::Input)
        );
    }
    #[test]
    fn graph_bounds_iteration_parameters_and_intermediate_pixels() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(
            SvgGraph::new(
                core::iter::repeat_with(|| node(Input::SourceGraphic)),
                rect,
                0
            ),
            Err(GraphError::Size)
        );
        let mut invalid = node(Input::SourceGraphic);
        invalid.region.x0 = f64::NAN;
        assert_eq!(
            SvgGraph::new([invalid], rect, 0),
            Err(GraphError::Parameter)
        );
        let graph = SvgGraph::new([node(Input::SourceGraphic)], rect, 0).unwrap();
        assert_eq!(graph.admit_pixels(4096, 4096), Err(GraphError::Memory));
        assert!(graph.admit_pixels(512, 512).is_ok());
    }
    #[test]
    fn svg_primitives_admit_arity_kernels_and_finite_magnitudes() {
        use crate::filter_effects::{
            ColorChannel, ConvolutionKernel, EdgeMode, MorphologyOperator,
        };
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let with = |primitive, input2| Node {
            primitive,
            input2,
            ..node(Input::SourceGraphic)
        };
        let displacement = FilterPrimitive::DisplacementMap {
            scale: 1e30,
            x_channel: ColorChannel::Red,
            y_channel: ColorChannel::Alpha,
        };
        assert!(
            SvgGraph::new(
                [with(displacement.clone(), Some(Input::SourceAlpha))],
                rect,
                0
            )
            .is_ok()
        );
        assert_eq!(
            SvgGraph::new([with(displacement, None)], rect, 0),
            Err(GraphError::Input)
        );
        let kernel = |columns: u32, rows: u32, values: usize, target_x| ConvolutionKernel {
            columns,
            rows,
            values: alloc::vec![1.0; values],
            target_x,
            target_y: 0,
            divisor: 1.0,
            bias: 0.0,
            edge_mode: EdgeMode::Wrap,
            preserve_alpha: false,
        };
        let convolve = |kernel| with(FilterPrimitive::ConvolveMatrix { kernel }, None);
        assert!(SvgGraph::new([convolve(kernel(128, 2, 256, 0))], rect, 0).is_ok());
        // Oversized kernels are admitted and pass their input through, as in Chrome.
        assert!(SvgGraph::new([convolve(kernel(129, 2, 258, 0))], rect, 0).is_ok());
        for invalid in [
            kernel(3, 3, 8, 0),
            kernel(3, 3, 9, 3),
            kernel(0, 3, 0, 0),
            kernel(u32::MAX, 2, 0, 0),
        ] {
            assert_eq!(
                SvgGraph::new([convolve(invalid)], rect, 0),
                Err(GraphError::Parameter)
            );
        }
        let morphology = |radius_x| {
            with(
                FilterPrimitive::Morphology {
                    operator: MorphologyOperator::Dilate,
                    radius_x,
                    radius_y: 0.0,
                },
                None,
            )
        };
        assert!(SvgGraph::new([morphology(1e30)], rect, 0).is_ok());
        assert_eq!(
            SvgGraph::new([morphology(-1.0)], rect, 0),
            Err(GraphError::Parameter)
        );
        assert_eq!(
            SvgGraph::new([morphology(f32::NAN)], rect, 0),
            Err(GraphError::Parameter)
        );
    }

    #[test]
    fn image_inputs_admit_finite_placements_and_count_their_pixels() {
        use crate::filter::image::FilterImage;
        use crate::pixmap::Pixmap;
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let image = |side: u16, destination: Rect| Node {
            primitive: FilterPrimitive::Image {
                image: FilterImage::new(Pixmap::new(side, side)),
                source: Rect::new(0.0, 0.0, f64::from(side), f64::from(side)),
                destination,
            },
            ..node(Input::SourceGraphic)
        };
        assert!(SvgGraph::new([image(4, rect)], rect, 0).is_ok());
        for invalid in [
            Rect::new(0.0, 0.0, f64::NAN, 1.0),
            Rect::new(2.0, 0.0, 1.0, 1.0),
            Rect::new(0.0, 0.0, 2e6, 1.0),
        ] {
            assert_eq!(
                SvgGraph::new([image(4, invalid)], rect, 0),
                Err(GraphError::Parameter)
            );
        }
        // Image pixels share the intermediate budget; one image used twice counts once.
        let large = image(3000, rect);
        let graph = SvgGraph::new([large.clone(), large], rect, 1).unwrap();
        assert!(graph.admit_pixels(64, 64).is_ok());
        let area = SvgGraph::max_admitted_area(2) as f64;
        let side = area.sqrt() as u16;
        assert_eq!(graph.admit_pixels(side, side), Err(GraphError::Memory));
        let two = SvgGraph::new([image(3000, rect), image(3000, rect)], rect, 1).unwrap();
        assert_eq!(two.admit_pixels(1, 1), Err(GraphError::Memory));
    }

    #[test]
    fn work_budget_counts_kernel_taps_and_octaves() {
        use crate::filter_effects::{ConvolutionKernel, EdgeMode, TurbulenceType};
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let taps = Node {
            primitive: FilterPrimitive::ConvolveMatrix {
                kernel: ConvolutionKernel {
                    columns: 16,
                    rows: 16,
                    values: alloc::vec![0.0; 256],
                    target_x: 8,
                    target_y: 8,
                    divisor: 1.0,
                    bias: 0.0,
                    edge_mode: EdgeMode::None,
                    preserve_alpha: false,
                },
            },
            ..node(Input::SourceGraphic)
        };
        let one = SvgGraph::new([taps.clone()], rect, 0).unwrap();
        assert!(one.admit_pixels(900, 900).is_ok());
        let three = SvgGraph::new([taps.clone(), taps.clone(), taps], rect, 2).unwrap();
        assert_eq!(three.admit_pixels(900, 900), Err(GraphError::Work));
        // Octaves beyond the cap cost no more than the cap.
        let noise = |num_octaves| Node {
            primitive: FilterPrimitive::Turbulence {
                base_frequency_x: 0.1,
                base_frequency_y: 0.1,
                num_octaves,
                seed: 1e30,
                stitch_tiles: false,
                turbulence_type: TurbulenceType::Turbulence,
            },
            ..node(Input::SourceGraphic)
        };
        assert_eq!(
            super::super::parameters::cost(&noise(u32::MAX).primitive),
            super::super::parameters::cost(&noise(MAX_TURBULENCE_OCTAVES).primitive)
        );
        assert!(SvgGraph::new([noise(u32::MAX)], rect, 0).is_ok());
    }
}
