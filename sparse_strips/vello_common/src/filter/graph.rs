// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Validated SVG graph plans. Parsing and resource authority belong to the caller.

use crate::filter_effects::{CompositeOperator, FilterPrimitive};
use crate::kurbo::Rect;
use alloc::vec::Vec;

/// Maximum retained primitives in one SVG graph.
pub const MAX_NODES: usize = 32;
/// Aggregate pixels retained by a graph, including source and working storage.
pub const MAX_INTERMEDIATE_PIXELS: usize = 16 * 1024 * 1024;

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
            output,
        })
    }

    /// Immutable nodes in topological order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Final filter clip in resolved graph coordinates.
    pub fn region(&self) -> Rect {
        self.region
    }
    /// The node whose pixels are the filter's output.
    pub fn output(&self) -> usize {
        self.output
    }

    /// Conservatively admit source, all results, alpha extraction and float convolution scratch
    /// surfaces. Admission happens before allocations or destination modification.
    pub fn admit_pixels(&self, width: u16, height: u16) -> Result<(), GraphError> {
        let area = usize::from(width)
            .checked_mul(usize::from(height))
            .ok_or(GraphError::Memory)?;
        let total = area
            .checked_mul(self.nodes.len() + 16)
            .ok_or(GraphError::Memory)?;
        if total > MAX_INTERMEDIATE_PIXELS {
            return Err(GraphError::Memory);
        }
        Ok(())
    }
}

fn validate_region(rect: Rect) -> Result<(), GraphError> {
    if ![rect.x0, rect.y0, rect.x1, rect.y1]
        .into_iter()
        .all(|v| v.is_finite() && v.abs() <= 1e6)
        || rect.x1 < rect.x0
        || rect.y1 < rect.y0
    {
        return Err(GraphError::Parameter);
    }
    Ok(())
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
        FilterPrimitive::Composite { .. } | FilterPrimitive::Blend { .. }
    );
    if binary != node.input2.is_some() {
        return Err(GraphError::Input);
    }
    match &node.primitive {
        FilterPrimitive::ComponentTransfer {
            red_function,
            green_function,
            blue_function,
            alpha_function,
        } => {
            for function in [red_function, green_function, blue_function, alpha_function]
                .into_iter()
                .flatten()
            {
                use crate::filter_effects::TransferFunction as T;
                let valid = |v: f32| v.is_finite() && v.abs() <= 1e6;
                let admitted = match function {
                    T::Identity => true,
                    T::Table { values } | T::Discrete { values } => {
                        values.len() <= 256 && values.iter().all(|v| valid(*v))
                    }
                    T::Linear { slope, intercept } => valid(*slope) && valid(*intercept),
                    T::Gamma {
                        amplitude,
                        exponent,
                        offset,
                    } => {
                        valid(*amplitude) && valid(*exponent) && *exponent >= 0.0 && valid(*offset)
                    }
                };
                if !admitted {
                    return Err(GraphError::Parameter);
                }
            }
        }
        FilterPrimitive::Blend { mode } => {
            use crate::peniko::Mix;
            if !matches!(
                mode,
                Mix::Normal | Mix::Multiply | Mix::Screen | Mix::Darken | Mix::Lighten
            ) {
                return Err(GraphError::Unsupported);
            }
        }
        FilterPrimitive::Composite { operator } => {
            if let CompositeOperator::Arithmetic { k1, k2, k3, k4 } = operator
                && ![*k1, *k2, *k3, *k4]
                    .into_iter()
                    .all(|v| v.is_finite() && v.abs() <= 1e6)
            {
                return Err(GraphError::Parameter);
            }
        }
        primitive if super::parameters::valid_unary_parameters(primitive) => {}
        _ => return Err(GraphError::Unsupported),
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
}
