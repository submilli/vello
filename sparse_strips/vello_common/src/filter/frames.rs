// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Where each node of an SVG graph is computed for a raster, as Skia computes
//! each image filter over its own layer bounds rather than one shared surface.
//!
//! The caller rasters the source graphic and receives the output over one
//! raster. Each node keeps only what its consumers (and the raster) read of
//! its result, and computes over that, its own crop-dependent reads and what
//! it reads of its inputs. Offsets and tiles copy straight from their input's
//! pixels, so paint that a generator creates 100,000 pixels away costs two
//! small frames, not one raster spanning both.

use super::graph::{
    ColorSpace, FIXED_SURFACES, GraphError, Input, MAX_INTERMEDIATE_PIXELS, MAX_WORK, SvgGraph,
};
use super::reach::{outward, pixels_within, union};
use crate::filter_effects::FilterPrimitive;
use crate::kurbo::Rect;
use alloc::vec::Vec;

/// The fixed surfaces as large as the raster: the source and the output.
const RASTERS: usize = 2;

/// Where a graph's nodes are computed for one raster, in graph coordinates on
/// the pixel grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Frames {
    raster: Rect,
    nodes: Vec<Option<Frame>>,
    cost: RasterCost,
}

/// Admitted resources of one execution, for callers sharing limits across graphs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterCost {
    /// Retained and working texels, including the caller's source and output.
    pub pixels: usize,
    /// Estimated pixel operations, using the same cost as graph admission.
    pub work: u64,
}

/// Where one node is computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// The pixels its result keeps: what its consumers and the raster read.
    pub kept: Rect,
    /// The pixels it computes over: what it keeps, the positions its own
    /// crop-dependent reads test, and what it reads of its inputs.
    pub work: Rect,
    /// What its primary input holds within what the node reads of it (Skia's
    /// input layer bounds within the desired input), which a rescaled blur
    /// centers on. Empty when the node reads nothing of it.
    pub layer: Rect,
}

impl Frames {
    /// The resources admitted for this raster and destination.
    pub fn cost(&self) -> RasterCost {
        self.cost
    }

    /// The raster the source is drawn into and the output is read from.
    pub fn raster(&self) -> Rect {
        self.raster
    }

    /// Where node `index` is computed; `None` when nothing reads its result.
    pub fn node(&self, index: usize) -> Option<&Frame> {
        self.nodes.get(index).and_then(Option::as_ref)
    }

    /// The pixels `input` holds: the raster, or what an earlier node keeps.
    pub(crate) fn input(&self, input: Input) -> Option<Rect> {
        match input {
            Input::SourceGraphic | Input::SourceAlpha => Some(self.raster),
            Input::Result(index) => self.node(index).map(|frame| frame.kept),
        }
    }
}

impl SvgGraph {
    /// Where each node is computed to produce the output over `destination`
    /// within `raster`. Like Skia's desired output, the destination (what the
    /// caller reads) rather than the whole raster decides what each node
    /// computes, and so where a rescaled blur centers.
    pub(crate) fn frames(&self, raster: Rect, destination: Rect) -> Frames {
        let raster = outward(raster);
        let trace = self.trace(outward(destination).intersect(raster));
        let mut frames = Frames {
            raster,
            nodes: Vec::with_capacity(self.nodes().len()),
            cost: RasterCost::default(),
        };
        for (index, node) in self.nodes().iter().enumerate() {
            let frame = trace.needs[index].map(|need| {
                let reads = self.reads(node, need);
                let read = |input: Input, rect: Option<Rect>| {
                    let held = frames.input(input)?;
                    pixels_within(rect?, held)
                };
                let primary = read(node.input, reads.input);
                let secondary = node.input2.and_then(|input| read(input, reads.input2));
                let work = match node.primitive {
                    // Copied from their input's pixels wherever those lie.
                    FilterPrimitive::Offset { .. } | FilterPrimitive::Tile => need,
                    _ => union(union(trace.reach.nodes[index], primary), secondary).unwrap_or(need),
                };
                let layer = primary.map_or(Rect::ZERO, |read| {
                    read.intersect(outward(self.layer_region(node.input)))
                });
                Frame {
                    kept: need,
                    work,
                    layer,
                }
            });
            frames.nodes.push(frame);
        }
        frames
    }

    /// Admit computing the graph over `frames` before anything is allocated:
    /// the source and output rasters, the results each node keeps, the other
    /// fixed working surfaces (alpha extraction and working copies such as a
    /// blur's rescaled images), each as large as the largest frame computed,
    /// the image inputs the graph retains, and the per-pixel work over each
    /// node's frame.
    fn admit_frames(&self, frames: &Frames) -> Result<RasterCost, GraphError> {
        let rasters = pixels(frames.raster)?.saturating_mul(RASTERS);
        let mut widest = 0_usize;
        let mut kept = 0_usize;
        let mut work = 0_u64;
        for (node, frame) in self.nodes().iter().zip(&frames.nodes) {
            let Some(frame) = frame else {
                continue;
            };
            let area = pixels(frame.work)?;
            widest = widest.max(area);
            kept = kept.saturating_add(pixels(frame.kept)?);
            let conversion = match node.color_space {
                ColorSpace::Srgb => 0,
                ColorSpace::LinearRgb => super::parameters::LINEAR_CONVERSION_COST,
            };
            let cost = super::parameters::cost(&node.primitive) + conversion;
            work = work.saturating_add(cost.saturating_mul(area as u64));
        }
        let surfaces = widest
            .checked_mul(FIXED_SURFACES - RASTERS)
            .ok_or(GraphError::Memory)?;
        let retained = surfaces
            .saturating_add(rasters)
            .saturating_add(kept)
            .saturating_add(self.image_texels());
        if retained > MAX_INTERMEDIATE_PIXELS {
            return Err(GraphError::Memory);
        }
        if work > MAX_WORK {
            return Err(GraphError::Work);
        }
        Ok(RasterCost {
            pixels: retained,
            work,
        })
    }

    /// Admit computing the graph to produce its output over `destination`,
    /// reading the source from and writing the output to `raster`.
    pub fn admit_raster(&self, raster: Rect, destination: Rect) -> Result<Frames, GraphError> {
        let mut frames = self.frames(raster, destination);
        frames.cost = self.admit_frames(&frames)?;
        Ok(frames)
    }

    /// Everything an input may hold, standing in for Skia's layer bounds:
    /// source paint reaches beyond the source's content (a canvas drawing
    /// past its bitmap) across the graph region, and results reach their
    /// node's region. Edge modes instead extend only the source's content
    /// (`input_bounds`), so the two differ on purpose.
    fn layer_region(&self, input: Input) -> Rect {
        match input {
            Input::SourceGraphic | Input::SourceAlpha => self.region(),
            Input::Result(index) => self.nodes()[index].region.intersect(self.region()),
        }
    }
}

/// The pixels of a frame, which must fit one raster.
fn pixels(rect: Rect) -> Result<usize, GraphError> {
    let side = f64::from(u16::MAX);
    if !(rect.width() <= side && rect.height() <= side) {
        return Err(GraphError::Memory);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Both sides are whole pixels within u16."
    )]
    Ok(rect.width() as usize * rect.height() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::palette::css::BLUE;
    use crate::filter::graph::Node;
    use alloc::vec;

    const HUGE: Rect = Rect::new(-1e6, -1e6, 1e6, 1e6);
    const RASTER: Rect = Rect::new(0.0, 0.0, 300.0, 150.0);

    fn node(primitive: FilterPrimitive, input: Input) -> Node {
        Node {
            primitive,
            input,
            input2: None,
            region: HUGE,
            color_space: ColorSpace::Srgb,
        }
    }

    fn flood() -> Node {
        node(FilterPrimitive::Flood { color: BLUE }, Input::SourceGraphic)
    }

    fn offset(dx: f32, input: usize) -> Node {
        node(
            FilterPrimitive::Offset { dx, dy: 0.0 },
            Input::Result(input),
        )
    }

    fn graph(nodes: Vec<Node>) -> SvgGraph {
        let output = nodes.len() - 1;
        SvgGraph::new(nodes, HUGE, output).unwrap()
    }

    #[test]
    fn admitted_cost_reports_the_same_surfaces_and_work_as_admission() {
        let raster = Rect::new(0.0, 0.0, 4.0, 3.0);
        let graph = graph(vec![flood()]);
        let cost = graph.admit_raster(raster, raster).unwrap().cost();
        assert_eq!(cost.pixels, (FIXED_SURFACES + 1) * 12);
        assert_eq!(
            cost.work,
            super::super::parameters::cost(&graph.nodes()[0].primitive) * 12
        );
        // Destination-driven execution charges only the node's actual frame;
        // the caller's source/output still occupy the whole raster.
        let tiny = Rect::new(0.0, 0.0, 1.0, 1.0);
        let clipped = graph.admit_raster(raster, tiny).unwrap().cost();
        assert_eq!(clipped.pixels, RASTERS * 12 + FIXED_SURFACES - RASTERS + 1);
        assert_eq!(clipped.work * 12, cost.work);
    }

    #[test]
    fn far_apart_reads_compute_over_their_own_frames() {
        // Chrome 154 (`frames.js`): a flood 100,000 pixels away offset into view.
        let graph = graph(vec![flood(), offset(1e5, 0)]);
        let frames = graph.admit_raster(RASTER, RASTER).unwrap();
        let far = RASTER - crate::kurbo::Vec2::new(1e5, 0.0);
        assert_eq!(frames.node(0).map(|frame| frame.kept), Some(far));
        assert_eq!(frames.node(0).map(|frame| frame.work), Some(far));
        assert_eq!(frames.node(1).map(|frame| frame.kept), Some(RASTER));
        assert_eq!(frames.input(Input::Result(0)), Some(far));
        assert_eq!(frames.input(Input::SourceAlpha), Some(RASTER));
    }

    #[test]
    fn admission_sums_the_frames_kept() {
        // Offsets 10,000 pixels apart: each keeps a raster-sized frame, so the
        // union is vast, but their sum fits until there are too many.
        let raster = Rect::new(0.0, 0.0, 600.0, 600.0);
        let chain = |count: usize| {
            let mut nodes = vec![flood()];
            nodes.extend((0..count).map(|index| offset(1e4, index)));
            graph(nodes)
        };
        assert!(chain(19).admit_raster(raster, raster).is_ok());
        assert_eq!(
            chain(31).admit_raster(raster, raster),
            Err(GraphError::Memory)
        );
    }

    #[test]
    fn neighborhoods_work_over_what_they_read_and_keep_what_is_read() {
        let blur = node(
            FilterPrimitive::GaussianBlur {
                std_deviation: 10.0,
                edge_mode: crate::filter_effects::EdgeMode::None,
            },
            Input::Result(0),
        );
        let graph = graph(vec![flood(), blur]);
        let frames = graph.frames(RASTER, RASTER);
        let read = RASTER.inflate(36.0, 36.0);
        assert_eq!(frames.node(0).map(|frame| frame.kept), Some(read));
        let blurred = frames.node(1).unwrap();
        assert_eq!((blurred.kept, blurred.work), (RASTER, read));
        // A rescaled blur centers on what its input holds within its reads.
        assert_eq!(blurred.layer, read);
    }

    #[test]
    fn frames_wider_than_a_raster_are_not_admitted() {
        let wide = Rect::new(0.0, 0.0, 70_000.0, 1.0);
        assert_eq!(
            graph(vec![flood()]).admit_raster(wide, wide),
            Err(GraphError::Memory)
        );
    }
}
