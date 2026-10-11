// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Where an SVG graph's results can be nonzero, as Skia bounds image filter
//! outputs (`SkImageFilter_Base::getOutputLayerBounds`), and the tiles that
//! Chrome cannot draw because of them.

use super::graph::{Input, Node, SvgGraph};
use super::offset::whole_pixels;
use super::reach::outward;
use crate::filter_effects::{CompositeOperator, EdgeMode, FilterPrimitive, TransferFunction};
use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use alloc::vec::Vec;

/// The largest single allocation Chrome's allocator makes (`PartitionAlloc`'s
/// direct-map limit, 2 GiB less a 2 MiB super page).
const CHROME_MAX_ALLOCATION: f64 = 2_145_386_496.0;

/// How a tile draws what is read of it.
pub(super) struct Tiling {
    /// The input's bounds, which the tile repeats.
    pub period: Rect,
    /// What the tile reads of its input's drawn content: on each axis, `need`
    /// mapped into the one period it shows there, or the whole span where
    /// `need` covers more than one period.
    pub reads: Rect,
}

impl SvgGraph {
    /// The period `node`, a tile, repeats to produce `need`: its input's
    /// bounds. `None` when Chrome draws nothing.
    pub fn tile_period(&self, node: &Node, need: Rect) -> Option<Rect> {
        self.tiling(node, need).map(|tiling| tiling.period)
    }

    /// How `node`, a tile, draws `need`, as Skia's repeating crop does
    /// (`FilterResult::applyCrop`). Skia draws the input's content within the
    /// period into a surface with a pixel of padding on each side, and where
    /// `need` spans more than one period and the content leaves part of the
    /// period transparent, also draws the whole period into a surface. Chrome
    /// draws nothing when either surface's 32-bit pixels exceed its largest
    /// allocation. Measured in Chrome 154: a 26,812 × 20,000 flood tiles and a
    /// 26,814 × 20,000 one does not; a small flood in an 800,000 × 600 period
    /// wrapping across the canvas tiles, in a 900,000 × 600 one it does not.
    pub(super) fn tiling(&self, node: &Node, need: Rect) -> Option<Tiling> {
        let period = self.input_bounds(node.input);
        if period.width() <= 0.0 || period.height() <= 0.0 {
            return None;
        }
        let drawn = period.intersect(self.content(node.input, &self.contents()));
        if !allocatable((drawn.width() + 2.0) * (drawn.height() + 2.0)) {
            return None;
        }
        let x = shown((need.x0, need.x1), (period.x0, period.x1));
        let y = shown((need.y0, need.y1), (period.y0, period.y1));
        let wraps = x.is_none() || y.is_none();
        if wraps && drawn != period && !allocatable(period.area()) {
            return None;
        }
        // Each axis reads the span it shows, or all it draws where it wraps;
        // nothing beyond what it draws.
        let (x0, x1) = x.unwrap_or((drawn.x0, drawn.x1));
        let (y0, y1) = y.unwrap_or((drawn.y0, drawn.y1));
        Some(Tiling {
            period,
            reads: Rect::new(x0, y0, x1, y1).intersect(drawn),
        })
    }

    /// Each node's content, in order. Recomputed per tile: graphs hold at
    /// most [`super::graph::MAX_NODES`] nodes.
    fn contents(&self) -> Vec<Rect> {
        let mut contents = Vec::with_capacity(self.nodes().len());
        for node in self.nodes() {
            let input = self.content(node.input, &contents);
            let input2 = node.input2.map(|input| self.content(input, &contents));
            let region = outward(node.region.intersect(self.region()));
            contents.push(produced(node, input, input2, region).intersect(region));
        }
        contents
    }

    /// Where `input` can be nonzero, given the earlier nodes' `contents`. The
    /// source graphic has no bounds of its own (in Chrome a tile of
    /// `SourceAlpha` offset into a huge subregion draws nothing), so it may
    /// fill the graph's region.
    fn content(&self, input: Input, contents: &[Rect]) -> Rect {
        match input {
            Input::SourceGraphic | Input::SourceAlpha => outward(self.region()),
            Input::Result(index) => contents[index],
        }
    }
}

/// Where `node` can draw before clipping to its `region`, from its inputs'
/// content.
fn produced(node: &Node, input: Rect, input2: Option<Rect>, region: Rect) -> Rect {
    match &node.primitive {
        FilterPrimitive::Flood { color } if color.components[3] <= 0.0 => Rect::ZERO,
        // Generators, tiles and lighting draw across their whole region.
        FilterPrimitive::Flood { .. }
        | FilterPrimitive::Turbulence { .. }
        | FilterPrimitive::Image { .. }
        | FilterPrimitive::Tile
        | FilterPrimitive::DiffuseLighting { .. }
        | FilterPrimitive::SpecularLighting { .. } => region,
        // Color functions that lift transparent alpha fill their region too.
        FilterPrimitive::ColorMatrix { matrix } if matrix[19] > 0.0 => region,
        FilterPrimitive::ComponentTransfer { alpha_function, .. }
            if alpha_function.as_ref().is_some_and(fills_transparent) =>
        {
            region
        }
        FilterPrimitive::ColorMatrix { .. } | FilterPrimitive::ComponentTransfer { .. } => input,
        FilterPrimitive::Composite {
            operator: CompositeOperator::Arithmetic { k4, .. },
        } if *k4 > 0.0 => region,
        // Skia bounds blends by both inputs whatever the operator: in Chrome a
        // small flood composited `in` source paint does not bound the tile.
        FilterPrimitive::Composite { .. } | FilterPrimitive::Blend { .. } => {
            input2.map_or(input, |other| union_drawn(input, other))
        }
        FilterPrimitive::Offset { .. } if input.area() <= 0.0 => Rect::ZERO,
        FilterPrimitive::Offset { dx, dy } => input + whole_pixels(*dx, *dy),
        // A bias brightens transparent pixels, and edge modes extend the input
        // across the crop.
        FilterPrimitive::ConvolveMatrix { kernel }
            if kernel.bias > 0.0 || kernel.edge_mode != EdgeMode::None =>
        {
            region
        }
        // Neighborhoods spread their input by at most their reach.
        FilterPrimitive::GaussianBlur { .. }
        | FilterPrimitive::AxisGaussianBlur { .. }
        | FilterPrimitive::DropShadow { .. }
        | FilterPrimitive::DropShadowOnly { .. }
        | FilterPrimitive::AxisDropShadow { .. }
        | FilterPrimitive::Morphology { .. }
        | FilterPrimitive::ConvolveMatrix { .. }
        | FilterPrimitive::DisplacementMap { .. } => {
            spread(input, node.primitive.filter_expansion())
        }
    }
}

/// Whether Chrome can allocate a surface of `pixels` 32-bit pixels.
fn allocatable(pixels: f64) -> bool {
    pixels * 4.0 <= CHROME_MAX_ALLOCATION
}

/// `need`'s span within the period it lies in, mapped into `period`, when it
/// shows only one period on this axis (Skia's `periodic_axis_transform`).
fn shown(need: (f64, f64), period: (f64, f64)) -> Option<(f64, f64)> {
    let size = period.1 - period.0;
    let first = ((need.0 - period.0) / size).floor();
    let last = ((need.1 - period.0) / size).ceil();
    (last - first <= 1.0).then_some((need.0 - first * size, need.1 - first * size))
}

/// `input` grown by an expansion; nothing grows from no content.
fn spread(input: Rect, reach: Rect) -> Rect {
    if input.area() <= 0.0 {
        return Rect::ZERO;
    }
    outward(Rect::new(
        input.x0 + reach.x0.min(0.0),
        input.y0 + reach.y0.min(0.0),
        input.x1 + reach.x1.max(0.0),
        input.y1 + reach.y1.max(0.0),
    ))
}

/// Whether a transfer function maps transparent alpha above zero.
fn fills_transparent(function: &TransferFunction) -> bool {
    match function {
        TransferFunction::Identity => false,
        TransferFunction::Table { values } | TransferFunction::Discrete { values } => {
            values.first().is_some_and(|first| *first > 0.0)
        }
        TransferFunction::Linear { intercept, .. } => *intercept > 0.0,
        TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => *offset > 0.0 || (*exponent <= 0.0 && *amplitude > 0.0),
    }
}

/// The union of two contents, where an empty one adds nothing.
fn union_drawn(a: Rect, b: Rect) -> Rect {
    match (a.area() > 0.0, b.area() > 0.0) {
        (true, true) => a.union(b),
        (true, false) => a,
        _ => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::palette::css::BLUE;
    use crate::filter::graph::ColorSpace;
    use alloc::vec;

    const HUGE: Rect = Rect::new(-1e6, -1e6, 1e6, 1e6);

    fn node(primitive: FilterPrimitive, input: Input, region: Rect) -> Node {
        Node {
            primitive,
            input,
            input2: None,
            region,
            color_space: ColorSpace::Srgb,
        }
    }

    /// A tile of `nodes`' last result over a 300×150 canvas; its period if
    /// Chrome draws it.
    fn period(mut nodes: Vec<Node>) -> Option<Rect> {
        let input = Input::Result(nodes.len() - 1);
        nodes.push(node(FilterPrimitive::Tile, input, HUGE));
        let output = nodes.len() - 1;
        let graph = SvgGraph::new(nodes, HUGE, output).unwrap();
        graph.tile_period(&graph.nodes()[output], Rect::new(0.0, 0.0, 300.0, 150.0))
    }

    fn flood(region: Rect) -> Node {
        node(
            FilterPrimitive::Flood { color: BLUE },
            Input::SourceGraphic,
            region,
        )
    }

    #[test]
    fn chrome_draws_no_tile_whose_content_exceeds_its_largest_allocation() {
        // Chrome 154 (`frames.js`): (w + 2)(h + 2) 32-bit pixels of content.
        let centered = |w: f64, h: f64| Rect::from_center_size((150.0, 75.0), (w, h));
        let within = centered(26_812.0, 20_000.0);
        assert_eq!(period(vec![flood(within)]), Some(within));
        assert_eq!(period(vec![flood(centered(26_814.0, 20_000.0))]), None);
        let tall = |h: f64| Rect::new(-50.0, -999_000.0, 350.0, -999_000.0 + h);
        assert!(period(vec![flood(tall(1_334_193.0))]).is_some());
        assert_eq!(period(vec![flood(tall(1_334_194.0))]), None);
    }

    #[test]
    fn tile_limits_count_drawn_content_not_the_period() {
        let huge_period = Rect::from_center_size((150.0, 75.0), (1e6, 600.0));
        let small = flood(Rect::new(140.0, 60.0, 160.0, 90.0));
        let moved = node(
            FilterPrimitive::Offset { dx: 0.0, dy: 0.0 },
            Input::Result(0),
            huge_period,
        );
        assert_eq!(period(vec![small, moved]), Some(huge_period));
        // Canvas source paint has no bounds of its own.
        let alpha = node(
            FilterPrimitive::Offset { dx: 0.0, dy: 0.0 },
            Input::SourceAlpha,
            huge_period,
        );
        assert_eq!(period(vec![alpha]), None);
        // A transparent flood draws nothing, so even a huge period tiles.
        let clear = node(
            FilterPrimitive::Flood {
                color: crate::color::palette::css::TRANSPARENT,
            },
            Input::SourceGraphic,
            huge_period,
        );
        assert_eq!(period(vec![clear]), Some(huge_period));
    }

    #[test]
    fn composites_bound_their_content_by_both_inputs() {
        // Chrome 154 (`frames.js` `tileInSourceBeyondLimit`).
        let huge_period = Rect::from_center_size((150.0, 75.0), (1e6, 600.0));
        let small = flood(Rect::new(140.0, 60.0, 160.0, 90.0));
        let mut composite = node(
            FilterPrimitive::Composite {
                operator: CompositeOperator::In,
            },
            Input::Result(0),
            huge_period,
        );
        composite.input2 = Some(Input::SourceAlpha);
        assert_eq!(period(vec![small, composite]), None);
    }

    #[test]
    fn wrapping_tiles_with_transparent_periods_allocate_the_period() {
        // Chrome 154: a 20×30 flood at the end of a period starting at x = 100
        // shows on the canvas past its wrap.
        let wrapping = |width: f64| {
            let period = Rect::new(100.0, -225.0, 100.0 + width, 375.0);
            let small = flood(Rect::new(period.x1 - 60.0, 60.0, period.x1 - 40.0, 90.0));
            let moved = node(
                FilterPrimitive::Offset { dx: 0.0, dy: 0.0 },
                Input::Result(0),
                period,
            );
            period_reads(vec![small, moved])
        };
        let tiled = wrapping(800_000.0).unwrap();
        // Only the drawn content is read, not the whole period.
        assert_eq!(tiled.reads, Rect::new(800_040.0, 60.0, 800_060.0, 90.0));
        assert!(wrapping(900_000.0).is_none());
    }

    #[test]
    fn tiles_read_the_span_each_axis_shows() {
        let read = |period: Rect| period_reads(vec![flood(period)]).map(|tiling| tiling.reads);
        // One period on both axes: the canvas mapped into it, edges included.
        let shifted = Rect::new(300.0, 150.0, 600.0, 300.0);
        assert_eq!(read(shifted), Some(shifted));
        let before = Rect::new(-900.0, -450.0, -600.0, -300.0);
        assert_eq!(read(before), Some(before));
        // Wrapping on x only: x reads what the period draws, y what it shows,
        // so a tall period costs no more than the canvas.
        let tall = Rect::new(100.0, -40_000.0, 400.0, 40_000.0);
        assert_eq!(read(tall), Some(Rect::new(100.0, 0.0, 400.0, 150.0)));
    }

    #[test]
    fn shown_maps_a_span_into_its_single_period() {
        assert_eq!(shown((10.0, 20.0), (0.0, 10.0)), Some((0.0, 10.0)));
        assert_eq!(shown((-10.0, 0.0), (0.0, 10.0)), Some((0.0, 10.0)));
        assert_eq!(shown((-3.0, 2.0), (0.0, 10.0)), None);
    }

    fn period_reads(mut nodes: Vec<Node>) -> Option<Tiling> {
        let input = Input::Result(nodes.len() - 1);
        nodes.push(node(FilterPrimitive::Tile, input, HUGE));
        let output = nodes.len() - 1;
        let graph = SvgGraph::new(nodes, HUGE, output).unwrap();
        graph.tiling(&graph.nodes()[output], Rect::new(0.0, 0.0, 300.0, 150.0))
    }
}
