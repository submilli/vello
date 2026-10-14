// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Convert expanded strokes to disjoint nonzero-filled trapezoids.
//!
//! The strip rasterizer integrates signed winding before clamping coverage.
//! Kurbo's inner joins can have winding two: their union must be taken before
//! this integration. A horizontal sweep applies the nonzero rule to intervals
//! between crossings, then emits each covered region exactly once.

use crate::flatten::{self, FlattenCtx, Line, Point};
use crate::geometry::RectU16;
use crate::geometry_budget::{GeometryLimit, StrokeBudget};
use crate::kurbo::{Affine, BezPath, Stroke, StrokeCtx};
use alloc::vec::Vec;
use fearless_simd::Level;

const MAX_PAIRS: usize = 1_048_576;
const MAX_EVENTS: usize = 16_384;
const MAX_SWEEP_WORK: usize = 16_777_216;
const MAX_OUTPUT_LINES: usize = 4_194_304;
const SURFACE_END: f64 = u16::MAX as f64;

/// Work charged cumulatively by the public geometry admission boundary.
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct Cost {
    pairs: usize,
    sweep: usize,
    output: usize,
}

impl Cost {
    pub(crate) fn add(&mut self, other: Self) -> Result<(), GeometryLimit> {
        self.pairs = self.pairs.checked_add(other.pairs).ok_or(GeometryLimit)?;
        self.sweep = self.sweep.checked_add(other.sweep).ok_or(GeometryLimit)?;
        self.output = self.output.checked_add(other.output).ok_or(GeometryLimit)?;
        self.check()
    }

    fn check(self) -> Result<(), GeometryLimit> {
        if self.pairs > MAX_PAIRS || self.sweep > MAX_SWEEP_WORK || self.output > MAX_OUTPUT_LINES {
            return Err(GeometryLimit);
        }
        Ok(())
    }
}

/// Run the same bounded preparation before the caller records any paint.
pub(crate) fn admit(
    path: &BezPath,
    style: &Stroke,
    transform: Affine,
    budget: &mut Cost,
) -> Result<(), GeometryLimit> {
    // These outlines already have disjoint inner and outer boundaries.
    if crate::stroke_rect::rectangle(path.elements()).is_some() && style.dash_pattern.is_empty() {
        return Ok(());
    }
    prepare(
        Level::new(),
        path,
        style,
        transform,
        &mut Vec::new(),
        &mut FlattenCtx::default(),
        &mut StrokeCtx::default(),
        budget,
    )
    .map(|_| ())
}

pub(crate) fn prepare(
    level: Level,
    path: &BezPath,
    style: &Stroke,
    transform: Affine,
    lines: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    stroke_ctx: &mut StrokeCtx,
    budget: &mut Cost,
) -> Result<Cost, GeometryLimit> {
    StrokeBudget::default().charge_expansion(path, style, transform)?;
    let tolerance = flatten::TOL / transform.spectral_norm().max(1.0);
    flatten::expand_stroke(path, style, tolerance, stroke_ctx);
    // Admission and rendering use one canonical cull region. A smaller clip
    // can otherwise change curve subdivision and intersection-work estimates.
    flatten::fill(
        level,
        stroke_ctx.output(),
        transform,
        lines,
        flatten_ctx,
        RectU16::new(0, 0, u16::MAX, u16::MAX),
    );
    normalize(lines, budget)
}

#[derive(Clone, Copy)]
struct Edge {
    top: (f64, f64),
    bottom: (f64, f64),
    slope: f64,
    winding: i32,
}

impl Edge {
    fn from_line(line: Line) -> Result<Option<Self>, GeometryLimit> {
        let a = (f64::from(line.p0.x), f64::from(line.p0.y));
        let b = (f64::from(line.p1.x), f64::from(line.p1.y));
        if ![a.0, a.1, b.0, b.1].iter().all(|v| v.is_finite()) {
            return Err(GeometryLimit);
        }
        if a.1 == b.1 {
            return Ok(None);
        }
        let (top, bottom, winding) = if a.1 < b.1 { (a, b, -1) } else { (b, a, 1) };
        if bottom.1 <= 0.0 || top.1 >= SURFACE_END {
            return Ok(None);
        }
        Ok(Some(Self {
            top,
            bottom,
            slope: (bottom.0 - top.0) / (bottom.1 - top.1),
            winding,
        }))
    }

    fn x(self, y: f64) -> f64 {
        self.top.0 + (y - self.top.1) * self.slope
    }

    fn intersection_y(self, other: Self) -> Option<f64> {
        let top = self.top.1.max(other.top.1).max(0.0);
        let bottom = self.bottom.1.min(other.bottom.1).min(SURFACE_END);
        let difference = self.slope - other.slope;
        if top >= bottom || difference == 0.0 {
            return None;
        }
        let y = top + (other.x(top) - self.x(top)) / difference;
        (y > top && y < bottom).then_some(y)
    }
}

fn normalize(lines: &mut Vec<Line>, budget: &mut Cost) -> Result<Cost, GeometryLimit> {
    let mut edges = Vec::new();
    for line in lines.iter().copied() {
        if let Some(edge) = Edge::from_line(line)? {
            edges.push(edge);
        }
    }
    let n = edges.len();
    let pairs = n.checked_mul(n.saturating_sub(1)).ok_or(GeometryLimit)? / 2;
    let mut cost = Cost {
        pairs,
        ..Cost::default()
    };
    budget.add(cost)?;
    let events = events(&edges)?;
    let slabs = events.len().saturating_sub(1);
    let sort_steps = usize::BITS as usize - n.max(1).leading_zeros() as usize;
    cost.sweep = slabs
        .checked_mul(n)
        .and_then(|v| v.checked_mul(sort_steps + 8))
        .ok_or(GeometryLimit)?;
    // Include side edges and the exposed horizontal differences between
    // adjacent slabs. Internal slab boundaries are removed before tiling.
    cost.output = slabs
        .checked_add(1)
        .ok_or(GeometryLimit)?
        .checked_mul(n + 1)
        .and_then(|v| v.checked_mul(6))
        .ok_or(GeometryLimit)?;
    budget.add(Cost { pairs: 0, ..cost })?;
    let mut result = Vec::new();
    let mut active = Vec::new();
    let mut boundaries = Boundaries::default();
    let mut previous_bottom = Vec::new();
    for slab in events.windows(2) {
        boundaries.top.clear();
        boundaries.bottom.clear();
        sweep_slab(
            &edges,
            slab[0],
            slab[1],
            &mut active,
            &mut boundaries,
            &mut result,
        );
        emit_difference(
            &boundaries.top,
            &previous_bottom,
            slab[0],
            false,
            &mut result,
        );
        emit_difference(
            &previous_bottom,
            &boundaries.top,
            slab[0],
            true,
            &mut result,
        );
        core::mem::swap(&mut previous_bottom, &mut boundaries.bottom);
    }
    if let Some(y) = events.last() {
        emit_difference(&previous_bottom, &[], *y, true, &mut result);
    }
    *lines = result;
    Ok(cost)
}

fn events(edges: &[Edge]) -> Result<Vec<f64>, GeometryLimit> {
    let mut events = Vec::new();
    for (index, edge) in edges.iter().enumerate() {
        events.push(edge.top.1.max(0.0));
        events.push(edge.bottom.1.min(SURFACE_END));
        if events.len() > MAX_EVENTS {
            return Err(GeometryLimit);
        }
        for other in &edges[index + 1..] {
            if let Some(y) = edge.intersection_y(*other) {
                events.push(y);
                if events.len() > MAX_EVENTS {
                    return Err(GeometryLimit);
                }
            }
        }
    }
    events.sort_unstable_by(f64::total_cmp);
    events.dedup();
    Ok(events)
}

#[derive(Default)]
struct Boundaries {
    top: Vec<(f64, f64)>,
    bottom: Vec<(f64, f64)>,
}

fn sweep_slab(
    edges: &[Edge],
    top: f64,
    bottom: f64,
    active: &mut Vec<(f64, usize)>,
    boundaries: &mut Boundaries,
    output: &mut Vec<Line>,
) {
    let midpoint = top + 0.5 * (bottom - top);
    active.clear();
    for (index, edge) in edges.iter().enumerate() {
        if edge.top.1 <= midpoint && edge.bottom.1 > midpoint {
            active.push((edge.x(midpoint), index));
        }
    }
    active.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| edges[a.1].slope.total_cmp(&edges[b.1].slope))
    });
    let mut winding = 0;
    let mut left = None;
    let mut index = 0;
    while index < active.len() {
        let (x, edge_index) = active[index];
        let previous = winding;
        // Coincident contours change winding together. They must not create
        // a spurious span merely because sorting chooses one contour first.
        while index < active.len() && active[index].0 == x {
            winding += edges[active[index].1].winding;
            index += 1;
        }
        if previous == 0 && winding != 0 {
            left = Some(edge_index);
        } else if previous != 0
            && winding == 0
            && let Some(left) = left.take()
        {
            emit_interval(
                edges[left],
                Some(edges[edge_index]),
                top,
                bottom,
                boundaries,
                output,
            );
        }
    }
    // The flattener culls boundaries to the right of the largest surface.
    if let Some(left) = left {
        emit_interval(edges[left], None, top, bottom, boundaries, output);
    }
}

fn emit_interval(
    left: Edge,
    right: Option<Edge>,
    top: f64,
    bottom: f64,
    boundaries: &mut Boundaries,
    output: &mut Vec<Line>,
) {
    push_interval(
        &mut boundaries.top,
        left.x(top),
        right.map_or(SURFACE_END, |edge| edge.x(top)),
    );
    push_interval(
        &mut boundaries.bottom,
        left.x(bottom),
        right.map_or(SURFACE_END, |edge| edge.x(bottom)),
    );
    let mut polygon = Vec::from([
        (left.x(top), top),
        (right.map_or(SURFACE_END, |edge| edge.x(top)), top),
        (right.map_or(SURFACE_END, |edge| edge.x(bottom)), bottom),
        (left.x(bottom), bottom),
    ]);
    // Clip geometry, rather than independently clamping its endpoints: an
    // offset can cross the surface boundary inside this slab.
    for (boundary, keep_right) in [(0.0, true), (SURFACE_END, false)] {
        let mut clipped = Vec::new();
        let mut previous = polygon.last().copied();
        for current in polygon.iter().copied() {
            if let Some(previous) = previous {
                let inside = |p: (f64, f64)| (p.0 >= boundary) == keep_right;
                if inside(previous) != inside(current) {
                    let t = (boundary - previous.0) / (current.0 - previous.0);
                    clipped.push((boundary, previous.1 + t * (current.1 - previous.1)));
                }
                if inside(current) {
                    clipped.push(current);
                }
            }
            previous = Some(current);
        }
        polygon = clipped;
    }
    let Some(mut previous) = polygon.last().copied() else {
        return;
    };
    for current in polygon {
        let p0 = Point::new(previous.0 as f32, previous.1 as f32);
        let p1 = Point::new(current.0 as f32, current.1 as f32);
        // Exposed horizontal boundaries are emitted separately after adjacent
        // slabs are compared. Their shared interior boundaries must cancel.
        if p0.y != p1.y {
            output.push(Line::new(p0, p1));
        }
        previous = current;
    }
}

fn push_interval(intervals: &mut Vec<(f64, f64)>, left: f64, right: f64) {
    let left = left.max(0.0);
    let right = right.min(SURFACE_END);
    if left >= right {
        return;
    }
    if let Some(last) = intervals.last_mut()
        && left <= last.1
    {
        last.1 = last.1.max(right);
    } else {
        intervals.push((left, right));
    }
}

/// Emit the part of each source interval not covered by the sorted cut list.
/// Horizontal exterior edges keep fractional coverage alive across sparse tile
/// gaps; emitting every slab edge instead would multiply long horizontal work.
fn emit_difference(
    source: &[(f64, f64)],
    cuts: &[(f64, f64)],
    y: f64,
    reverse: bool,
    output: &mut Vec<Line>,
) {
    let mut cut_index = 0;
    for &(left, right) in source {
        let mut cursor = left;
        while cut_index < cuts.len() && cuts[cut_index].1 <= cursor {
            cut_index += 1;
        }
        let mut index = cut_index;
        while index < cuts.len() && cuts[index].0 < right {
            let (cut_left, cut_right) = cuts[index];
            emit_horizontal(cursor, right.min(cut_left), y, reverse, output);
            cursor = cursor.max(cut_right);
            if cursor >= right {
                break;
            }
            index += 1;
        }
        emit_horizontal(cursor, right, y, reverse, output);
        cut_index = index;
    }
}

fn emit_horizontal(left: f64, right: f64, y: f64, reverse: bool, output: &mut Vec<Line>) {
    if left >= right {
        return;
    }
    let a = Point::new(left as f32, y as f32);
    let b = Point::new(right as f32, y as f32);
    if a != b {
        output.push(if reverse {
            Line::new(b, a)
        } else {
            Line::new(a, b)
        });
    }
}

#[cfg(test)]
mod tests;
