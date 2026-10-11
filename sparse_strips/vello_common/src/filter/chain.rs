// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The layers Chrome (Skia) computes a sequential chain of blurs over.
//!
//! Skia saves a filtered draw into a layer bounded by its content within what
//! the filter's output reads (`SkCanvas` layer bounds from the filter's input
//! bounds), and lays each blur's result over its input grown by the kernel,
//! within what later stages read; a large blur's rescale centers on its input's
//! layer. Only chains built by [`Filter::from_chain`] (Canvas CSS filters) are
//! planned. Single primitives keep their padded layer and per-primitive path,
//! so other renderers' scenes are unchanged.
//! See: <https://skia.googlesource.com/skia/+/main/src/core/SkImageFilterTypes.cpp>

use super::gaussian_blur::transform_blur_params;
use crate::filter_effects::{Filter, FilterPrimitive};
use crate::kurbo::{Affine, Rect};
use alloc::vec::Vec;

/// One primitive of a planned chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChainStep<'a> {
    /// A Gaussian blur by a device-space deviation, rescaled about `layer`.
    Blur {
        /// The deviation after the layer transform.
        std_deviation: f32,
        /// The pixels the input may hold, in the layer's coordinates.
        layer: Rect,
    },
    /// A color matrix that maps transparent input to transparent output.
    ColorMatrix(&'a [f32; 20]),
}

/// The steps of `filter` over its layer, or `None` unless it is a sequential
/// chain of blurs and color matrices with a zero alpha offset.
///
/// `content` is what the layer's drawing covers and `visible` what its parent
/// shows, both in the layer's coordinates. Reaches are the primitives'
/// expansions under the transform's linear part, as the layer's padding uses;
/// Canvas filters run untransformed. This is a sequential form of
/// [`super::graph::SvgGraph::reach`], adding the content's forward pass.
pub fn plan<'a>(
    filter: &'a Filter,
    transform: &Affine,
    content: Rect,
    visible: Rect,
) -> Option<Vec<ChainStep<'a>>> {
    let graph = &filter.graph;
    if !graph.is_sequential() {
        return None;
    }
    // Expansions are offsets, so only the linear part of the transform applies.
    let [a, b, c, d, ..] = transform.as_coeffs();
    let linear = Affine::new([a, b, c, d, 0.0, 0.0]);
    let mut steps = Vec::with_capacity(graph.primitives.len());
    let mut reaches = Vec::with_capacity(graph.primitives.len());
    for primitive in &graph.primitives {
        let (step, reach) = match primitive {
            FilterPrimitive::GaussianBlur { std_deviation, .. } => (
                ChainStep::Blur {
                    std_deviation: transform_blur_params(*std_deviation, transform),
                    layer: Rect::ZERO,
                },
                linear.transform_rect_bbox(primitive.filter_expansion()),
            ),
            FilterPrimitive::ColorMatrix { matrix } if matrix[19] == 0.0 => {
                (ChainStep::ColorMatrix(matrix), Rect::ZERO)
            }
            _ => return None,
        };
        steps.push(step);
        reaches.push(reach);
    }
    // What each primitive's input must cover, from the parent back to the content.
    let mut reads = Vec::with_capacity(reaches.len());
    let mut need = visible;
    for reach in reaches.iter().rev() {
        need = grow(need, *reach);
        reads.push(need);
    }
    reads.reverse();
    let mut layer = content;
    for ((step, reach), read) in steps.iter_mut().zip(&reaches).zip(&reads) {
        layer = layer.intersect(*read);
        if let ChainStep::Blur { layer: input, .. } = step {
            *input = layer;
        }
        // Transparent input stays transparent; a blur grows only what it holds.
        if layer.area() > 0.0 {
            layer = grow(layer, *reach);
        }
    }
    Some(steps)
}

/// The pixels a planned chain computes over: what the parent shows and every
/// blur's input layer (a color matrix keeps transparent pixels transparent, so
/// it needs none). Outside them inputs are transparent and outputs unseen.
pub fn extent(steps: &[ChainStep<'_>], visible: Rect) -> Rect {
    steps
        .iter()
        .filter_map(|step| match step {
            ChainStep::Blur { layer, .. } => Some(*layer),
            ChainStep::ColorMatrix(_) => None,
        })
        // Disjoint intersections are empty but still positioned; union would grow to them.
        .filter(|layer| layer.area() > 0.0)
        .fold(visible, |extent, layer| extent.union(layer))
}

/// `rect` grown by an expansion whose minimum corner is nonpositive.
fn grow(rect: Rect, expansion: Rect) -> Rect {
    Rect::new(
        rect.x0 + expansion.x0,
        rect.y0 + expansion.y0,
        rect.x1 + expansion.x1,
        rect.y1 + expansion.y1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::{EdgeMode, FilterFunction, matrices};

    const CONTENT: Rect = Rect::new(100.0, 100.0, 200.0, 200.0);
    const VISIBLE: Rect = Rect::new(50.0, 50.0, 1050.0, 1050.0);

    fn chain(functions: impl IntoIterator<Item = FilterFunction>) -> Filter {
        let primitives = functions
            .into_iter()
            .flat_map(|function| Filter::from_function(function).graph.primitives.clone());
        Filter::from_chain(primitives).unwrap()
    }

    fn blur(radius: f32) -> FilterFunction {
        FilterFunction::Blur { radius }
    }

    /// The input layers of the chain's blurs.
    fn layers(filter: &Filter, transform: Affine, content: Rect) -> Vec<Rect> {
        plan(filter, &transform, content, VISIBLE)
            .unwrap()
            .into_iter()
            .filter_map(|step| match step {
                ChainStep::Blur { layer, .. } => Some(layer),
                ChainStep::ColorMatrix(_) => None,
            })
            .collect()
    }

    #[test]
    fn a_blur_reads_its_content_and_computes_what_the_parent_shows() {
        let one = chain([blur(100.0)]);
        assert_eq!(layers(&one, Affine::IDENTITY, CONTENT), [CONTENT]);
        let steps = plan(&one, &Affine::IDENTITY, CONTENT, VISIBLE).unwrap();
        assert_eq!(extent(&steps, VISIBLE), VISIBLE);
    }

    #[test]
    fn content_beyond_what_the_output_reads_is_cut() {
        let far = Rect::new(-5000.0, 0.0, 0.0, 10.0);
        let one = chain([blur(100.0)]);
        assert_eq!(
            layers(&one, Affine::IDENTITY, far),
            [Rect::new(-250.0, 0.0, 0.0, 10.0)]
        );
        // Scaled layers grow by the scaled deviation, which the step carries.
        assert_eq!(
            layers(&one, Affine::scale(2.0), far),
            [Rect::new(-550.0, 0.0, 0.0, 10.0)]
        );
        let steps = plan(&one, &Affine::scale(2.0), far, VISIBLE).unwrap();
        assert!(
            matches!(steps[0], ChainStep::Blur { std_deviation, .. } if std_deviation == 200.0)
        );
    }

    #[test]
    fn later_blurs_read_earlier_results_grown_by_their_clamped_kernels() {
        let two = chain([
            blur(1000.0),
            FilterFunction::Opacity { amount: 0.5 },
            blur(10.0),
        ]);
        let second = Rect::new(20.0, 20.0, 1080.0, 1080.0);
        assert_eq!(layers(&two, Affine::IDENTITY, CONTENT), [CONTENT, second]);
        // Empty content stays empty rather than growing by the kernel.
        let unseen = Rect::new(5000.0, 0.0, 5010.0, 10.0);
        let none = layers(&two, Affine::IDENTITY, unseen);
        assert!(none.iter().all(|layer| layer.area() == 0.0));
        let steps = plan(&two, &Affine::IDENTITY, unseen, VISIBLE);
        assert_eq!(extent(&steps.unwrap(), VISIBLE), VISIBLE);
    }

    #[test]
    fn chains_that_move_or_create_alpha_are_not_planned() {
        let shadow = FilterPrimitive::DropShadow {
            dx: 1.0,
            dy: 1.0,
            std_deviation: 1.0,
            color: crate::color::palette::css::RED,
            edge_mode: EdgeMode::None,
        };
        let mut alpha = matrices::IDENTITY;
        alpha[19] = 0.5;
        let blurred = Filter::from_function(blur(1.0)).graph.primitives[0].clone();
        for other in [shadow, FilterPrimitive::ColorMatrix { matrix: alpha }] {
            let filter = Filter::from_chain([blurred.clone(), other]).unwrap();
            assert_eq!(plan(&filter, &Affine::IDENTITY, CONTENT, VISIBLE), None);
        }
        let single = Filter::from_primitive(blurred);
        assert_eq!(plan(&single, &Affine::IDENTITY, CONTENT, VISIBLE), None);
    }
}
