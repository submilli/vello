// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CSS filter conversion, the bounded sequential CPU subset, and the layers
//! Chrome (Skia) computes a chain of blurs over.

use super::parameters::MAX_BLUR_DEVIATION;
use crate::filter_effects::{EdgeMode, Filter, FilterFunction, FilterPrimitive, matrices};
use crate::kurbo::{Affine, Rect};
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

pub(crate) fn primitive(function: FilterFunction) -> FilterPrimitive {
    let mut matrix = matrices::IDENTITY;
    match function {
        // Skia's blur clamps deviations; comparing keeps NaN, which admission rejects.
        FilterFunction::Blur { radius } => {
            return FilterPrimitive::GaussianBlur {
                std_deviation: if radius > MAX_BLUR_DEVIATION {
                    MAX_BLUR_DEVIATION
                } else {
                    radius
                },
                edge_mode: EdgeMode::None,
            };
        }
        FilterFunction::Brightness { amount } => rgb_linear(&mut matrix, amount, 0.0),
        FilterFunction::Contrast { amount } => {
            rgb_linear(&mut matrix, amount, (1.0 - amount) / 2.0);
        }
        FilterFunction::Invert { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            rgb_linear(&mut matrix, 1.0 - 2.0 * amount, amount);
        }
        FilterFunction::Opacity { amount } => matrix[18] = amount.clamp(0.0, 1.0),
        FilterFunction::Grayscale { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            for (output, grayscale) in matrix.iter_mut().zip(matrices::GRAYSCALE) {
                *output = *output * (1.0 - amount) + grayscale * amount;
            }
        }
        FilterFunction::Saturate { amount } => matrix = saturation(amount),
        FilterFunction::Sepia { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            for (output, sepia) in matrix.iter_mut().zip(matrices::SEPIA) {
                *output = *output * (1.0 - amount) + sepia * amount;
            }
        }
        FilterFunction::HueRotate { angle } => {
            let (s, c) = angle.to_radians().sin_cos();
            matrix = [
                0.213 + 0.787 * c - 0.213 * s,
                0.715 - 0.715 * c - 0.715 * s,
                0.072 - 0.072 * c + 0.928 * s,
                0.0,
                0.0,
                0.213 - 0.213 * c + 0.143 * s,
                0.715 + 0.285 * c + 0.140 * s,
                0.072 - 0.072 * c - 0.283 * s,
                0.0,
                0.0,
                0.213 - 0.213 * c - 0.787 * s,
                0.715 - 0.715 * c + 0.715 * s,
                0.072 + 0.928 * c + 0.072 * s,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ];
        }
    }
    FilterPrimitive::ColorMatrix { matrix }
}

fn rgb_linear(matrix: &mut [f32; 20], slope: f32, intercept: f32) {
    for row in 0..3 {
        matrix[row * 6] = slope;
        matrix[row * 5 + 4] = intercept;
    }
}

fn saturation(amount: f32) -> [f32; 20] {
    let mut matrix = matrices::IDENTITY;
    for row in 0..3 {
        for (column, luma) in [0.213, 0.715, 0.072].into_iter().enumerate() {
            matrix[row * 5 + column] =
                luma * (1.0 - amount) + if row == column { amount } else { 0.0 };
        }
    }
    matrix
}

/// Primitives the sequential CPU chain executes, with admitted parameters.
pub(crate) fn supported(primitive: &FilterPrimitive) -> bool {
    matches!(
        primitive,
        FilterPrimitive::ColorMatrix { .. }
            | FilterPrimitive::GaussianBlur { .. }
            | FilterPrimitive::DropShadow { .. }
            | FilterPrimitive::DropShadowOnly { .. }
            | FilterPrimitive::Offset { .. }
            | FilterPrimitive::Flood { .. }
    ) && super::parameters::valid(primitive)
}

/// The layer each primitive of a sequential chain of blurs reads, in the
/// filter layer's coordinates, or `None` for chains with other primitives.
///
/// Skia lays a filtered draw out over its `content` within what the output
/// reads (`SkCanvas` layer bounds from the filter's input bounds), and each
/// blur's result over its input grown by the kernel, within what later stages
/// read. The rescale of a large blur centers on its input's layer. `visible` is
/// what the layer's parent shows. Color matrices that keep transparent pixels
/// transparent keep their input's layer.
/// See: <https://skia.googlesource.com/skia/+/main/src/core/SkImageFilterTypes.cpp>
pub fn chain_layers(
    filter: &Filter,
    transform: &Affine,
    content: Rect,
    visible: Rect,
) -> Option<Vec<Rect>> {
    let graph = &filter.graph;
    if !graph.is_sequential() {
        return None;
    }
    let [a, b, c, d, ..] = transform.as_coeffs();
    let linear = Affine::new([a, b, c, d, 0.0, 0.0]);
    let mut reaches = Vec::with_capacity(graph.primitives.len());
    for primitive in &graph.primitives {
        match primitive {
            FilterPrimitive::GaussianBlur { .. } => {
                reaches.push(linear.transform_rect_bbox(primitive.filter_expansion()));
            }
            // Alpha that does not depend on transparent input keeps it transparent.
            FilterPrimitive::ColorMatrix { matrix } if matrix[19] == 0.0 => {
                reaches.push(Rect::ZERO);
            }
            _ => return None,
        }
    }
    // What each primitive's output must cover, from the parent back to the content.
    let mut reads = Vec::with_capacity(reaches.len());
    let mut need = visible;
    for reach in reaches.iter().rev() {
        need = grow(need, *reach);
        reads.push(need);
    }
    reads.reverse();
    let mut layer = content;
    let mut layers = Vec::with_capacity(reaches.len());
    for (reach, read) in reaches.iter().zip(&reads) {
        layer = layer.intersect(*read);
        layers.push(layer);
        layer = grow(layer, *reach);
    }
    Some(layers)
}

/// The pixels a sequential chain of blurs computes over: what the parent shows
/// and every primitive's input layer ([`chain_layers`]). Outside them inputs
/// are transparent and outputs unseen. `None` for other chains.
pub fn chain_extent(
    filter: &Filter,
    transform: &Affine,
    content: Rect,
    visible: Rect,
) -> Option<Rect> {
    let layers = chain_layers(filter, transform, content, visible)?;
    Some(
        layers
            .into_iter()
            .filter(|layer| layer.area() > 0.0)
            .fold(visible, |extent, layer| extent.union(layer)),
    )
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
    use crate::filter_effects::Filter;
    use crate::kurbo::{Affine, Rect};

    #[test]
    fn chain_accumulates_padding_and_rejects_unbounded_or_unsupported_input() {
        let blur = || primitive(FilterFunction::Blur { radius: 2.0 });
        let chain = Filter::from_chain([blur(), blur()]).unwrap();
        assert_eq!(
            chain.filter_expansion(&Affine::IDENTITY),
            Rect::new(-12.0, -12.0, 12.0, 12.0)
        );
        assert_eq!(
            chain.source_expansion(&Affine::IDENTITY),
            Rect::new(-12.0, -12.0, 12.0, 12.0)
        );
        assert!(Filter::from_chain(core::iter::repeat_with(blur)).is_none());
        assert!(Filter::from_chain([FilterPrimitive::Tile]).is_none());
        assert!(
            Filter::from_chain([primitive(FilterFunction::Blur { radius: f32::NAN })]).is_none()
        );
    }

    #[test]
    fn blur_deviations_clamp_to_skia_maximum() {
        let deviation = |radius| match primitive(FilterFunction::Blur { radius }) {
            FilterPrimitive::GaussianBlur { std_deviation, .. } => std_deviation,
            _ => unreachable!("blur() converts to a Gaussian blur"),
        };
        assert_eq!(deviation(531.0), 531.0);
        assert_eq!(deviation(1000.0), MAX_BLUR_DEVIATION);
        assert_eq!(deviation(f32::INFINITY), MAX_BLUR_DEVIATION);
        assert!(deviation(f32::NAN).is_nan());
        let chain = Filter::from_chain([primitive(FilterFunction::Blur { radius: 1e9 })]).unwrap();
        assert_eq!(
            chain.filter_expansion(&Affine::IDENTITY),
            Rect::new(-1596.0, -1596.0, 1596.0, 1596.0)
        );
    }

    #[test]
    fn chains_of_blurs_lay_out_like_skia() {
        let blur = |radius| primitive(FilterFunction::Blur { radius });
        let identity = Affine::IDENTITY;
        let content = Rect::new(100.0, 100.0, 200.0, 200.0);
        let visible = Rect::new(50.0, 50.0, 1050.0, 1050.0);
        // One blur reads its content and computes what the parent shows.
        let one = Filter::from_chain([blur(100.0)]).unwrap();
        assert_eq!(
            chain_layers(&one, &identity, content, visible),
            Some(alloc::vec![content])
        );
        assert_eq!(
            chain_extent(&one, &identity, content, visible),
            Some(visible)
        );
        // Content beyond what the output reads (3σ past it) is cut.
        let far = Rect::new(-5000.0, 0.0, 0.0, 10.0);
        assert_eq!(
            chain_layers(&one, &identity, far, visible),
            Some(alloc::vec![Rect::new(-250.0, 0.0, 0.0, 10.0)])
        );
        // A second blur reads the first's result grown by its clamped 3σ,
        // within what the second reads; color matrices keep their input's.
        let two = Filter::from_chain([
            blur(1000.0),
            primitive(FilterFunction::Opacity { amount: 0.5 }),
            blur(10.0),
        ])
        .unwrap();
        let layers = chain_layers(&two, &identity, content, visible).unwrap();
        let second = Rect::new(20.0, 20.0, 1080.0, 1080.0);
        assert_eq!(layers, [content, second, second]);
        // Scaled layers grow by the scaled deviation.
        let scaled = chain_layers(&one, &Affine::scale(2.0), far, visible).unwrap();
        assert_eq!(scaled, [Rect::new(-550.0, 0.0, 0.0, 10.0)]);
        // Primitives that move or create alpha keep the padded layer.
        let shadow = FilterPrimitive::DropShadow {
            dx: 1.0,
            dy: 1.0,
            std_deviation: 1.0,
            color: crate::color::palette::css::RED,
            edge_mode: EdgeMode::None,
        };
        let mut alpha = matrices::IDENTITY;
        alpha[19] = 0.5;
        for other in [shadow, FilterPrimitive::ColorMatrix { matrix: alpha }] {
            let chain = Filter::from_chain([blur(1.0), other]).unwrap();
            assert_eq!(chain_layers(&chain, &identity, content, visible), None);
        }
        assert_eq!(
            chain_layers(
                &Filter::from_primitive(blur(1.0)),
                &identity,
                content,
                visible
            ),
            None
        );
    }
}
