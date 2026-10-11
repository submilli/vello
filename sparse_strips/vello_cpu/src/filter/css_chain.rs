// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sequential CSS chains of blurs, computed as Chrome (Skia) computes them.
//!
//! Each blur takes the box and rescale passes of SVG graph blurs on 8-bit
//! surfaces, with transparent edges, and rescales about the center of its
//! input's layer ([`chain_layers`]).
use super::bounds::PixelBounds;
use vello_common::filter::FilterLayerPlacement;
use vello_common::filter::css::chain_layers;
use vello_common::filter::gaussian_blur::transform_blur_params;
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Point};
use vello_common::pixmap::Pixmap;

/// Apply `filter` to a layer's `pixmap` if it is a chain of blurs; returns
/// `false`, leaving the pixmap unchanged, for other filters.
pub(crate) fn try_apply(
    filter: &Filter,
    pixmap: &mut Pixmap,
    transform: Affine,
    placement: FilterLayerPlacement,
) -> bool {
    let Some(layers) = chain_layers(
        filter,
        &transform,
        placement.content.as_rect(),
        placement.visible.as_rect(),
    ) else {
        return false;
    };
    let (width, height) = (pixmap.width(), pixmap.height());
    for (primitive, layer) in filter.graph.primitives.iter().zip(layers) {
        match primitive {
            FilterPrimitive::GaussianBlur { std_deviation, .. } => {
                let sigma = transform_blur_params(*std_deviation, &transform);
                let layer = PixelBounds::of(layer, Point::ZERO, width, height);
                super::svg_blur::blur(pixmap, [sigma, sigma], layer, ColorSpace::Srgb);
            }
            FilterPrimitive::ColorMatrix { matrix } => super::color_matrix::apply(pixmap, matrix),
            _ => unreachable!("chain_layers plans only blurs and color matrices"),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::{RenderContext, Resources};
    use alloc::vec::Vec;
    use vello_common::color::palette::css::RED;
    use vello_common::filter_effects::{Filter, FilterFunction};
    use vello_common::kurbo::Rect;
    use vello_common::pixmap::Pixmap;

    /// The alpha of row 600 every 100 pixels of a 1200-pixel canvas filled
    /// through `blur(radius)`.
    fn filled_canvas(radius: f32) -> Vec<u8> {
        let blur = Filter::from_function(FilterFunction::Blur { radius });
        let chain = Filter::from_chain(blur.graph.primitives.iter().cloned()).unwrap();
        let mut scene = RenderContext::new(1200, 1200);
        scene.push_filter_layer(chain);
        scene.set_paint(RED);
        scene.fill_rect(&Rect::new(0.0, 0.0, 1200.0, 1200.0));
        scene.pop_layer();
        scene.flush();
        // Only the canvas is computed, not 3σ of padding around it.
        assert!(scene.filter_pixels() <= 1208 * 1208);
        let mut pixmap = Pixmap::new(1200, 1200);
        scene.render(&mut pixmap, &mut Resources::new());
        (0..12).map(|i| pixmap.sample(i * 100, 600).a).collect()
    }

    #[test]
    fn blurs_of_a_filled_canvas_match_chrome() {
        // Chrome 154 (`css-blurs.js` in submilli-browser): two rescale steps at
        // 1000, which clamps to 532, and one at 300.
        for (radius, chrome) in [(1000.0, CHROME_532), (300.0, CHROME_300)] {
            for (index, (actual, expected)) in filled_canvas(radius).iter().zip(chrome).enumerate()
            {
                assert!(actual.abs_diff(expected) <= 1, "{radius} {index}: {actual}");
            }
        }
    }

    const CHROME_532: [u8; 12] = [95, 108, 120, 131, 138, 143, 146, 143, 138, 131, 120, 108];
    const CHROME_300: [u8; 12] = [124, 155, 186, 210, 227, 237, 241, 237, 227, 210, 186, 155];
}
