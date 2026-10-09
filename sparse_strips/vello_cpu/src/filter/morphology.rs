// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feMorphology` with a radius-independent sliding window.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feMorphologyElement>
use super::bounds::for_each_column;
use super::channels::{store, working};
use alloc::vec::Vec;
use vello_common::filter::graph::{ColorSpace, MAX_MORPHOLOGY_RADIUS};
use vello_common::filter_effects::MorphologyOperator;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

/// Erode or dilate premultiplied channels over a `(2rx+1) x (2ry+1)` rectangle.
/// Radii round half away from zero and are capped at [`MAX_MORPHOLOGY_RADIUS`], as in Chrome;
/// pixels beyond the raster are transparent black. A zero radius leaves that axis
/// unchanged.
pub(super) fn apply(
    pixels: &mut Pixmap,
    operator: MorphologyOperator,
    radius: [f32; 2],
    space: ColorSpace,
) {
    let width = usize::from(pixels.width());
    let height = usize::from(pixels.height());
    if width == 0 || height == 0 {
        return;
    }
    let dilate = operator == MorphologyOperator::Dilate;
    // Beyond the raster every window already covers the whole axis.
    let rx = window_radius(radius[0], width);
    let ry = window_radius(radius[1], height);
    let mut data = working(pixels, space);
    let mut line = Window::default();
    if rx > 0 {
        for row in data.chunks_exact_mut(width) {
            line.apply(row, rx, dilate);
        }
    }
    if ry > 0 {
        for_each_column(&mut data, width, |column| line.apply(column, ry, dilate));
    }
    store(pixels, &data, space);
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Radii are finite and nonnegative; the cast saturates before the cap."
)]
fn window_radius(radius: f32, extent: usize) -> usize {
    (radius.max(0.0).round() as usize)
        .min(MAX_MORPHOLOGY_RADIUS as usize)
        .min(extent)
}

/// Scratch for van Herk/Gil-Werman: block prefix and suffix extrema over a line
/// padded with `radius` transparent pixels on each side.
#[derive(Default)]
struct Window {
    padded: Vec<[f32; 4]>,
    prefix: Vec<[f32; 4]>,
    suffix: Vec<[f32; 4]>,
}

impl Window {
    fn apply(&mut self, line: &mut [[f32; 4]], radius: usize, dilate: bool) {
        let size = 2 * radius + 1;
        let pick = |a: [f32; 4], b: [f32; 4]| {
            core::array::from_fn(|i| {
                if dilate {
                    a[i].max(b[i])
                } else {
                    a[i].min(b[i])
                }
            })
        };
        self.padded.clear();
        self.padded.resize(radius, [0.0; 4]);
        self.padded.extend_from_slice(line);
        self.padded.resize(line.len() + 2 * radius, [0.0; 4]);
        let n = self.padded.len();
        self.prefix.clear();
        self.prefix.extend_from_slice(&self.padded);
        self.suffix.clear();
        self.suffix.extend_from_slice(&self.padded);
        for i in 1..n {
            if i % size != 0 {
                self.prefix[i] = pick(self.prefix[i - 1], self.padded[i]);
            }
        }
        for i in (0..n - 1).rev() {
            if (i + 1) % size != 0 {
                self.suffix[i] = pick(self.suffix[i + 1], self.padded[i]);
            }
        }
        // Window [i, i + size) spans at most two blocks: a suffix and a prefix.
        for (i, value) in line.iter_mut().enumerate() {
            *value = pick(self.suffix[i], self.prefix[i + size - 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use vello_common::color::PremulRgba8;

    fn row(alpha: &[u8]) -> Pixmap {
        let mut pixels = Pixmap::new(alpha.len() as u16, 1);
        for (p, a) in pixels.data_mut().iter_mut().zip(alpha) {
            *p = PremulRgba8 {
                r: *a,
                g: 0,
                b: 0,
                a: *a,
            };
        }
        pixels
    }
    fn alphas(pixels: &Pixmap) -> Vec<u8> {
        pixels.data().iter().map(|p| p.a).collect()
    }

    #[test]
    fn radii_round_like_chrome_and_edges_are_transparent() {
        for (radius, expected) in [
            (0.4, [0, 0, 0, 255, 0, 0, 0]),
            (0.5, [0, 0, 255, 255, 255, 0, 0]),
            (1.5, [0, 255, 255, 255, 255, 255, 0]),
        ] {
            let mut pixels = row(&[0, 0, 0, 255, 0, 0, 0]);
            apply(
                &mut pixels,
                MorphologyOperator::Dilate,
                [radius, 0.0],
                ColorSpace::Srgb,
            );
            assert_eq!(alphas(&pixels), expected, "{radius}");
        }
        let mut pixels = row(&[255, 255, 255, 255, 128]);
        apply(
            &mut pixels,
            MorphologyOperator::Erode,
            [1.0, 0.0],
            ColorSpace::Srgb,
        );
        assert_eq!(alphas(&pixels), [0, 255, 255, 128, 0]);
    }

    #[test]
    fn radii_are_capped_like_chrome() {
        let mut alpha = vec![0_u8; 600];
        alpha[0] = 255;
        let mut pixels = row(&alpha);
        apply(
            &mut pixels,
            MorphologyOperator::Dilate,
            [1000.0, 0.0],
            ColorSpace::Srgb,
        );
        assert_eq!(pixels.data()[256].a, 255);
        assert_eq!(pixels.data()[257].a, 0);
    }

    #[test]
    fn huge_radii_cover_the_raster_without_growing_work() {
        let mut pixels = row(&[0, 9, 0, 200, 0]);
        apply(
            &mut pixels,
            MorphologyOperator::Dilate,
            [1e6, 1e6],
            ColorSpace::Srgb,
        );
        assert_eq!(alphas(&pixels), [200; 5]);
        apply(
            &mut pixels,
            MorphologyOperator::Erode,
            [1e6, 0.0],
            ColorSpace::Srgb,
        );
        assert_eq!(alphas(&pixels), [0; 5]);
    }
}
