// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image sampling as Skia's raster bitmap shader does it
//! ([`SamplingPrecision::FixedPoint`]).
//!
//! `SkBitmapProcState` maps each pixel center through the inverse matrix in
//! single precision (`SkMatrix::Affine_xy`; the inverse as `SkMatrix::invert`
//! computes it) and keeps the result in 16.16
//! fixed point. Filtering, it offsets the point by half a texel: the two
//! texels on each axis are the integer part and the next one, clamped to the
//! image, and the next one's weight is the top four fractional bits
//! (`extract_low_bits_clamp_clamp`). `filter_and_scale_by_alpha` sums the four
//! texels with those weights, in 256ths, and truncates. Unfiltered, it biases
//! the point down by one fixed-point unit so that exact texel boundaries pick
//! the earlier texel. Along a span, only the first pixel is mapped; the rest
//! step from it by the inverse matrix's x column in 32.32 fixed point.
//!
//! Bit-exactness, not speed, is the goal: each pixel is mapped and sampled
//! in scalar code, where the exact painters are SIMD. The `feImage` sampler
//! (`crate::filter::image`) approximates the same bilinear weights in floats.

use crate::fine::macros::u8x16_painter;
use crate::peniko::{Extend, ImageQuality};
use vello_common::color::PremulRgba8;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{Simd, SimdBase, u8x16};
use vello_common::geometry::RectU16;
use vello_common::kurbo::Affine;
use vello_common::paint::SamplingPrecision;
use vello_common::pixmap::Pixmap;

/// The largest image side Skia's shader filters: its packed coordinates keep
/// 14 bits of texel index beside the 4-bit weight.
const MAX_FILTERED_SIDE: u16 = (1 << 14) - 1;
/// The largest image side Skia's shader samples at all; larger images take its
/// exact pipeline.
const MAX_SHADER_SIDE: u16 = (1 << 15) - 1;

/// How Skia's bitmap shader samples an image.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedPointSampling {
    /// Bilinear with four-bit weights, or the nearest texel.
    filtered: bool,
    /// The texels sampled.
    texels: Texels,
}

impl FixedPointSampling {
    /// How `image`, whose pixels are `pixmap`, is sampled at
    /// [`SamplingPrecision::FixedPoint`]; `None` when exact sampling applies.
    pub(crate) fn of(image: &EncodedImage, pixmap: &Pixmap) -> Option<Self> {
        let clamped =
            image.sampler.x_extend == Extend::Pad && image.sampler.y_extend == Extend::Pad;
        let subset = match image.sampling_precision {
            SamplingPrecision::FixedPoint { subset, .. } => subset,
            SamplingPrecision::FixedPointAtOrigin { .. } => None,
            SamplingPrecision::Exact => return None,
        };
        // Callers paint empty images transparent first; the painter's
        // clamping still needs at least one texel.
        let texels = Texels::of(subset, pixmap)?;
        if !clamped {
            return None;
        }
        let [_, _, width, height] = texels.bounds;
        let side = width.max(height);
        let filtered = match image.sampler.quality {
            ImageQuality::Medium if side <= MAX_FILTERED_SIDE => true,
            ImageQuality::Low | ImageQuality::Medium if side <= MAX_SHADER_SIDE => false,
            _ => return None,
        };
        Some(Self { filtered, texels })
    }
}

/// The texels Skia samples: the whole image, or the part it extracted.
#[derive(Clone, Copy, Debug)]
struct Texels {
    /// The first column and row, then the number of columns and rows.
    bounds: [u16; 4],
}

impl Texels {
    /// `subset` within `pixmap`, or the whole pixmap; `None` when empty.
    fn of(subset: Option<RectU16>, pixmap: &Pixmap) -> Option<Self> {
        let (width, height) = (pixmap.width(), pixmap.height());
        let (x0, y0, x1, y1) = match subset {
            Some(r) => (r.x0, r.y0, r.x1.min(width), r.y1.min(height)),
            None => (0, 0, width, height),
        };
        (x0 < x1 && y0 < y1).then_some(Self {
            bounds: [x0, y0, x1 - x0, y1 - y0],
        })
    }
}

/// A painter for clamped images sampled as Skia's bitmap shader does,
/// filtered with quantized weights or unfiltered.
#[derive(Debug)]
pub(crate) struct FixedPointImagePainter<'a, S: Simd> {
    pixmap: &'a Pixmap,
    texels: Texels,
    /// The inverse transform's coefficients, in single precision as Skia
    /// keeps its matrices.
    transform: [f32; 6],
    origin: [f32; 2],
    /// The x and y steps along a span, in 32.32 fixed point.
    steps: [i64; 2],
    /// The center of the device column where spans start.
    span_x: Option<f32>,
    /// The pixels Skia shades per chunk of a span.
    chunk: i64,
    filtered: bool,
    /// The device position of the next column's top pixel center.
    x: f32,
    y: f32,
    simd: S,
}

impl<'a, S: Simd> FixedPointImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        FixedPointSampling { filtered, texels }: FixedPointSampling,
        [start_x, start_y]: [f64; 2],
    ) -> Self {
        let (span_x, origin) = match image.sampling_precision {
            SamplingPrecision::FixedPoint { span_x, .. } => (span_x, [0, 0]),
            SamplingPrecision::FixedPointAtOrigin { span_x, origin } => (span_x, origin),
            SamplingPrecision::Exact => (None, [0, 0]),
        };
        let [x0, y0, ..] = texels.bounds;
        let original =
            image.transform * Affine::translate((-f64::from(origin[0]), -f64::from(origin[1])));
        let transform = skia_inverse(original, [x0, y0]);
        let origin = origin.map(|v| v as f32);
        let span_x = span_x.map(|x| x as f32 + origin[0] + 0.5);

        Self {
            pixmap,
            texels,
            transform,
            origin,
            steps: [fractional(transform[0]), fractional(transform[1])],
            span_x,
            chunk: chunk(transform, filtered),
            filtered,
            x: single(start_x),
            y: single(start_y),
            simd,
        }
    }

    /// The premultiplied pixel centered at device `(x, y)`.
    fn pixel(&self, x: f32, y: f32) -> [u8; 4] {
        let [_, _, width, height] = self.texels.bounds;
        let [u, v] = self.position(x, y);
        if !self.filtered {
            let PremulRgba8 { r, g, b, a } = self.texel(nearest(u, width), nearest(v, height));
            return [r, g, b, a];
        }
        let (x0, x1, wx) = filtered(u, width);
        let (y0, y1, wy) = filtered(v, height);
        // `filter_and_scale_by_alpha`: weights in 256ths that sum to 256.
        let taps = [
            (x0, y0, (16 - wx) * (16 - wy)),
            (x1, y0, wx * (16 - wy)),
            (x0, y1, (16 - wx) * wy),
            (x1, y1, wx * wy),
        ];
        let mut sum = [0_u32; 4];
        for (x, y, weight) in taps {
            let PremulRgba8 { r, g, b, a } = self.texel(x, y);
            for (s, channel) in sum.iter_mut().zip([r, g, b, a]) {
                *s += u32::from(channel) * weight;
            }
        }
        sum.map(|s| (s >> 8) as u8)
    }

    /// The image position of device pixel center `(x, y)` in 32.32 fixed
    /// point. Skia shades a span in chunks that fit its buffer
    /// (`SkBitmapProcState::maxCountForBufferSize`), mapping each chunk's
    /// first pixel in single precision and stepping from it in fixed point;
    /// without a known span start, each pixel is mapped on its own.
    fn position(&self, x: f32, y: f32) -> [i64; 2] {
        let [a, b, c, d, e, f] = self.transform;
        let (x, y) = (x + self.origin[0], y + self.origin[1]);
        let map = |x: f32| {
            [
                fractional((x * a + y * c) + e),
                fractional((x * b + y * d) + f),
            ]
        };
        let Some(span_x) = self.span_x else {
            return map(x);
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Both are pixel centers of a u16 surface, a whole number of pixels apart."
        )]
        let steps = (x - span_x) as i64;
        let chunk = self.chunk;
        let chunk_x = span_x + (steps.div_euclid(chunk) * chunk) as f32;
        let steps = steps.rem_euclid(chunk);
        let start = map(chunk_x);
        [0, 1].map(|axis| start[axis].saturating_add(steps.saturating_mul(self.steps[axis])))
    }

    /// The texel at `(x, y)` of the sampled texels.
    fn texel(&self, x: u16, y: u16) -> PremulRgba8 {
        let [x0, y0, ..] = self.texels.bounds;
        self.pixmap.sample(x0 + x, y0 + y)
    }
}

impl<S: Simd> Iterator for FixedPointImagePainter<'_, S> {
    type Item = u8x16<S>;

    fn next(&mut self) -> Option<Self::Item> {
        // A column of `Tile::HEIGHT` (four) pixels.
        let mut column = [0_u8; 16];
        for (row, pixel) in column.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *pixel = self.pixel(self.x, self.y + row as f32);
        }
        self.x += 1.0;
        Some(u8x16::from_slice(self.simd, &column))
    }
}

u8x16_painter!(FixedPointImagePainter<'_, S>);

/// The device-to-texel matrix as Skia computes it (`SkMatrix::invert`) from
/// its single-precision matrix for the texels sampled. `inverse` is the exact
/// inverse of the image-to-device paint transform; moved to the sampled
/// part's corner `origin` and inverted again, it recovers that matrix's
/// single-precision values (see [`SamplingPrecision::FixedPoint`]). Skia
/// inverts a scale and translation in single precision, and any other matrix
/// in double precision, rounded.
fn skia_inverse(inverse: Affine, [x0, y0]: [u16; 2]) -> [f32; 6] {
    let origin = Affine::translate((f64::from(x0), f64::from(y0)));
    let [a, b, c, d, e, f] = (inverse.inverse() * origin).as_coeffs().map(single);
    if b == 0.0 && c == 0.0 && a != 0.0 && d != 0.0 {
        let (inv_x, inv_y) = (1.0 / a, 1.0 / d);
        return [inv_x, 0.0, 0.0, inv_y, -e * inv_x, -f * inv_y];
    }
    let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(f64::from);
    let inv_det = 1.0 / (a * d - b * c);
    [
        d * inv_det,
        -b * inv_det,
        -c * inv_det,
        a * inv_det,
        (c * f - d * e) * inv_det,
        (b * e - a * f) * inv_det,
    ]
    .map(single)
}

/// The pixels Skia's shader maps per chunk: its 512-byte buffer holds a
/// shared y and 16-bit x pairs for a scale and translation, or 32-bit x and
/// y pairs otherwise, halved again for the two taps of each filtered axis.
fn chunk(transform: [f32; 6], filtered: bool) -> i64 {
    let [_, b, c, ..] = transform;
    let per_buffer = if b == 0.0 && c == 0.0 { 254 } else { 128 };
    if filtered { per_buffer / 2 } else { per_buffer }
}

/// `SkScalarToFractionalInt`: `t` in 32.32 fixed point, truncated. Values
/// beyond ±2^30 saturate there, far past any image, and NaN becomes zero.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The value is clamped well within the i64 range."
)]
fn fractional(t: f32) -> i64 {
    const LIMIT: f64 = (1 << 30) as f64;
    (f64::from(t).clamp(-LIMIT, LIMIT) * 4_294_967_296.0) as i64
}

/// The two texels around 32.32 image coordinate `t` on an axis of `size`
/// texels, clamped to it, and the second one's weight in sixteenths. The
/// coordinate is offset by half a texel, then kept in 16.16.
#[expect(clippy::cast_sign_loss, reason = "The low four bits are masked.")]
fn filtered(t: i64, size: u16) -> (u16, u16, u32) {
    let fixed = t.saturating_sub(1 << 31) >> 16;
    let weight = ((fixed >> 12) & 0xF) as u32;
    (
        clamp(fixed, size),
        clamp(fixed.saturating_add(0x10000), size),
        weight,
    )
}

/// The texel containing 32.32 image coordinate `t` on an axis of `size`
/// texels, clamped to it. Biased down one 16.16 unit, a boundary belongs to
/// the earlier texel.
fn nearest(t: i64, size: u16) -> u16 {
    clamp((t >> 16).saturating_sub(1), size)
}

/// The texel of 16.16 coordinate `fixed` on an axis of `size` texels.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "The index is clamped to the axis, whose size is a u16."
)]
fn clamp(fixed: i64, size: u16) -> u16 {
    (fixed >> 16).clamp(0, i64::from(size) - 1) as u16
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Skia keeps matrices and device positions in single precision."
)]
fn single(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_axes_keep_four_fractional_bits() {
        let filtered = |t: f32, size| filtered(fractional(t), size);
        assert_eq!(filtered(5.0, 9), (4, 5, 8));
        assert_eq!(filtered(4.8, 9), (4, 5, 4));
        // A hair less than a sixteenth truncates to none.
        assert_eq!(filtered(4.5625 - 1e-6, 9), (4, 5, 0));
        assert_eq!(filtered(4.5625, 9), (4, 5, 1));
    }

    #[test]
    fn filtered_axes_clamp_to_the_image() {
        let filtered = |t: f32, size| filtered(fractional(t), size);
        assert_eq!(filtered(0.2, 9), (0, 0, 11));
        let (first, second, _) = filtered(-1e30, 9);
        assert_eq!((first, second), (0, 0));
        assert_eq!(filtered(8.9, 9), (8, 8, 6));
        let (first, second, _) = filtered(f32::INFINITY, 9);
        assert_eq!((first, second), (8, 8));
        let (first, second, _) = filtered(f32::NAN, 9);
        assert_eq!((first, second), (0, 0));
    }

    #[test]
    fn nearest_boundaries_pick_the_earlier_texel() {
        let nearest = |t: f32, size| nearest(fractional(t), size);
        assert_eq!(nearest(5.0, 9), 4);
        // Less than a 16.16 unit past the boundary still truncates.
        assert_eq!(nearest(5.0 + 1e-6, 9), 4);
        assert_eq!(nearest(5.0 + 1.0 / 32768.0, 9), 5);
        assert_eq!(nearest(0.0, 9), 0);
        assert_eq!(nearest(-1e30, 9), 0);
        assert_eq!(nearest(1e30, 9), 8);
    }

    mod render {
        use crate::peniko::{Extend, ImageBrush, ImageQuality, ImageSampler};
        use crate::{RasterizerSettings, RenderContext, RenderMode, Resources};
        use alloc::sync::Arc;
        use alloc::vec::Vec;
        use core::f64::consts::FRAC_PI_2;
        use vello_common::color::PremulRgba8;
        use vello_common::geometry::RectU16;
        use vello_common::kurbo::{Affine, Rect};
        use vello_common::paint::{ImageSource, PaintType, SamplingPrecision};
        use vello_common::pixmap::Pixmap;

        const WHITE: PremulRgba8 = PremulRgba8 {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };

        /// A `width`×1 transparent image with one white texel at 4, as a
        /// canvas `fillRect(4, 0, 1, 1)` makes it.
        fn impulse(width: u16) -> Arc<Pixmap> {
            let mut pixmap = Pixmap::new(width, 1);
            pixmap.data_mut()[4] = WHITE;
            Arc::new(pixmap)
        }

        /// Draw `image` with `sampler` at fixed-point precision through `transform` (the
        /// canvas's) and `image_to_dst`, over `area` in user space; return
        /// the alphas of `row` of a `width`×`height` canvas.
        fn draw(
            image: Arc<Pixmap>,
            [transform, image_to_dst]: [Affine; 2],
            area: Rect,
            sampler: ImageSampler,
            (width, height, row): (u16, u16, u16),
            span_x: Option<i32>,
            mode: RenderMode,
        ) -> Vec<u8> {
            let mut scene = RenderContext::new(width, height);
            scene.set_sampling_precision(SamplingPrecision::FixedPoint {
                span_x,
                subset: None,
            });
            scene.set_transform(transform);
            scene.set_paint(PaintType::Image(ImageBrush {
                image: ImageSource::Pixmap(image),
                sampler,
            }));
            scene.set_paint_transform(image_to_dst);
            scene.fill_rect(&area);
            scene.flush();
            let mut pixmap = Pixmap::new(width, height);
            let settings = RasterizerSettings {
                render_mode: mode,
                ..RasterizerSettings::default()
            };
            scene.render_with(&mut pixmap, &mut Resources::new(), settings);
            (0..width).map(|x| pixmap.sample(x, row).a).collect()
        }

        /// Spread `(index, alpha)` pairs over a zeroed row of `width`.
        fn row(width: usize, values: &[(usize, u8)]) -> Vec<u8> {
            let mut row = alloc::vec![0; width];
            for &(index, alpha) in values {
                row[index] = alpha;
            }
            row
        }

        fn modes() -> Vec<RenderMode> {
            let mut modes = alloc::vec![RenderMode::OptimizeSpeed];
            if cfg!(feature = "f32_pipeline") {
                modes.push(RenderMode::OptimizeQuality);
            }
            modes
        }

        /// A 9×1 impulse drawn 2.5 times wider (`drawImage(source, 0, 0, 22.5, 4)`).
        fn widened(width: u16, sampler: ImageSampler, mode: RenderMode) -> Vec<u8> {
            let placement = [Affine::IDENTITY, Affine::scale_non_uniform(2.5, 4.0)];
            let area = Rect::new(0.0, 0.0, 22.5, 4.0);
            draw(
                impulse(width),
                placement,
                area,
                sampler,
                (24, 4, 1),
                None,
                mode,
            )
        }

        /// A named placement (canvas and image transforms), its area and
        /// Chrome's nonzero alphas.
        type Case = (&'static str, [Affine; 2], Rect, &'static [(usize, u8)]);

        /// Chrome 154 (`fixtures/local/canvas-upscale` in submilli-browser):
        /// alphas of a 9×1 impulse upscaled at low smoothing quality.
        #[test]
        fn upscales_match_chrome() {
            let scaled =
                |x: f64, sx: f64| Affine::translate((x, 0.0)) * Affine::scale_non_uniform(sx, 4.0);
            let cases: [Case; 4] = [
                (
                    "2.5",
                    [Affine::IDENTITY, scaled(0.0, 2.5)],
                    Rect::new(0.0, 0.0, 22.5, 4.0),
                    &[(9, 63), (10, 175), (11, 239), (12, 127), (13, 31)],
                ),
                (
                    "offset",
                    [Affine::IDENTITY, scaled(0.3, 2.5)],
                    Rect::new(0.3, 0.0, 22.8, 4.0),
                    &[(9, 31), (10, 143), (11, 239), (12, 159), (13, 63)],
                ),
                (
                    "flipped",
                    [
                        Affine::translate((22.5, 0.0)) * Affine::scale_non_uniform(-1.0, 1.0),
                        scaled(0.0, 2.5),
                    ],
                    Rect::new(0.0, 0.0, 22.5, 4.0),
                    &[(9, 79), (10, 191), (11, 223), (12, 127), (13, 15)],
                ),
                (
                    "3",
                    [Affine::IDENTITY, scaled(0.0, 3.0)],
                    Rect::new(0.0, 0.0, 27.0, 4.0),
                    &[(11, 79), (12, 159), (13, 255), (14, 175), (15, 95)],
                ),
            ];
            for mode in modes() {
                for (name, placement, area, chrome) in cases {
                    let alphas = draw(
                        impulse(9),
                        placement,
                        area,
                        ImageSampler::default(),
                        (28, 4, 1),
                        None,
                        mode,
                    );
                    assert_eq!(alphas, row(28, chrome), "{name} {mode:?}");
                }
            }
        }

        /// Chrome 154: rotated a quarter turn, each image column fills a row.
        #[test]
        fn rotated_upscales_match_chrome() {
            let transform = Affine::translate((4.0, 0.0)) * Affine::rotate(FRAC_PI_2);
            let placement = [transform, Affine::scale_non_uniform(2.5, 4.0)];
            let area = Rect::new(0.0, 0.0, 22.5, 4.0);
            for mode in modes() {
                let alphas = draw(
                    impulse(9),
                    placement,
                    area,
                    ImageSampler::default(),
                    (4, 24, 10),
                    None,
                    mode,
                );
                assert_eq!(alphas, [175; 4], "{mode:?}");
            }
        }

        /// Chrome 154: Skia filters only images whose sides fit in 14 bits,
        /// samples the nearest texel up to 15 bits, and leaves larger ones
        /// to its exact pipeline.
        #[test]
        fn large_images_fall_back_like_skia() {
            let exact = [(9, 77), (10, 179), (11, 230), (12, 128), (13, 25)];
            for mode in modes() {
                let filtered = widened(16383, ImageSampler::default(), mode);
                assert_eq!(
                    filtered,
                    row(24, &[(9, 63), (10, 175), (11, 239), (12, 127), (13, 31)]),
                    "{mode:?}"
                );
                let nearest = widened(16384, ImageSampler::default(), mode);
                assert_eq!(
                    nearest,
                    row(24, &[(10, 255), (11, 255), (12, 255)]),
                    "{mode:?}"
                );
                let unfiltered = widened(32768, ImageSampler::default(), mode);
                assert_near(&unfiltered, &row(24, &exact), mode);
            }
        }

        /// Chrome 154 with smoothing disabled: a texel boundary at an exact
        /// pixel center (12.5 maps to 5) belongs to the earlier texel.
        #[test]
        fn nearest_ties_pick_the_earlier_texel_like_chrome() {
            let nearest = ImageSampler::default().with_quality(ImageQuality::Low);
            for mode in modes() {
                let alphas = widened(9, nearest, mode);
                assert_eq!(
                    alphas,
                    row(24, &[(10, 255), (11, 255), (12, 255)]),
                    "{mode:?}"
                );
            }
        }

        /// Chrome 154: `drawImage(source, 1, 0, 5, 1, 0, 0, 88 / 9, 3)`, a
        /// span starting at column 0. Column 5 maps exactly to a sixteenth
        /// (3.3125 less half a texel) when mapped on its own, but stepping
        /// from the span start in fixed point lands just below it.
        #[test]
        fn spans_step_from_their_first_pixel_like_chrome() {
            let width = 88.0 / 9.0;
            let image_to_dst =
                Affine::scale_non_uniform(width / 5.0, 3.0) * Affine::translate((-1.0, 0.0));
            let placement = [Affine::IDENTITY, image_to_dst];
            let area = Rect::new(0.0, 0.0, 10.0, 3.0);
            let chrome = row(12, &[(5, 63), (6, 207), (7, 175), (8, 47)]);
            for mode in modes() {
                let sampler = ImageSampler::default();
                let alphas = draw(
                    impulse(9),
                    placement,
                    area,
                    sampler,
                    (12, 3, 1),
                    Some(0),
                    mode,
                );
                assert_eq!(alphas, chrome, "{mode:?}");
            }
        }

        /// Chrome 154: a 64×1 image of alternating white and transparent
        /// texels drawn 350 wide. Skia maps again at each 127-pixel chunk of
        /// the span; stepping straight through reads 1/16 off at column 262.
        #[test]
        fn long_spans_restart_every_chunk_like_chrome() {
            let mut stripes = Pixmap::new(64, 1);
            for texel in stripes.data_mut().iter_mut().step_by(2) {
                *texel = WHITE;
            }
            let placement = [
                Affine::IDENTITY,
                Affine::scale_non_uniform(350.0 / 64.0, 3.0),
            ];
            let area = Rect::new(0.0, 0.0, 350.0, 3.0);
            let chrome = [
                63, 111, 159, 207, 255, 207, 159, 111, 63, 15, 31, 79, 127, 159, 207, 255, 207,
                159, 111, 63,
            ];
            for mode in modes() {
                let sampler = ImageSampler::default();
                let row = (360, 3, 1);
                let alphas = draw(
                    Arc::new(stripes.clone()),
                    placement,
                    area,
                    sampler,
                    row,
                    Some(0),
                    mode,
                );
                assert_eq!(alphas[250..270], chrome, "{mode:?}");
            }
        }

        /// Chrome 154: `drawImage(stripes, 3800, 0, 100, 1, 0, 0, 410, 1)`
        /// unsmoothed, from a 4000-texel image of alternating white and
        /// transparent texels. Skia maps relative to the extracted part,
        /// where single precision is fine enough for column 20 (texel 3805
        /// exactly) to take the earlier texel; relative to the whole image it
        /// lands past the boundary.
        #[test]
        fn unfiltered_parts_map_relative_to_their_subset_like_chrome() {
            let mut stripes = Pixmap::new(4000, 1);
            for texel in stripes.data_mut().iter_mut().step_by(2) {
                *texel = WHITE;
            }
            let image_to_dst =
                Affine::scale_non_uniform(4.1, 1.0) * Affine::translate((-3800.0, 0.0));
            let chrome = [0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 255];
            for mode in modes() {
                let mut scene = RenderContext::new(420, 1);
                scene.set_sampling_precision(SamplingPrecision::FixedPoint {
                    span_x: Some(0),
                    subset: Some(RectU16::new(3800, 0, 3900, 1)),
                });
                scene.set_paint(PaintType::Image(ImageBrush {
                    image: ImageSource::Pixmap(Arc::new(stripes.clone())),
                    sampler: ImageSampler::default().with_quality(ImageQuality::Low),
                }));
                scene.set_paint_transform(image_to_dst);
                scene.fill_rect(&Rect::new(0.0, 0.0, 410.0, 1.0));
                scene.flush();
                let mut pixmap = Pixmap::new(420, 1);
                let settings = RasterizerSettings {
                    render_mode: mode,
                    ..RasterizerSettings::default()
                };
                scene.render_with(&mut pixmap, &mut Resources::new(), settings);
                let alphas: Vec<u8> = (15..26).map(|x| pixmap.sample(x, 0).a).collect();
                assert_eq!(alphas, chrome, "{mode:?}");
            }
        }

        /// An empty image has no texels: it paints transparent, without panicking.
        #[test]
        fn empty_images_paint_transparent() {
            for mode in modes() {
                let alphas = draw(
                    Arc::new(Pixmap::new(0, 1)),
                    [Affine::IDENTITY, Affine::scale(2.0)],
                    Rect::new(0.0, 0.0, 4.0, 4.0),
                    ImageSampler::default(),
                    (4, 4, 1),
                    Some(0),
                    mode,
                );
                assert_eq!(alphas, [0; 4], "{mode:?}");
            }
        }

        /// Skia's shader only clamps, so repeating images keep exact weights.
        #[test]
        fn repeating_images_keep_exact_weights() {
            let exact = [(9, 77), (10, 179), (11, 230), (12, 128), (13, 25)];
            for mode in modes() {
                assert_near(
                    &widened(9, ImageSampler::default().with_extend(Extend::Repeat), mode),
                    &row(24, &exact),
                    mode,
                );
            }
        }

        #[test]
        fn translated_surfaces_keep_original_fixed_point_sample_positions() {
            let image = impulse(14);
            let placement = Affine::new([0.85714287, 0.0, 0.0, 0.85714287, 4.0, 11.1428566]);
            let render = |origin: [i32; 2], size: [u16; 2], preserve| {
                let mut context = RenderContext::new(size[0], size[1]);
                let local =
                    Affine::translate((-f64::from(origin[0]), -f64::from(origin[1]))) * placement;
                let span_x = Some(
                    (placement
                        .transform_rect_bbox(Rect::new(0.0, 0.0, 14.0, 1.0))
                        .x0
                        .ceil() as i32)
                        - origin[0],
                );
                context.set_sampling_precision(if preserve {
                    SamplingPrecision::FixedPointAtOrigin { span_x, origin }
                } else {
                    SamplingPrecision::FixedPoint {
                        span_x,
                        subset: None,
                    }
                });
                context.set_transform(Affine::IDENTITY);
                context.set_paint(PaintType::Image(ImageBrush {
                    image: ImageSource::Pixmap(Arc::clone(&image)),
                    sampler: ImageSampler::default(),
                }));
                context.set_paint_transform(local);
                context.fill_rect(&Rect::new(0.0, 0.0, f64::from(size[0]), f64::from(size[1])));
                let mut output = Pixmap::new(size[0], size[1]);
                context.render(output.as_mut(), &mut Resources::new());
                output
            };
            let full = render([0, 0], [24, 24], true);
            let cropped = render([4, 11], [12, 2], true);
            let rebased = render([4, 11], [12, 2], false);
            for y in 0..2 {
                for x in 0..12 {
                    assert_eq!(cropped.sample(x, y), full.sample(x + 4, y + 11));
                }
            }
            assert_ne!(
                cropped.data(),
                rebased.data(),
                "rebasing before f32 mapping loses the phase"
            );
        }

        #[track_caller]
        fn assert_near(actual: &[u8], expected: &[u8], mode: RenderMode) {
            let near = actual
                .iter()
                .zip(expected)
                .all(|(a, e)| a.abs_diff(*e) <= 1);
            assert!(near, "{mode:?}: {actual:?} versus {expected:?}");
        }
    }
}
