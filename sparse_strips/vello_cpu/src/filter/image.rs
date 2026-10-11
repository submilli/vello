// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sampling of planned `feImage` texels, as Chrome's Skia backend does.
//!
//! The plan ([`Sampling`]) picks the filter: Mitchell bicubic (B = C = 1/3)
//! when upscaled on both axes; otherwise bilinear with exact weights for a
//! whole-pixel source, blending two mip levels once it downscales, or with
//! weights quantized to sixteenths for a fractional one. Texels clamp to the
//! sampled subset; the destination's edges are anti-aliased by coverage.
//! Image paints at `SamplingPrecision::FixedPoint` reproduce Skia's bitmap
//! shader exactly instead (`crate::fine::common::fixed_point_image`).
use super::bounds::PixelBounds;
use super::channels::TRANSPARENT;
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
use vello_common::filter::image::{FilterImage, Window};
use vello_common::filter::image_sampling::{ImageFilter, Sampling};
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

/// Replace `output` with the image drawn inside `crop`; elsewhere it is transparent.
pub(super) fn draw(output: &mut Pixmap, image: &FilterImage, crop: PixelBounds, origin: Point) {
    output.data_mut().fill(TRANSPARENT);
    if let Some(sampler) = Sampler::new(image) {
        let width = usize::from(output.width());
        for y in crop.y0..crop.y1 {
            for x in crop.x0..crop.x1 {
                let pixel = Point::new(origin.x + x as f64, origin.y + y as f64);
                output.data_mut()[y * width + x] = sampler.pixel(pixel);
            }
        }
    }
    output.recompute_may_have_transparency();
}

/// One level a sample reads, and its share.
struct Level<'a> {
    window: &'a Window,
    level: u8,
    weight: f32,
}

/// Maps destination pixels to texels of the planned levels.
struct Sampler<'a> {
    sampling: &'a Sampling,
    /// One or two levels.
    levels: Vec<Level<'a>>,
}

impl<'a> Sampler<'a> {
    /// `None` when the image lacks a level its plan samples.
    fn new(image: &'a FilterImage) -> Option<Self> {
        let sampling = image.sampling();
        let levels = sampling
            .levels()
            .map(|(level, weight)| {
                Some(Level {
                    window: image.window(level)?,
                    level,
                    weight,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { sampling, levels })
    }

    /// The pixel whose top-left corner is `pixel`, in filter coordinates.
    fn pixel(&self, pixel: Point) -> PremulRgba8 {
        let coverage = coverage(self.sampling.destination, pixel);
        if coverage <= 0.0 {
            return TRANSPARENT;
        }
        let center = Point::new(pixel.x + 0.5, pixel.y + 0.5);
        let mut channels = [0.0_f32; 4];
        for level in &self.levels {
            let sample = self.sample(level, center);
            for (c, s) in channels.iter_mut().zip(sample) {
                *c += s * level.weight;
            }
        }
        premultiplied(channels, coverage)
    }

    fn sample(&self, level: &Level<'_>, center: Point) -> [f32; 4] {
        let point = self.sampling.level_point(level.level, center);
        // Texels clamp to the level, so coordinates beyond the furthest tap
        // (two texels for bicubic) sample the same; clamping first keeps tiny
        // destinations from overflowing the integer texel arithmetic.
        let [width, height] = self.sampling.level_size(level.level).map(f64::from);
        let u = point.x.clamp(-2.0, width + 2.0);
        let v = point.y.clamp(-2.0, height + 2.0);
        let texels = Texels(level.window);
        match self.sampling.filter {
            ImageFilter::Bilinear => texels.bilinear(u, v),
            ImageFilter::QuantizedBilinear => texels.quantized_bilinear(u, v),
            ImageFilter::Bicubic => texels.bicubic(u, v),
        }
    }
}

/// One level's texels, read with clamping.
struct Texels<'a>(&'a Window);

impl Texels<'_> {
    fn texel(&self, x: i64, y: i64) -> [f32; 4] {
        let p = self.0.texel(x, y);
        [p.r, p.g, p.b, p.a].map(f32::from)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Coordinates are clamped near the level, whose sides are u32."
    )]
    fn bilinear(&self, u: f64, v: f64) -> [f32; 4] {
        let (x, y) = (u.floor(), v.floor());
        let (fx, fy) = ((u - x) as f32, (v - y) as f32);
        self.weighted(x as i64, y as i64, [1.0 - fx, fx], [1.0 - fy, fy])
    }

    /// Skia's raster bilinear keeps four fractional bits of each coordinate.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Coordinates are clamped near the level, whose sides are u32."
    )]
    fn quantized_bilinear(&self, u: f64, v: f64) -> [f32; 4] {
        let (x, y) = (u.floor(), v.floor());
        let fx = ((u - x) * 16.0).floor() as f32 / 16.0;
        let fy = ((v - y) * 16.0).floor() as f32 / 16.0;
        let sum = self.weighted(x as i64, y as i64, [1.0 - fx, fx], [1.0 - fy, fy]);
        // Its fixed-point sum truncates rather than rounds.
        sum.map(|c| c.floor())
    }

    fn weighted(&self, x: i64, y: i64, wx: [f32; 2], wy: [f32; 2]) -> [f32; 4] {
        let mut sum = [0.0_f32; 4];
        for (dy, wy) in wy.into_iter().enumerate() {
            for (dx, wx) in wx.into_iter().enumerate() {
                let texel = self.texel(x + dx as i64, y + dy as i64);
                for (s, t) in sum.iter_mut().zip(texel) {
                    *s += t * wx * wy;
                }
            }
        }
        sum
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Coordinates are clamped near the level, whose sides are u32."
    )]
    fn bicubic(&self, u: f64, v: f64) -> [f32; 4] {
        let (x, y) = (u.floor(), v.floor());
        let wx = mitchell((u - x) as f32);
        let wy = mitchell((v - y) as f32);
        let mut sum = [0.0_f32; 4];
        for (dy, wy) in wy.into_iter().enumerate() {
            for (dx, wx) in wx.into_iter().enumerate() {
                let texel = self.texel(x as i64 + dx as i64 - 1, y as i64 + dy as i64 - 1);
                for (s, t) in sum.iter_mut().zip(texel) {
                    *s += t * wx * wy;
                }
            }
        }
        // Negative lobes can leave channels outside the premultiplied range.
        let alpha = sum[3].clamp(0.0, 255.0);
        [
            sum[0].clamp(0.0, alpha),
            sum[1].clamp(0.0, alpha),
            sum[2].clamp(0.0, alpha),
            alpha,
        ]
    }
}

/// Mitchell–Netravali weights (B = C = 1/3) of the four texels around fraction `t`.
fn mitchell(t: f32) -> [f32; 4] {
    let near = |x: f32| (7.0 * x * x * x - 12.0 * x * x + 16.0 / 3.0) / 6.0;
    let far = |x: f32| (-7.0 / 3.0 * x * x * x + 12.0 * x * x - 20.0 * x + 32.0 / 3.0) / 6.0;
    [far(1.0 + t), near(t), near(1.0 - t), far(2.0 - t)]
}

/// The area of the unit pixel at `pixel` that `rect` covers.
fn coverage(rect: Rect, pixel: Point) -> f32 {
    let width = (rect.x1.min(pixel.x + 1.0) - rect.x0.max(pixel.x)).max(0.0);
    let height = (rect.y1.min(pixel.y + 1.0) - rect.y0.max(pixel.y)).max(0.0);
    (width * height) as f32
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Channels are clamped to the 8-bit range before conversion."
)]
fn premultiplied(channels: [f32; 4], coverage: f32) -> PremulRgba8 {
    let [r, g, b, a] = channels.map(|c| (c * coverage).round().clamp(0.0, 255.0) as u8);
    PremulRgba8 { r, g, b, a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use vello_common::filter::image::ImageRows;

    fn alpha_row(alphas: &[u8]) -> Pixmap {
        let mut pixmap = Pixmap::new(alphas.len() as u16, 1);
        for (pixel, &a) in pixmap.data_mut().iter_mut().zip(alphas) {
            *pixel = PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a,
            };
        }
        pixmap
    }

    /// Chrome rounds ties in f32, so readings may differ by one level.
    #[track_caller]
    fn assert_near(actual: &[u8], expected: &[u8]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, e)| a.abs_diff(*e) <= 1),
            "{actual:?} versus Chrome {expected:?}"
        );
    }

    /// Draw `source` of `image` at `destination` into a `width`×`height`
    /// raster, cropped to `crop`.
    fn render(
        image: &Pixmap,
        [source, destination, crop]: [Rect; 3],
        width: u16,
        height: u16,
    ) -> Pixmap {
        let mut output = Pixmap::new(width, height);
        let area = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
        let pixels = PixelBounds::of(area, Point::ORIGIN, width, height);
        let size = [image.width(), image.height()].map(u32::from);
        match FilterImage::plan(size, source, destination, crop, area) {
            Some(plan) => {
                let image = plan.share(Arc::new(image.clone())).unwrap();
                draw(&mut output, &image, pixels, Point::ORIGIN);
            }
            None => output.data_mut().fill(TRANSPARENT),
        }
        output
    }

    /// No crop: the image may extend anywhere.
    const UNCROPPED: Rect = Rect::new(-1e5, -1e5, 1e5, 1e5);

    fn row(image: &Pixmap, source: Rect, destination: Rect, width: u16) -> Vec<u8> {
        let placement = [source, destination, UNCROPPED];
        let output = render(image, placement, width, 1);
        alphas(&output)
    }

    fn alphas(pixmap: &Pixmap) -> Vec<u8> {
        pixmap.data().iter().map(|p| p.a).collect()
    }

    /// Chrome 154, a 4×1 `[1, 0, 1, 0]` alpha image drawn into one row.
    #[test]
    fn whole_pixel_sources_sample_bilinearly_with_clamped_edges() {
        let image = alpha_row(&[255, 0, 255, 0]);
        let source = Rect::new(0.0, 0.0, 4.0, 1.0);
        let at = |x0: f64, width: f64| Rect::new(x0, 0.0, x0 + width, 1.0);
        assert_near(
            &row(&image, source, at(0.0, 20.0), 8),
            &[255, 255, 255, 204, 153, 102, 51, 0],
        );
        assert_near(&row(&image, source, at(-0.4, 4.8), 4), &[191, 21, 234, 64]);
        assert_near(&row(&image, source, at(0.0, 3.0), 3), &[212, 128, 42]);
        // A fractional edge is covered by area.
        assert_near(&row(&image, source, at(0.5, 4.0), 2), &[127, 128]);
    }

    /// Chrome 154: a 10-pixel raster of a 9.8-pixel container, with one opaque
    /// column, shifted by tenths: weights are whole sixteenths, truncated.
    #[test]
    fn fractional_sources_use_four_bit_weights() {
        let mut alphas = [0; 10];
        alphas[4] = 255;
        let image = alpha_row(&alphas);
        let source = Rect::new(0.0, 0.0, 9.8, 1.0);
        let shifted =
            |x: f64| row(&image, source, Rect::new(x, 0.0, x + 9.8, 1.0), 6)[4..6].to_vec();
        assert_eq!(shifted(0.1), [223, 31]);
        assert_eq!(shifted(0.3), [175, 79]);
        assert_eq!(shifted(0.6), [95, 159]);
        assert_eq!(shifted(0.9), [15, 239]);
        // The same raster from a whole-pixel source shifts exactly.
        let whole = Rect::new(0.0, 0.0, 10.0, 1.0);
        assert_near(
            &row(&image, whole, Rect::new(0.1, 0.0, 10.1, 1.0), 6)[4..6],
            &[230, 25],
        );
    }

    /// Chrome 154: a 2×2 red, green / blue, transparent image upscaled on both axes.
    #[test]
    fn sources_upscaled_on_both_axes_are_bicubic() {
        let mut image = Pixmap::new(2, 2);
        let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0; 4]];
        for (pixel, [r, g, b, a]) in image.data_mut().iter_mut().zip(colors) {
            *pixel = PremulRgba8 { r, g, b, a };
        }
        let placement = [
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Rect::new(4.0, 0.0, 8.0, 4.0),
            UNCROPPED,
        ];
        let output = render(&image, placement, 8, 4);
        let at = |x: usize, y: usize| {
            let p = output.data()[y * 8 + x];
            [p.r, p.g, p.b, p.a]
        };
        assert_eq!(at(3, 0), [0; 4]);
        assert_eq!(at(4, 0), [255, 0, 0, 255]);
        // Chrome reads back (198, 63, 0, 255) and (156, 50, 50, 240) unpremultiplied.
        assert_eq!(at(5, 0), [198, 63, 0, 255]);
        let [r, g, b, a] = at(5, 1);
        assert_eq!(a, 240);
        assert!(
            [r, g, b]
                .iter()
                .zip([146, 47, 47])
                .all(|(x, y)| x.abs_diff(y) <= 1)
        );
    }

    /// Chrome 154: a 2x2 image sliced to its top two thirds, upscaled 3x. The
    /// sampled subset is the top row, so the bottom row never blends in.
    #[test]
    fn texels_clamp_to_the_whole_texels_of_the_source() {
        let mut image = Pixmap::new(2, 2);
        let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0; 4]];
        for (pixel, [r, g, b, a]) in image.data_mut().iter_mut().zip(colors) {
            *pixel = PremulRgba8 { r, g, b, a };
        }
        let placement = [
            Rect::new(0.0, 0.0, 2.0, 2.0 / 3.0),
            Rect::new(1.0, 1.0, 7.0, 3.0),
            UNCROPPED,
        ];
        let output = render(&image, placement, 3, 4);
        let p = output.data()[2 * 3 + 1];
        assert_eq!([p.r, p.g, p.b, p.a], [255, 0, 0, 255]);
    }

    /// A destination far smaller than a texel maps samples far beyond the image,
    /// which must clamp rather than overflow.
    #[test]
    fn tiny_destinations_sample_clamped_texels() {
        let image = alpha_row(&[255, 0]);
        let source = Rect::new(0.0, 0.0, 2.0, 1.0);
        assert_eq!(
            row(&image, source, Rect::new(0.0, 0.0, 1e-30, 1.0), 2),
            [0, 0]
        );
        assert_eq!(
            row(&image, source, Rect::new(0.0, 0.0, 1e-9, 1.0), 2),
            [0, 0]
        );
    }

    #[test]
    fn empty_or_disjoint_placements_draw_nothing() {
        let image = alpha_row(&[255]);
        let full = Rect::new(0.0, 0.0, 2.0, 1.0);
        for (source, destination) in [
            (Rect::new(0.0, 0.0, 0.0, 1.0), full),
            (Rect::new(0.0, 0.0, 1.0, 1.0), Rect::new(0.0, 0.0, 0.0, 1.0)),
            (Rect::new(5.0, 0.0, 6.0, 1.0), full),
        ] {
            assert_eq!(row(&image, source, destination, 2), [0, 0]);
        }
    }

    /// The alphas the Chrome probe drew its random images from, in order: the
    /// high bits of a 31-bit linear congruential generator.
    fn probe_alphas() -> impl Iterator<Item = u8> {
        let mut seed = 12345_u32;
        core::iter::from_fn(move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            Some((seed >> 16) as u8)
        })
    }

    fn alpha_image(width: u16, height: u16, alphas: &mut impl Iterator<Item = u8>) -> Pixmap {
        let mut pixmap = Pixmap::new(width, height);
        for (pixel, a) in pixmap.data_mut().iter_mut().zip(alphas) {
            *pixel = PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a,
            };
        }
        pixmap
    }

    /// Chrome 154 drawing random alpha images, stretched over whole canvases:
    /// whole-pixel downscales blend two mip levels chosen by the geometric
    /// mean of the scales, minus half a level. The probe page drew the one-dot
    /// image, then the cases below in order, each taking width × height alphas
    /// from [`probe_alphas`].
    #[test]
    fn whole_pixel_downscales_blend_mip_levels_like_chrome() {
        let dot = alpha_row(&[255, 0, 0, 0, 0, 0, 0, 0]);
        let whole = |w: u16, h: u16| Rect::new(0.0, 0.0, f64::from(w), f64::from(h));
        let draw = |image: &Pixmap, w: u16, h: u16| {
            let placement = [
                whole(image.width(), image.height()),
                whole(w, h),
                whole(w, h),
            ];
            alphas(&render(image, placement, w, h))
        };
        // Bilinear alone would read 0.
        assert_eq!(draw(&dot, 2, 1), [32, 0]);
        let mut stream = probe_alphas();
        let cases: [(u16, u16, u16, u16, &[u8]); 12] = [
            (8, 1, 2, 1, &[88, 91]),
            (8, 1, 3, 1, &[222, 39, 41]),
            (16, 1, 5, 1, &[122, 170, 67, 172, 41]),
            (7, 1, 2, 1, &[110, 139]),
            (13, 1, 3, 1, &[99, 67, 172]),
            (100, 1, 7, 1, &[138, 73, 167, 186, 95, 140, 112]),
            (8, 1, 5, 1, &[84, 156, 44, 27, 210]),
            (8, 1, 6, 1, &[136, 171, 210, 132, 133, 124]),
            (
                16,
                16,
                5,
                3,
                &[
                    153, 129, 113, 113, 123, 121, 133, 140, 138, 128, 152, 132, 117, 111, 133,
                ],
            ),
            (
                15,
                9,
                4,
                4,
                &[
                    129, 149, 151, 106, 109, 145, 159, 131, 143, 122, 119, 119, 129, 120, 121, 151,
                ],
            ),
            (
                64,
                4,
                4,
                4,
                &[
                    131, 141, 149, 113, 134, 132, 147, 113, 141, 113, 144, 112, 145, 103, 143, 112,
                ],
            ),
            // Upscaled across, downscaled down: still mipmapped.
            (
                4,
                64,
                8,
                4,
                &[
                    115, 115, 115, 116, 116, 117, 117, 117, 137, 137, 138, 140, 142, 144, 145, 145,
                    140, 140, 128, 106, 84, 62, 51, 51, 122, 122, 127, 135, 143, 151, 155, 155,
                ],
            ),
        ];
        for (width, height, w, h, chrome) in cases {
            let image = alpha_image(width, height, &mut stream);
            assert_near(&draw(&image, w, h), chrome);
        }
    }

    /// Chrome 154: a whole-pixel slice is mipmapped as its own subset, while a
    /// source the crop leaves fractional is not. A second probe page drew the
    /// slice first, then 15×8, 31×8, 9×4, 27×12, 50×9, 64×37, 16×8 and 16×1
    /// images (3810 alphas with the slice's) before the cropped row.
    #[test]
    fn mipmaps_start_at_the_subset_and_fractional_crops_skip_them() {
        // A second probe page, whose stream starts over.
        let mut stream = probe_alphas();
        // A 15x8 image sliced to its right eight columns, drawn 4x4.
        let image = alpha_image(15, 8, &mut stream);
        let square = Rect::new(0.0, 0.0, 4.0, 4.0);
        let placement = [Rect::new(7.0, 0.0, 15.0, 8.0), square, square];
        assert_near(
            &alphas(&render(&image, placement, 4, 4)),
            &[
                144, 108, 102, 36, 81, 113, 144, 141, 112, 118, 158, 113, 128, 194, 123, 114,
            ],
        );
        // A 16x1 row placed at -2..3, cropped to 0..3: the source becomes
        // 6.4..16, drawn with 4-bit weights and no mipmaps.
        let mut stream = probe_alphas().skip(3810);
        let image = alpha_image(16, 1, &mut stream);
        let placement = [
            Rect::new(0.0, 0.0, 16.0, 1.0),
            Rect::new(-2.0, 0.0, 3.0, 1.0),
            Rect::new(0.0, 0.0, 3.0, 1.0),
        ];
        assert_eq!(alphas(&render(&image, placement, 3, 1)), [118, 69, 72]);
    }

    /// A 2048x2048 image, red on its left half, drawn into an 8x8 raster from
    /// -0.8 to 8.8 as Chrome's default filter region places it: only the two
    /// coarse levels' texels are kept, and the edge softens as Chrome's does.
    #[test]
    fn large_downscales_keep_only_the_levels_they_read() {
        struct Halves;
        impl ImageRows for Halves {
            fn read(&self, x0: u32, _y: u32, row: &mut [PremulRgba8]) -> bool {
                for (x, texel) in (x0..).zip(row.iter_mut()) {
                    let a = if x < 1024 { 255 } else { 0 };
                    *texel = PremulRgba8 {
                        r: a,
                        g: 0,
                        b: 0,
                        a,
                    };
                }
                true
            }
        }
        let region = Rect::new(-0.8, -0.8, 8.8, 8.8);
        let raster = Rect::new(0.0, 0.0, 8.0, 8.0);
        let source = Rect::new(0.0, 0.0, 2048.0, 2048.0);
        let plan = FilterImage::plan([2048, 2048], source, region, region, raster).unwrap();
        let image = plan.share(Arc::new(Halves)).unwrap();
        assert_eq!(image.sampling().mipmap.map(|m| m.level), Some(7));
        assert!(image.retained_texels() <= 16 * 16 + 8 * 8);
        let mut output = Pixmap::new(8, 8);
        draw(
            &mut output,
            &image,
            PixelBounds::of(raster, Point::ORIGIN, 8, 8),
            Point::ORIGIN,
        );
        assert_near(&alphas(&output)[..8], &[255, 255, 255, 250, 5, 0, 0, 0]);
    }
}
