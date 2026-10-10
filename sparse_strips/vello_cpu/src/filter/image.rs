// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sampling of caller-supplied `feImage` pixels, as Chrome's Skia backend does.
//!
//! Chrome's high filter quality samples a source upscaled on both axes with
//! Mitchell bicubic (B = C = 1/3). Otherwise Skia's `FilterResult::MakeFromImage`
//! keeps a whole-pixel source as a transformed image, sampled with exact
//! bilinear weights, and draws a fractional source with `drawImageRect`, whose
//! bilinear raster path quantizes subpixel weights to sixteenths. Either way
//! Skia samples a subset of the source rounded out to whole texels, so texels
//! clamp to it; the destination's edges are anti-aliased by coverage.
use super::bounds::PixelBounds;
use super::channels::TRANSPARENT;
use vello_common::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

/// One image node: where its pixels come from and go.
pub(super) struct Placement<'a> {
    pub image: &'a Pixmap,
    pub source: Rect,
    pub destination: Rect,
}

/// Replace `output` with the image drawn inside `crop`; elsewhere it is transparent.
pub(super) fn draw(
    output: &mut Pixmap,
    placement: &Placement<'_>,
    crop: PixelBounds,
    origin: Point,
) {
    output.data_mut().fill(TRANSPARENT);
    if let Some(sampler) = Sampler::new(placement) {
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Filter {
    Bilinear,
    /// Skia's raster `drawImageRect` path: bilinear with 4-bit weights.
    QuantizedBilinear,
    /// Mitchell–Netravali, B = C = 1/3.
    Bicubic,
}

/// Maps destination pixels to image texels.
struct Sampler<'a> {
    image: &'a Pixmap,
    /// Whole texels sampled; reads outside clamp to it.
    subset: [i64; 4],
    source: Rect,
    destination: Rect,
    scale: [f64; 2],
    filter: Filter,
}

impl<'a> Sampler<'a> {
    /// `None` when nothing of the image reaches the destination.
    fn new(placement: &Placement<'a>) -> Option<Self> {
        let image = placement.image;
        let bounds = Rect::new(
            0.0,
            0.0,
            f64::from(image.width()),
            f64::from(image.height()),
        );
        let (source, destination) = (placement.source, placement.destination);
        if source.width() <= 0.0 || source.height() <= 0.0 || destination.area() <= 0.0 {
            return None;
        }
        let scale = [
            destination.width() / source.width(),
            destination.height() / source.height(),
        ];
        // Skia first clips the source to the image, moving the destination with it.
        let clipped = source.intersect(bounds);
        if clipped.width() <= 0.0 || clipped.height() <= 0.0 {
            return None;
        }
        let destination = Rect::new(
            destination.x0 + (clipped.x0 - source.x0) * scale[0],
            destination.y0 + (clipped.y0 - source.y0) * scale[1],
            destination.x0 + (clipped.x1 - source.x0) * scale[0],
            destination.y0 + (clipped.y1 - source.y0) * scale[1],
        );
        let whole = [clipped.x0, clipped.y0, clipped.x1, clipped.y1]
            .into_iter()
            .all(|v| v == v.round());
        let filter = match (whole, scale[0] > 1.0 && scale[1] > 1.0) {
            (_, true) => Filter::Bicubic,
            (true, false) => Filter::Bilinear,
            (false, false) => Filter::QuantizedBilinear,
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The subset lies within the image, whose sides are u16."
        )]
        let subset = {
            let texels = clipped.expand();
            [texels.x0, texels.y0, texels.x1, texels.y1].map(|v| v as i64)
        };
        Some(Self {
            image,
            subset,
            source: clipped,
            destination,
            scale,
            filter,
        })
    }

    /// The pixel whose top-left corner is `pixel`, in filter coordinates.
    fn pixel(&self, pixel: Point) -> PremulRgba8 {
        let coverage = coverage(self.destination, pixel);
        if coverage <= 0.0 {
            return TRANSPARENT;
        }
        // Texel space puts texel centers at half-integers.
        let u = self.source.x0 + (pixel.x + 0.5 - self.destination.x0) / self.scale[0] - 0.5;
        let v = self.source.y0 + (pixel.y + 0.5 - self.destination.y0) / self.scale[1] - 0.5;
        // Texels clamp to the subset, so coordinates beyond the furthest tap
        // (two texels for bicubic) sample the same; clamping first keeps tiny
        // destinations from overflowing the integer texel arithmetic.
        let [x0, y0, x1, y1] = self.subset.map(|v| v as f64);
        let u = u.clamp(x0 - 2.0, x1 + 2.0);
        let v = v.clamp(y0 - 2.0, y1 + 2.0);
        let channels = match self.filter {
            Filter::Bilinear => self.bilinear(u, v),
            Filter::QuantizedBilinear => self.quantized_bilinear(u, v),
            Filter::Bicubic => self.bicubic(u, v),
        };
        premultiplied(channels, coverage)
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "Texels clamp into the subset, which is within the image."
    )]
    fn texel(&self, x: i64, y: i64) -> [f32; 4] {
        let [x0, y0, x1, y1] = self.subset;
        let x = x.clamp(x0, x1 - 1) as usize;
        let y = y.clamp(y0, y1 - 1) as usize;
        let p = self.image.data()[y * usize::from(self.image.width()) + x];
        [p.r, p.g, p.b, p.a].map(f32::from)
    }

    fn bilinear(&self, u: f64, v: f64) -> [f32; 4] {
        let (x, y) = (u.floor(), v.floor());
        let (fx, fy) = ((u - x) as f32, (v - y) as f32);
        self.weighted(x as i64, y as i64, [1.0 - fx, fx], [1.0 - fy, fy])
    }

    /// Skia's raster bilinear keeps four fractional bits of each coordinate.
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
    use alloc::vec::Vec;

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

    fn row(image: &Pixmap, source: Rect, destination: Rect, width: u16) -> Vec<u8> {
        let mut output = Pixmap::new(width, 1);
        let placement = Placement {
            image,
            source,
            destination,
        };
        let crop = PixelBounds::of(
            Rect::new(0.0, 0.0, f64::from(width), 1.0),
            Point::ORIGIN,
            width,
            1,
        );
        draw(&mut output, &placement, crop, Point::ORIGIN);
        output.data().iter().map(|p| p.a).collect()
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
        let mut output = Pixmap::new(8, 4);
        let placement = Placement {
            image: &image,
            source: Rect::new(0.0, 0.0, 2.0, 2.0),
            destination: Rect::new(4.0, 0.0, 8.0, 4.0),
        };
        let crop = PixelBounds::of(Rect::new(0.0, 0.0, 8.0, 4.0), Point::ORIGIN, 8, 4);
        draw(&mut output, &placement, crop, Point::ORIGIN);
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
        let mut output = Pixmap::new(3, 4);
        let placement = Placement {
            image: &image,
            source: Rect::new(0.0, 0.0, 2.0, 2.0 / 3.0),
            destination: Rect::new(1.0, 1.0, 7.0, 3.0),
        };
        let crop = PixelBounds::of(Rect::new(0.0, 0.0, 3.0, 4.0), Point::ORIGIN, 3, 4);
        draw(&mut output, &placement, crop, Point::ORIGIN);
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
}
