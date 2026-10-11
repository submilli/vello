// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! How Chrome's Skia backend samples an `feImage` raster, decided once per draw.
//!
//! Blink's `FEImage::CreateImageFilter` draws only the part of the placed image
//! within the primitive's crop (its subregion within the filter region): the
//! crop becomes the destination and is mapped back onto the source. Skia's
//! `FilterResult::MakeFromImage` clips that source to the image, moving the
//! destination with it. Chrome's high filter quality then samples by scale
//! (`PaintFlags::FilterQualityToSkSamplingOptions`):
//!
//! - Upscaled on both axes, Mitchell bicubic (B = C = 1/3).
//! - Otherwise linear with linear mipmaps. A whole-pixel source becomes an
//!   image subset, sampled trilinearly from the subset's mipmaps once the draw
//!   downscales enough. A fractional source is drawn with `drawImageRect`,
//!   whose raster bilinear path takes no mipmaps and quantizes subpixel weights
//!   to sixteenths.
//!
//! Skia picks the mip level from the geometric mean of the two axis scales,
//! sharpened by half a level (`SkMipmap::ComputeLevel`), and blends the two
//! levels around it (`SkMipmapAccessor`). Texels clamp to the sampled subset,
//! at every level.

use super::graph::valid_rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Point, Rect};

/// A half-open rectangle of whole texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelRect {
    /// First column.
    pub x0: u32,
    /// First row.
    pub y0: u32,
    /// Column past the last.
    pub x1: u32,
    /// Row past the last.
    pub y1: u32,
}

impl TexelRect {
    /// Columns covered.
    pub fn width(&self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Rows covered.
    pub fn height(&self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }

    /// Texels covered.
    pub fn area(&self) -> u64 {
        u64::from(self.width()) * u64::from(self.height())
    }

    /// Whether `(x, y)` lies inside.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        (self.x0..self.x1).contains(&x) && (self.y0..self.y1).contains(&y)
    }

    /// The texels as a rectangle in texel coordinates.
    pub fn to_rect(&self) -> Rect {
        Rect::new(
            f64::from(self.x0),
            f64::from(self.y0),
            f64::from(self.x1),
            f64::from(self.y1),
        )
    }

    /// The smallest rectangle containing both; an empty one is ignored.
    pub fn union(&self, other: Self) -> Self {
        if other.area() == 0 {
            return *self;
        }
        if self.area() == 0 {
            return other;
        }
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

/// The texel filter Skia applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFilter {
    /// Exact bilinear weights, from mipmaps when downscaled.
    Bilinear,
    /// Skia's raster `drawImageRect` path: bilinear with 4-bit weights.
    QuantizedBilinear,
    /// Mitchell–Netravali, B = C = 1/3.
    Bicubic,
}

/// The mip levels a trilinear draw blends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mipmap {
    /// The finer level; level 0 is the subset itself.
    pub level: u8,
    /// The weight of the next coarser level, in `[0, 1)`.
    pub weight: f32,
}

/// Chrome's sampling of one image draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// The sampled rectangle in image texels, within the image.
    pub source: Rect,
    /// Where `source` lands, in filter coordinates, within the crop.
    pub destination: Rect,
    /// `source` rounded out to whole texels: reads clamp to it, and it is the
    /// base of the mipmaps.
    pub subset: TexelRect,
    /// The texel filter.
    pub filter: ImageFilter,
    /// The levels a whole-pixel downscale blends, if it uses mipmaps.
    pub mipmap: Option<Mipmap>,
}

impl Sampling {
    /// Chrome's sampling of a `size` image whose `source` (in texels) is placed
    /// at `destination`, drawn within `crop` (both in filter coordinates).
    /// `None` when nothing of the image reaches the crop, or the placement is
    /// not finite.
    pub fn new(size: [u32; 2], source: Rect, destination: Rect, crop: Rect) -> Option<Self> {
        if !positive(source) || !positive(destination) || !valid_rect(crop) {
            return None;
        }
        let scale = [
            destination.width() / source.width(),
            destination.height() / source.height(),
        ];
        if !scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            return None;
        }
        // Blink's `ImagePaintFilter` decides the upscale on the rectangles it is
        // given; cropping keeps their scale.
        let upscaled = scale[0] > 1.0 && scale[1] > 1.0;
        let cropped = destination.intersect(crop);
        if !positive(cropped) {
            return None;
        }
        // Blink and Skia keep rectangles in single precision, so whether the
        // source is whole and the subset it rounds out to both follow it.
        let shown = if cropped == destination {
            single(source)
        } else {
            blink_map(cropped, destination, source)
        };
        let image = Rect::new(0.0, 0.0, f64::from(size[0]), f64::from(size[1]));
        let clipped = shown.intersect(image);
        let moved = map(clipped, shown, cropped);
        if !positive(clipped) || !positive(moved) {
            return None;
        }
        let (source, destination) = (clipped, moved);
        let subset = round_out(source, size);
        let filter = if upscaled {
            ImageFilter::Bicubic
        } else if whole(source) {
            ImageFilter::Bilinear
        } else {
            ImageFilter::QuantizedBilinear
        };
        let mipmap = match filter {
            ImageFilter::Bilinear => mipmap(scale, [subset.width(), subset.height()]),
            ImageFilter::QuantizedBilinear | ImageFilter::Bicubic => None,
        };
        Some(Self {
            source,
            destination,
            subset,
            filter,
            mipmap,
        })
    }

    /// The levels sampled with their weights, finest first: the subset alone,
    /// or one or two mip levels.
    pub fn levels(&self) -> impl Iterator<Item = (u8, f32)> {
        let (first, second) = match self.mipmap {
            None => ((0, 1.0), None),
            Some(Mipmap { level, weight }) => (
                (level, 1.0 - weight),
                (weight > 0.0).then_some((level + 1, weight)),
            ),
        };
        core::iter::once(first).chain(second)
    }

    /// The size of mip `level` of the subset, as `SkMipmap` halves it.
    pub fn level_size(&self, level: u8) -> [u32; 2] {
        level_size([self.subset.width(), self.subset.height()], level)
    }

    /// Where the filter-space `point` samples mip `level`, in that level's
    /// texels relative to the subset, with texel centers at whole numbers.
    pub fn level_point(&self, level: u8, point: Point) -> Point {
        let [width, height] = self.level_size(level);
        let kx = f64::from(width) / f64::from(self.subset.width());
        let ky = f64::from(height) / f64::from(self.subset.height());
        let u = self.source.x0
            + (point.x - self.destination.x0) * self.source.width() / self.destination.width();
        let v = self.source.y0
            + (point.y - self.destination.y0) * self.source.height() / self.destination.height();
        Point::new(
            (u - f64::from(self.subset.x0)) * kx - 0.5,
            (v - f64::from(self.subset.y0)) * ky - 0.5,
        )
    }

    /// The texels of mip `level` that pixels overlapping `area` (in filter
    /// coordinates) read, or `None` when no such pixel shows the image.
    pub fn window(&self, level: u8, area: Rect) -> Option<TexelRect> {
        // The centers of pixels overlapping both the area and the
        // destination, wherever the raster's grid lies: within half a pixel
        // of each. A pixel straddling their shared edge counts.
        let destination = self.destination;
        let x0 = area.x0.max(destination.x0) - 0.5;
        let y0 = area.y0.max(destination.y0) - 0.5;
        let x1 = area.x1.min(destination.x1) + 0.5;
        let y1 = area.y1.min(destination.y1) + 0.5;
        if !(x1 > x0 && y1 > y0) {
            return None;
        }
        let first = self.level_point(level, Point::new(x0, y0));
        let last = self.level_point(level, Point::new(x1, y1));
        // Taps beyond the floor of a coordinate, as an inclusive start and an
        // exclusive end: bicubic reads floor - 1 to floor + 2, bilinear floor
        // and floor + 1. Clamping first keeps huge coordinates in range.
        let (before, after) = match self.filter {
            ImageFilter::Bicubic => (1.0, 3.0),
            ImageFilter::Bilinear | ImageFilter::QuantizedBilinear => (0.0, 2.0),
        };
        let [width, height] = self.level_size(level);
        let span = |from: f64, to: f64, size: u32| {
            let size = f64::from(size);
            let start = (from.clamp(-4.0, size + 4.0).floor() - before).clamp(0.0, size);
            let end = (to.clamp(-4.0, size + 4.0).floor() + after).clamp(0.0, size);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "Both ends are clamped to the level, whose sides are u32."
            )]
            (start as u32, end as u32)
        };
        let (x0, x1) = span(first.x, last.x, width);
        let (y0, y1) = span(first.y, last.y, height);
        let window = TexelRect { x0, y0, x1, y1 };
        (window.area() > 0).then_some(window)
    }
}

/// Whether `rect` is finite with a positive area. Extreme scales can leave
/// mapped rectangles infinite or NaN.
fn positive(rect: Rect) -> bool {
    [rect.x0, rect.y0, rect.x1, rect.y1]
        .iter()
        .all(|v| v.is_finite())
        && rect.width() > 0.0
        && rect.height() > 0.0
}

/// `rect` rounded to single precision.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Rounding to single precision is the point."
)]
fn single(rect: Rect) -> Rect {
    let round = |v: f64| f64::from(v as f32);
    Rect::new(
        round(rect.x0),
        round(rect.y0),
        round(rect.x1),
        round(rect.y1),
    )
}

/// `rect`, in the space of `from`, mapped onto `to` as Blink's `gfx::MapRect`
/// does: in single precision, as an origin and a size, so the far edge is the
/// origin plus the mapped size. Whether a cut lands on a whole texel depends
/// on that rounding.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Blink computes in single precision."
)]
fn blink_map(rect: Rect, from: Rect, to: Rect) -> Rect {
    let f = |v: f64| v as f32;
    let axis = |r0: f64, r1: f64, from0: f64, from1: f64, to0: f64, to1: f64| {
        let scale = (f(to1) - f(to0)) / (f(from1) - f(from0));
        let origin = f(to0) + (f(r0) - f(from0)) * scale;
        let size = (f(r1) - f(r0)) * scale;
        (f64::from(origin), f64::from(origin + size))
    };
    let (x0, x1) = axis(rect.x0, rect.x1, from.x0, from.x1, to.x0, to.x1);
    let (y0, y1) = axis(rect.y0, rect.y1, from.y0, from.y1, to.y0, to.y1);
    Rect::new(x0, y0, x1, y1)
}

/// `rect`, in the space of `from`, mapped onto `to`.
fn map(rect: Rect, from: Rect, to: Rect) -> Rect {
    let sx = to.width() / from.width();
    let sy = to.height() / from.height();
    Rect::new(
        to.x0 + (rect.x0 - from.x0) * sx,
        to.y0 + (rect.y0 - from.y0) * sy,
        to.x0 + (rect.x1 - from.x0) * sx,
        to.y0 + (rect.y1 - from.y0) * sy,
    )
}

/// Whether Skia keeps `source` as an image subset: its edges are whole texels
/// in single precision, as Skia stores rectangles.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Skia compares the rectangle in single precision."
)]
fn whole(source: Rect) -> bool {
    [source.x0, source.y0, source.x1, source.y1]
        .into_iter()
        .map(|v| v as f32)
        .all(|v| v == v.round())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "The rectangle lies within the image, whose sides are u32."
)]
fn round_out(source: Rect, size: [u32; 2]) -> TexelRect {
    let side = |v: f64, max: u32| v.clamp(0.0, f64::from(max)) as u32;
    TexelRect {
        x0: side(source.x0.floor(), size[0]),
        y0: side(source.y0.floor(), size[1]),
        x1: side(source.x1.ceil(), size[0]),
        y1: side(source.y1.ceil(), size[1]),
    }
}

/// The mipmaps a whole-pixel draw at `scale` blends, for a subset of `size`.
fn mipmap(scale: [f64; 2], size: [u32; 2]) -> Option<Mipmap> {
    let mean = (scale[0] * scale[1]).sqrt();
    if mean >= 1.0 {
        return None;
    }
    let level = -mean.log2() - 0.5;
    let coarsest = level_count(size);
    if level.is_nan() || level <= 0.0 || coarsest == 0 {
        return None;
    }
    let finer = level.floor();
    // Past the coarsest level, a draw samples it alone.
    if finer >= f64::from(coarsest) {
        return Some(Mipmap {
            level: coarsest,
            weight: 0.0,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "The level is below the coarsest, at most 32."
    )]
    Some(Mipmap {
        level: finer as u8,
        weight: (level - finer) as f32,
    })
}

/// Levels below the subset: halvings until it is one texel.
fn level_count(size: [u32; 2]) -> u8 {
    let mut count = 0;
    let mut size = size;
    while size[0] > 1 || size[1] > 1 {
        size = halve(size);
        count += 1;
    }
    count
}

/// The size of mip `level` of a subset of `size`.
pub(crate) fn level_size(size: [u32; 2], level: u8) -> [u32; 2] {
    (0..level).fold(size, |size, _| halve(size))
}

/// The next level's size: sides halve, rounding down, to at least one texel.
fn halve(size: [u32; 2]) -> [u32; 2] {
    size.map(|side| (side >> 1).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDE: Rect = Rect::new(-1e5, -1e5, 1e5, 1e5);

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Rect {
        Rect::new(x0, y0, x1, y1)
    }

    #[test]
    fn scale_and_source_pick_the_filter() {
        let source = rect(0.0, 0.0, 4.0, 4.0);
        let filter = |source, destination| {
            Sampling::new([4, 4], source, destination, WIDE).map(|s| (s.filter, s.mipmap))
        };
        assert_eq!(
            filter(source, rect(0.0, 0.0, 8.0, 8.0)),
            Some((ImageFilter::Bicubic, None))
        );
        // Upscaled along one axis only: bilinear.
        assert_eq!(
            filter(source, rect(0.0, 0.0, 8.0, 4.0)),
            Some((ImageFilter::Bilinear, None))
        );
        assert_eq!(
            filter(rect(0.0, 0.5, 4.0, 3.5), rect(0.0, 0.0, 3.0, 3.0)),
            Some((ImageFilter::QuantizedBilinear, None))
        );
        // A quarter scale: level 1.5.
        let quarter = filter(source, rect(0.0, 0.0, 1.0, 1.0)).unwrap();
        assert_eq!(
            quarter,
            (
                ImageFilter::Bilinear,
                Some(Mipmap {
                    level: 1,
                    weight: 0.5
                })
            )
        );
        // Within half a level of the image, no mipmaps.
        assert_eq!(
            filter(source, rect(0.0, 0.0, 3.0, 3.0)),
            Some((ImageFilter::Bilinear, None))
        );
    }

    #[test]
    fn levels_stop_at_one_texel() {
        let sampling = Sampling::new(
            [4, 2],
            rect(0.0, 0.0, 4.0, 2.0),
            rect(0.0, 0.0, 0.01, 0.01),
            WIDE,
        )
        .unwrap();
        assert_eq!(
            sampling.mipmap,
            Some(Mipmap {
                level: 2,
                weight: 0.0
            })
        );
        assert_eq!(
            sampling.levels().collect::<alloc::vec::Vec<_>>(),
            [(2, 1.0)]
        );
        assert_eq!(sampling.level_size(1), [2, 1]);
        assert_eq!(sampling.level_size(2), [1, 1]);
    }

    /// Blink maps the crop back onto the source, and Skia clips that to the
    /// image: whether the source is whole is decided after both.
    #[test]
    fn the_crop_and_the_image_bound_the_source() {
        let row = rect(0.0, 0.0, 16.0, 1.0);
        let placed = rect(-2.0, 0.0, 3.0, 1.0);
        let cropped = Sampling::new([16, 1], row, placed, rect(0.0, 0.0, 3.0, 1.0)).unwrap();
        assert_eq!(cropped.source.x0, f64::from(6.4_f32));
        assert_eq!(cropped.destination, rect(0.0, 0.0, 3.0, 1.0));
        assert_eq!(cropped.filter, ImageFilter::QuantizedBilinear);
        assert_eq!(
            cropped.subset,
            TexelRect {
                x0: 6,
                y0: 0,
                x1: 16,
                y1: 1
            }
        );
        // A source beyond the image moves the destination with the clip.
        let beyond = Sampling::new(
            [4, 1],
            rect(-4.0, 0.0, 4.0, 1.0),
            rect(0.0, 0.0, 8.0, 1.0),
            WIDE,
        )
        .unwrap();
        assert_eq!(beyond.source, rect(0.0, 0.0, 4.0, 1.0));
        assert_eq!(beyond.destination, rect(4.0, 0.0, 8.0, 1.0));
        // A crop landing on a whole texel in Blink's single-precision mapping
        // keeps the source whole: 11/22 of 60 texels is 30.
        let whole = Sampling::new(
            [60, 1],
            rect(0.0, 0.0, 60.0, 1.0),
            rect(-11.0, 0.0, 11.0, 1.0),
            rect(0.0, 0.0, 11.0, 1.0),
        )
        .unwrap();
        assert_eq!((whole.subset.x0, whole.filter), (30, ImageFilter::Bilinear));
        // Blink's single-precision arithmetic lands 11/22 of 26 texels just
        // short of 13, so Chrome draws that source as fractional.
        let near = Sampling::new(
            [26, 1],
            rect(0.0, 0.0, 26.0, 1.0),
            rect(-11.0, 0.0, 11.0, 1.0),
            rect(0.0, 0.0, 11.0, 1.0),
        )
        .unwrap();
        assert_eq!(
            (near.subset.x0, near.filter),
            (12, ImageFilter::QuantizedBilinear)
        );
        // An uncut slice whose double-precision edge falls just short of a
        // texel is judged and rounded out alike: a 3x3 image sliced into
        // 1.1 x 3.3 shows texel column 1 alone.
        let slice = crate::filter::image::AspectRatio {
            slice: true,
            ..Default::default()
        };
        let (source, destination) = slice.fit(rect(0.0, 0.0, 3.0, 3.0), rect(0.0, 0.0, 1.1, 3.3));
        let sliced = Sampling::new([3, 3], source, destination, WIDE).unwrap();
        assert_eq!((sliced.subset.x0, sliced.subset.x1), (1, 2));
        // Nothing shows outside the crop or the image.
        assert!(Sampling::new([4, 1], row, placed, rect(5.0, 0.0, 6.0, 1.0)).is_none());
        assert!(Sampling::new([4, 1], rect(5.0, 0.0, 6.0, 1.0), placed, WIDE).is_none());
    }

    #[test]
    fn hostile_placements_are_refused() {
        let unit = rect(0.0, 0.0, 1.0, 1.0);
        for (source, destination, crop) in [
            (rect(0.0, 0.0, f64::NAN, 1.0), unit, WIDE),
            (unit, rect(0.0, 0.0, f64::INFINITY, 1.0), WIDE),
            (unit, unit, rect(0.0, 0.0, 2e6, 1.0)),
            (rect(0.0, 0.0, 0.0, 1.0), unit, WIDE),
            // Reversed on both axes, with a positive area.
            (rect(1.0, 1.0, 0.0, 0.0), rect(1.0, 1.0, 0.0, 0.0), WIDE),
        ] {
            let sampling = Sampling::new([1, 1], source, destination, crop);
            assert!(
                sampling.is_none(),
                "{source:?} {destination:?} {crop:?}: {sampling:?}"
            );
        }
    }

    #[test]
    fn windows_cover_the_taps_of_overlapping_pixels() {
        let sampling = Sampling::new(
            [16, 1],
            rect(0.0, 0.0, 16.0, 1.0),
            rect(0.0, 0.0, 16.0, 1.0),
            WIDE,
        )
        .unwrap();
        let window = |x0, x1| sampling.window(0, rect(x0, 0.0, x1, 1.0));
        // Pixels overlapping 4..6 wherever the grid lies have centers in
        // 3.5..6.5, reading texels 3 to 7.
        assert_eq!(
            window(4.0, 6.0),
            Some(TexelRect {
                x0: 3,
                y0: 0,
                x1: 8,
                y1: 1
            })
        );
        assert_eq!(window(20.0, 30.0), None);
        // A pixel straddling the destination's edge and the area's still reads.
        let half = Sampling::new(
            [16, 1],
            rect(0.0, 0.0, 16.0, 1.0),
            rect(0.0, 0.0, 2.5, 1.0),
            WIDE,
        )
        .unwrap();
        assert!(half.window(0, rect(2.5, 0.0, 5.0, 1.0)).is_some());
        // A quarter-scale draw reads level 1 and level 2 near its pixels.
        let quarter = Sampling::new(
            [64, 1],
            rect(0.0, 0.0, 64.0, 1.0),
            rect(0.0, 0.0, 16.0, 1.0),
            WIDE,
        )
        .unwrap();
        let area = rect(2.0, 0.0, 3.0, 1.0);
        let level = |level| quarter.window(level, area).map(|w| (w.x0, w.x1));
        // Centers 1.5..3.5 sample image columns 6..14: level 1 at 2.5..6.5,
        // level 2 at 1..3, each with the next texel.
        assert_eq!(level(1), Some((2, 8)));
        assert_eq!(level(2), Some((1, 5)));
    }
}
