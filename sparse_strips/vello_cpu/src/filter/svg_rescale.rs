// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Gaussian blurs beyond the three-box range, rescaled as Chrome (Skia) does.
//!
//! Skia's raster box blur handles deviations up to 135. For larger ones,
//! `FilterResult::Builder::blur` scales the input by `135 / deviation` through
//! `FilterResult::rescale`: halvings and then one final step landing on that
//! scale, each a bilinear draw about the center of the input's layer bounds
//! with transparent edges. It blurs that low-resolution image by the deviation
//! mapped into it, at most 135, and draws the result back with bilinear
//! filtering.
//! See: <https://skia.googlesource.com/skia/+/main/src/core/SkImageFilterTypes.cpp>
use super::bounds::PixelBounds;
use super::channels::{TRANSPARENT, quantize};
use super::svg_gaussian::BOX_SIGMA;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

/// Blur `pixels` (row-major, nonempty, `width > 0` columns) by deviations
/// that may exceed [`BOX_SIGMA`], centered on `layer`, the pixels the input may
/// hold. `box_blur` blurs the low-resolution image by deviations of at most
/// [`BOX_SIGMA`]. Returns `false`, leaving `pixels` unchanged, when Skia would
/// not rescale: then a box blur by the deviations clamped to [`BOX_SIGMA`] is
/// Chrome's result.
pub(super) fn try_blur(
    pixels: &mut [PremulRgba8],
    width: usize,
    std_deviation: [f32; 2],
    layer: PixelBounds,
    box_blur: impl Fn(&mut [PremulRgba8], usize, [f32; 2]),
) -> bool {
    let height = pixels.len() / width;
    let x = Axis::new(std_deviation[0], layer.x0, layer.x1);
    let y = Axis::new(std_deviation[1], layer.y0, layer.y1);
    let (Some(mut x), Some(mut y)) = (x, y) else {
        return false;
    };
    if x.steps == 0 && y.steps == 0 {
        return false;
    }
    let mut image = Image {
        data: pixels.to_vec(),
        x0: 0,
        y0: 0,
        width,
        height,
    };
    while x.steps > 0 || y.steps > 0 {
        let (sx, sy) = (x.next_step(), y.next_step());
        image = downscale(&image, [&x, &y], [sx, sy]);
        x.commit(sx);
        y.commit(sy);
    }
    // The blur spreads past the low-resolution content, so it covers everything
    // the raster maps onto, plus a texel for bilinear sampling at the edges.
    let (bx0, bx1) = x.footprint(width);
    let (by0, by1) = y.footprint(height);
    let mut blurred = Image::transparent(bx0, by0, bx1 - bx0, by1 - by0);
    blurred.paste(&image);
    box_blur(
        &mut blurred.data,
        blurred.width,
        [x.low_sigma(), y.low_sigma()],
    );
    upscale(&blurred, pixels, width, [&x, &y]);
    true
}

/// One axis of the rescale, tracked in floats as Skia tracks `stepBoundsF`.
struct Axis {
    /// The deviation in raster pixels.
    sigma: f32,
    /// Remaining downscale steps.
    steps: u32,
    /// The net scale the last step reaches.
    scale: f32,
    /// The input's span in raster pixels (Skia's `srcRect`).
    source: (f32, f32),
    /// That span in the current low-resolution space.
    bounds: (f32, f32),
}

impl Axis {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Raster coordinates are below 2^16, exact in f32."
    )]
    fn new(sigma: f32, start: usize, end: usize) -> Option<Self> {
        if end <= start {
            return None;
        }
        let scale = if sigma > BOX_SIGMA {
            BOX_SIGMA / sigma
        } else {
            1.0
        };
        let source = (start as f32, end as f32);
        Some(Self {
            sigma,
            steps: step_count(scale),
            scale,
            source,
            bounds: source,
        })
    }

    /// This step's scale: halvings until the last, which lands on the net scale.
    fn next_step(&self) -> f32 {
        match self.steps {
            0 => 1.0,
            1 => (self.source.1 - self.source.0) * self.scale / (self.bounds.1 - self.bounds.0),
            _ => 0.5,
        }
    }

    fn commit(&mut self, scale: f32) {
        self.bounds = scale_about_center(self.bounds, scale);
        self.steps = self.steps.saturating_sub(1);
    }

    /// Raster coordinate to low-resolution coordinate.
    fn to_low(&self, coordinate: f64) -> f64 {
        map_span(coordinate, self.source, self.bounds)
    }

    /// Skia maps the deviation through the inverse of the low-resolution
    /// transform's scale and clamps rounding error back into the box range.
    fn low_sigma(&self) -> f32 {
        let upscale = (self.source.1 - self.source.0) / (self.bounds.1 - self.bounds.0);
        (self.sigma * (1.0 / upscale)).min(BOX_SIGMA)
    }

    /// The low-resolution pixels a raster `size` pixels long maps onto, with a
    /// texel of margin on both sides.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Raster sizes and their low-resolution coordinates are within 2^16."
    )]
    fn footprint(&self, size: usize) -> (i64, i64) {
        let start = self.to_low(0.0).floor() as i64 - 1;
        let end = self.to_low(size as f64).ceil() as i64 + 1;
        (start, end)
    }
}

/// Skia's `downscale_step_count`: steps for `scale` are `ceil(log2(1 / scale))`,
/// one fewer when the final step would be nearly the identity.
fn step_count(scale: f32) -> u32 {
    if scale >= 1.0 {
        return 0;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Deviations are capped, so the inverse scale is a small positive integer."
    )]
    let inverse = (1.0 / scale).ceil() as u32;
    let mut steps = inverse.next_power_of_two().trailing_zeros();
    if steps > 0 {
        let last = scale * (1_u32 << (steps - 1)) as f32;
        // Skia's `kNearIdentityLimit` (1 - 1e-3) for one step, `kMultiPassLimit` otherwise.
        let limit = if steps == 1 { 1.0 - 1e-3 } else { 0.9 };
        if last >= limit {
            steps -= 1;
        }
    }
    steps
}

/// Skia's `scale_about_center`: scale a span about its center and move that
/// center to the origin, so low-resolution coordinates may be negative. A span
/// that does not scale keeps its coordinates.
fn scale_about_center((start, end): (f32, f32), scale: f32) -> (f32, f32) {
    if scale == 1.0 {
        return (start, end);
    }
    let center = 0.5 * start + 0.5 * end;
    ((start - center) * scale, (end - center) * scale)
}

/// Map `position` from the span `from` onto the span `to`, exactly when they match.
fn map_span(position: f64, from: (f32, f32), to: (f32, f32)) -> f64 {
    if from == to {
        return position;
    }
    let (f0, f1) = (f64::from(from.0), f64::from(from.1));
    let (t0, t1) = (f64::from(to.0), f64::from(to.1));
    t0 + (position - f0) * (t1 - t0) / (f1 - f0)
}

/// An 8-bit premultiplied image placed at integer coordinates of its space;
/// everything outside it is transparent.
struct Image {
    data: Vec<PremulRgba8>,
    x0: i64,
    y0: i64,
    width: usize,
    height: usize,
}

impl Image {
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Callers pass nonnegative extents within the raster's 2^16 bound."
    )]
    fn transparent(x0: i64, y0: i64, width: i64, height: i64) -> Self {
        let (width, height) = (width.max(0) as usize, height.max(0) as usize);
        Self {
            data: vec![TRANSPARENT; width * height],
            x0,
            y0,
            width,
            height,
        }
    }

    fn texel(&self, x: i64, y: i64) -> [f32; 4] {
        let (Ok(column), Ok(row)) = (usize::try_from(x - self.x0), usize::try_from(y - self.y0))
        else {
            return [0.0; 4];
        };
        if column >= self.width || row >= self.height {
            return [0.0; 4];
        }
        let p = self.data[row * self.width + column];
        [p.r, p.g, p.b, p.a].map(f32::from)
    }

    /// Bilinear sample at a position of this image's space (texel centers at
    /// half-integers), reading transparency outside it.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Sample positions lie within the raster's 2^16 extent."
    )]
    fn bilinear(&self, x: f64, y: f64) -> PremulRgba8 {
        let (x, y) = (x - 0.5, y - 0.5);
        let (left, top) = (x.floor(), y.floor());
        let (tx, ty) = ((x - left) as f32, (y - top) as f32);
        let (left, top) = (left as i64, top as i64);
        let mut sum = [0.0_f32; 4];
        for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
            for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                let weight = wx * wy;
                if weight == 0.0 {
                    continue;
                }
                let texel = self.texel(left + dx, top + dy);
                for (s, t) in sum.iter_mut().zip(texel) {
                    *s += weight * t;
                }
            }
        }
        let [r, g, b, a] = sum.map(|v| quantize(v / 255.0));
        PremulRgba8 { r, g, b, a }
    }

    /// Copy `other` into this image where they overlap.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Offsets are measured from each image's origin inside the overlap."
    )]
    fn paste(&mut self, other: &Self) {
        let left = self.x0.max(other.x0);
        let right = (self.x0 + self.width as i64).min(other.x0 + other.width as i64);
        let top = self.y0.max(other.y0);
        let bottom = (self.y0 + self.height as i64).min(other.y0 + other.height as i64);
        if right <= left {
            return;
        }
        let length = (right - left) as usize;
        for y in top..bottom {
            let from = (y - other.y0) as usize * other.width + (left - other.x0) as usize;
            let to = (y - self.y0) as usize * self.width + (left - self.x0) as usize;
            self.data[to..to + length].copy_from_slice(&other.data[from..from + length]);
        }
    }
}

/// One rescale step: draw `image`, whose content spans the axes' current
/// bounds, into those bounds scaled about their centers.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Low-resolution coordinates lie within the raster's 2^16 extent."
)]
fn downscale(image: &Image, [x, y]: [&Axis; 2], [sx, sy]: [f32; 2]) -> Image {
    let (dx, dy) = (
        scale_about_center(x.bounds, sx),
        scale_about_center(y.bounds, sy),
    );
    // Skia rounds the destination out to whole pixels.
    let (x0, x1) = (dx.0.floor() as i64, dx.1.ceil() as i64);
    let (y0, y1) = (dy.0.floor() as i64, dy.1.ceil() as i64);
    let mut out = Image::transparent(x0, y0, x1 - x0, y1 - y0);
    for row in 0..out.height {
        let v = map_span((y0 + row as i64) as f64 + 0.5, dy, y.bounds);
        for column in 0..out.width {
            let u = map_span((x0 + column as i64) as f64 + 0.5, dx, x.bounds);
            out.data[row * out.width + column] = image.bilinear(u, v);
        }
    }
    out
}

/// Draw the blurred low-resolution image back over every raster pixel.
fn upscale(blurred: &Image, pixels: &mut [PremulRgba8], width: usize, [x, y]: [&Axis; 2]) {
    for (row, line) in pixels.chunks_exact_mut(width).enumerate() {
        let v = y.to_low(row as f64 + 0.5);
        for (column, pixel) in line.iter_mut().enumerate() {
            *pixel = blurred.bilinear(x.to_low(column as f64 + 0.5), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip() -> Vec<PremulRgba8> {
        (0..600)
            .map(|x| PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: if (200..400).contains(&x) { 255 } else { 0 },
            })
            .collect()
    }

    fn row(x0: usize, x1: usize) -> PixelBounds {
        PixelBounds {
            x0,
            y0: 0,
            x1,
            y1: 1,
        }
    }

    #[test]
    fn step_counts_follow_skia() {
        // No scale, a single step unless nearly the identity, then halvings.
        assert_eq!(step_count(1.0), 0);
        assert_eq!(step_count(BOX_SIGMA / 135.1), 0);
        assert_eq!(step_count(BOX_SIGMA / 136.0), 1);
        assert_eq!(step_count(BOX_SIGMA / 150.0), 1);
        // 0.45 needs two steps, but a final step of 0.9 meets the multi-pass
        // limit, so one step scales straight to 0.45.
        assert_eq!(step_count(BOX_SIGMA / 300.0), 1);
        assert_eq!(step_count(BOX_SIGMA / 300.5), 2);
        // The largest deviation Skia blurs, 532, takes two steps.
        assert_eq!(step_count(BOX_SIGMA / 532.0), 2);
    }

    #[test]
    fn scaling_about_the_center_keeps_unscaled_axes() {
        assert_eq!(scale_about_center((200.0, 400.0), 0.5), (-50.0, 50.0));
        assert_eq!(scale_about_center((200.0, 400.0), 1.0), (200.0, 400.0));
    }

    #[test]
    fn deviations_within_the_identity_limit_are_not_rescaled() {
        let mut pixels = strip();
        let unchanged = pixels.clone();
        let rescaled = try_blur(&mut pixels, 600, [135.1, 0.0], row(200, 400), |_, _, _| {
            panic!("no blur without a rescale")
        });
        assert!(!rescaled);
        assert_eq!(pixels, unchanged);
        // An empty layer has nothing to center a rescale on.
        assert!(!try_blur(
            &mut pixels,
            600,
            [300.0, 0.0],
            row(5, 5),
            |_, _, _| {}
        ));
    }

    #[test]
    fn pasting_copies_only_the_overlap() {
        let mut target = Image::transparent(-1, 0, 3, 1);
        let mut source = Image::transparent(0, 0, 4, 1);
        source.data.fill(PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 7,
        });
        target.paste(&source);
        assert_eq!(
            target.data.iter().map(|p| p.a).collect::<Vec<_>>(),
            [0, 7, 7]
        );
    }
}
