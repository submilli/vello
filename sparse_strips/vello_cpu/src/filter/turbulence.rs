// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feTurbulence`, following the specification's lattice as Chrome (Skia) evaluates it.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feTurbulenceElement>
use super::bounds::PixelBounds;
use super::channels::store;
use alloc::vec;
use vello_common::filter::graph::{ColorSpace, MAX_TURBULENCE_OCTAVES};
use vello_common::filter_effects::TurbulenceType;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::kurbo::{Point, Rect};
use vello_common::pixmap::Pixmap;

const BLOCK: usize = 256;
const RAND_MAX: i64 = 2_147_483_647;
/// The largest `f32` below 2^31.
const MAX_SEED: f32 = 2_147_483_520.0;

/// Noise parameters after admission; frequencies are finite and nonnegative.
pub(super) struct Turbulence {
    pub base_frequency: [f32; 2],
    pub num_octaves: u32,
    pub seed: f32,
    pub stitch_tiles: bool,
    pub kind: TurbulenceType,
}

/// Fill the pixels of `region` with noise at absolute filter coordinates.
/// Stitching tiles the noise with the primitive region's size from the origin.
pub(super) fn apply(
    pixels: &mut Pixmap,
    turbulence: &Turbulence,
    region: Rect,
    origin: Point,
    space: ColorSpace,
) {
    let width = usize::from(pixels.width());
    let bounds = PixelBounds::of(region, origin, pixels.width(), pixels.height());
    let lattice = Lattice::new(turbulence.seed);
    let (frequency, stitch) = stitching(turbulence, region);
    let octaves = turbulence.num_octaves.min(MAX_TURBULENCE_OCTAVES);
    let fractal = turbulence.kind == TurbulenceType::FractalNoise;
    let mut data = vec![[0.0_f32; 4]; pixels.data().len()];
    for y in bounds.y0..bounds.y1 {
        for x in bounds.x0..bounds.x1 {
            // Skia offsets the pixel center by another half pixel.
            let point = [
                (origin.x + x as f64 + 1.0) as f32,
                (origin.y + y as f64 + 1.0) as f32,
            ];
            let mut sum = lattice.sum(point, frequency, octaves, fractal, stitch);
            if fractal {
                sum = sum.map(|v| v * 0.5 + 0.5);
            }
            // The unclamped alpha scales color, as in Skia.
            let rgb = [0, 1, 2].map(|i| sum[i].clamp(0.0, 1.0) * sum[3]);
            data[y * width + x] = [rgb[0], rgb[1], rgb[2], sum[3].clamp(0.0, 1.0)];
        }
    }
    store(pixels, &data, space);
}

/// Stitching frequencies fit a whole number of lattice cells into the region.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Region sizes are bounded by the raster admission."
)]
fn stitching(turbulence: &Turbulence, region: Rect) -> ([f32; 2], Option<[f32; 2]>) {
    let mut frequency = turbulence.base_frequency;
    if !turbulence.stitch_tiles {
        return (frequency, None);
    }
    // Chrome truncates the region to whole pixels before rounding the tile.
    let size = [
        region.width().trunc() as f32,
        region.height().trunc() as f32,
    ];
    if size[0] <= 0.0 || size[1] <= 0.0 {
        return (frequency, None);
    }
    for axis in 0..2 {
        let f = frequency[axis];
        if f == 0.0 {
            continue;
        }
        let low = (size[axis] * f).floor() / size[axis];
        let high = (size[axis] * f).ceil() / size[axis];
        frequency[axis] = if f / low < high / f { low } else { high };
    }
    let stitch = [
        (size[0] * frequency[0]).round(),
        (size[1] * frequency[1]).round(),
    ];
    (frequency, Some(stitch))
}

/// The seeded permutation and 16-bit quantized per-channel gradients.
struct Lattice {
    selector: [u8; BLOCK],
    gradient: [[[f32; 2]; BLOCK]; 4],
}

impl Lattice {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "The seed is truncated and clamped; random values are positive modulo BLOCK."
    )]
    fn new(seed: f32) -> Self {
        // The specification truncates the seed rather than rounding it. Like Chrome,
        // saturate to the largest float magnitude below 2^31 first.
        let mut seed = f64::from(seed.trunc().clamp(-MAX_SEED, MAX_SEED)) as i64;
        if seed <= 0 {
            seed = -(seed % (RAND_MAX - 1)) + 1;
        }
        seed = seed.min(RAND_MAX - 1);
        let mut random = || {
            seed = next(seed);
            seed
        };
        let mut noise = [[[0_i64; 2]; BLOCK]; 4];
        let mut selector = [0_u8; BLOCK];
        for channel in &mut noise {
            for (i, vector) in channel.iter_mut().enumerate() {
                selector[i] = i as u8;
                vector[0] = random() % (2 * BLOCK as i64);
                vector[1] = random() % (2 * BLOCK as i64);
            }
        }
        for i in (1..BLOCK).rev() {
            let j = (random() % BLOCK as i64) as usize;
            selector.swap(i, j);
        }
        let mut gradient = [[[0.0; 2]; BLOCK]; 4];
        for (channel, vectors) in gradient.iter_mut().enumerate() {
            for (i, vector) in vectors.iter_mut().enumerate() {
                let raw = noise[channel][usize::from(selector[i])];
                let gx = (raw[0] - BLOCK as i64) as f32 / BLOCK as f32;
                let gy = (raw[1] - BLOCK as i64) as f32 / BLOCK as f32;
                let length = (gx * gx + gy * gy).sqrt();
                let (gx, gy) = if length > 0.0 {
                    (gx / length, gy / length)
                } else {
                    (gx, gy)
                };
                *vector = [stored_gradient(gx), stored_gradient(gy)];
            }
        }
        Self { selector, gradient }
    }

    fn sum(
        &self,
        point: [f32; 2],
        frequency: [f32; 2],
        octaves: u32,
        fractal: bool,
        stitch: Option<[f32; 2]>,
    ) -> [f32; 4] {
        let mut vector = [point[0] * frequency[0], point[1] * frequency[1]];
        let mut stitch = stitch;
        let mut ratio = 1.0;
        let mut sum = [0.0; 4];
        for _ in 0..octaves {
            let noise = self.noise(vector, stitch);
            for (total, value) in sum.iter_mut().zip(noise) {
                *total += if fractal { value } else { value.abs() } * ratio;
            }
            vector = vector.map(|v| v * 2.0);
            stitch = stitch.map(|s| s.map(|v| v * 2.0));
            ratio *= 0.5;
        }
        sum
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Lattice coordinates are masked to the table size after rounding."
    )]
    fn noise(&self, vector: [f32; 2], stitch: Option<[f32; 2]>) -> [f32; 4] {
        let mut floor = vector.map(f32::floor);
        let mut ceil = floor.map(|v| v + 1.0);
        let fraction = [vector[0] - floor[0], vector[1] - floor[1]];
        if let Some(stitch) = stitch {
            for axis in 0..2 {
                if floor[axis] >= stitch[axis] {
                    floor[axis] -= stitch[axis];
                }
                if ceil[axis] >= stitch[axis] {
                    ceil[axis] -= stitch[axis];
                }
            }
        }
        let index = |v: f32| (v.round() as i64 & 0xff) as usize;
        let i = f32::from(self.selector[index(floor[0])]);
        let j = f32::from(self.selector[index(ceil[0])]);
        let b00 = index(i + floor[1]);
        let b10 = index(j + floor[1]);
        let b01 = index(i + ceil[1]);
        let b11 = index(j + ceil[1]);
        let smooth = fraction.map(|t| t * t * (3.0 - 2.0 * t));
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        core::array::from_fn(|channel| {
            let dot = |b: usize, x: f32, y: f32| {
                let g = self.gradient[channel][b];
                g[0] * x + g[1] * y
            };
            let (fx, fy) = (fraction[0], fraction[1]);
            let a = lerp(dot(b00, fx, fy), dot(b10, fx - 1.0, fy), smooth[0]);
            let b = lerp(
                dot(b01, fx, fy - 1.0),
                dot(b11, fx - 1.0, fy - 1.0),
                smooth[0],
            );
            lerp(a, b, smooth[1])
        })
    }
}

/// Gradients round-trip through 16-bit storage, as in Chrome.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Normalized components map into the u16 range."
)]
fn stored_gradient(component: f32) -> f32 {
    let stored = ((component + 1.0) * 32767.5).round().clamp(0.0, 65535.0) as u16;
    f32::from(stored) * (2.0 / 65535.0) - 1.0
}

/// Park and Miller's minimal standard generator, as the specification defines it.
fn next(seed: i64) -> i64 {
    const A: i64 = 16807;
    const Q: i64 = 127_773;
    const R: i64 = 2836;
    let result = A * (seed % Q) - R * (seed / Q);
    if result <= 0 {
        result + RAND_MAX
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(kind: TurbulenceType, stitch_tiles: bool, octaves: u32) -> Pixmap {
        let mut pixels = Pixmap::new(8, 8);
        let turbulence = Turbulence {
            base_frequency: [0.3, 0.3],
            num_octaves: octaves,
            seed: 2.0,
            stitch_tiles,
            kind,
        };
        let region = Rect::new(0.0, 0.0, 8.0, 8.0);
        apply(
            &mut pixels,
            &turbulence,
            region,
            Point::ORIGIN,
            ColorSpace::Srgb,
        );
        pixels
    }

    #[test]
    fn noise_is_deterministic_and_fills_the_region() {
        let a = noise(TurbulenceType::Turbulence, false, 2);
        assert_eq!(a.data(), noise(TurbulenceType::Turbulence, false, 2).data());
        assert!(a.data().iter().any(|p| p.a > 0));
        assert_ne!(
            a.data(),
            noise(TurbulenceType::FractalNoise, false, 2).data()
        );
    }

    #[test]
    fn huge_octave_counts_are_bounded_without_changing_pixels() {
        assert_eq!(
            noise(TurbulenceType::FractalNoise, true, MAX_TURBULENCE_OCTAVES).data(),
            noise(TurbulenceType::FractalNoise, true, u32::MAX).data()
        );
    }

    #[test]
    fn zero_octaves_are_transparent_turbulence_and_gray_fractal_noise() {
        let turbulence = noise(TurbulenceType::Turbulence, false, 0);
        assert!(turbulence.data().iter().all(|p| p.a == 0));
        let fractal = noise(TurbulenceType::FractalNoise, false, 0);
        assert!(fractal.data().iter().all(|p| p.a == 128 && p.r == 64));
    }

    #[test]
    fn huge_seeds_saturate_below_the_generator_range() {
        let saturated = Lattice::new(1e10);
        assert_eq!(saturated.selector, Lattice::new(MAX_SEED).selector);
        assert_ne!(saturated.selector, Lattice::new(2.0e9).selector);
    }

    #[test]
    fn random_matches_the_park_miller_reference() {
        assert_eq!(next(1), 16807);
        assert_eq!(next(RAND_MAX - 1), RAND_MAX - 16807);
    }
}
