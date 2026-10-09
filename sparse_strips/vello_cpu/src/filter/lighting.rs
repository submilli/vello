// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feDiffuseLighting` and `feSpecularLighting`.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feDiffuseLightingElement>
use super::bounds::PixelBounds;
use super::channels::{in_space, store};
use alloc::vec;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::LightSource;
use vello_common::kurbo::Point;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

/// Chrome (Skia) smooths the spotlight cone edge over this cosine range.
const CONE_SMOOTHING: f32 = 0.016;

/// The reflection model and its constants.
#[derive(Clone, Copy)]
pub(super) enum Reflection {
    Diffuse { constant: f32 },
    Specular { constant: f32, exponent: f32 },
}

/// A lit surface: the input alpha scaled by `surface_scale` is a height map.
pub(super) struct Lighting<'a> {
    pub reflection: Reflection,
    pub surface_scale: f32,
    pub color: AlphaColor<Srgb>,
    pub light: &'a LightSource,
}

/// Light the node's `bounds`, its clipped region; pixels outside become transparent.
/// The height map is sampled at pixel centers, and `origin` places pixel (0, 0) in
/// the filter coordinates of light positions.
pub(super) fn apply(
    pixels: &mut Pixmap,
    lighting: &Lighting<'_>,
    bounds: PixelBounds,
    origin: Point,
    space: ColorSpace,
) {
    let width = usize::from(pixels.width());
    let alpha = |x: usize, y: usize| f32::from(pixels.data()[y * width + x].a) / 255.0;
    // Chrome converts the light color to the working space at 8-bit precision.
    let color =
        in_space(lighting.color, space).map(|c| (c.clamp(0.0, 1.0) * 255.0).round() / 255.0);
    let mut output = vec![[0.0_f32; 4]; pixels.data().len()];
    for y in bounds.y0..bounds.y1 {
        for x in bounds.x0..bounds.x1 {
            let normal = normal(&alpha, bounds, x, y, lighting.surface_scale);
            let position = [
                (origin.x + x as f64 + 0.5) as f32,
                (origin.y + y as f64 + 0.5) as f32,
                lighting.surface_scale * alpha(x, y),
            ];
            let (to_light, intensity) = light(lighting.light, position);
            let factor = match lighting.reflection {
                Reflection::Diffuse { constant } => constant * dot(normal, to_light).max(0.0),
                Reflection::Specular { constant, exponent } => {
                    let half = normalize([to_light[0], to_light[1], to_light[2] + 1.0]);
                    constant * dot(normal, half).max(0.0).powf(exponent)
                }
            };
            let rgb: [f32; 3] =
                core::array::from_fn(|i| (factor * intensity * color[i]).clamp(0.0, 1.0));
            // Specular alpha is the brightest channel, leaving premultiplied color.
            let a = match lighting.reflection {
                Reflection::Diffuse { .. } => 1.0,
                Reflection::Specular { .. } => rgb[0].max(rgb[1]).max(rgb[2]),
            };
            output[y * width + x] = [rgb[0], rgb[1], rgb[2], a];
        }
    }
    store(pixels, &output, space);
}

/// Sobel normals of the height map. Like Chrome, edge pixels repeat the nearest
/// pixel inside `bounds` instead of using the specification's edge kernels.
fn normal(
    alpha: &impl Fn(usize, usize) -> f32,
    bounds: PixelBounds,
    x: usize,
    y: usize,
    surface_scale: f32,
) -> [f32; 3] {
    let left = if x > bounds.x0 { x - 1 } else { x };
    let right = if x + 1 < bounds.x1 { x + 1 } else { x };
    let up = if y > bounds.y0 { y - 1 } else { y };
    let down = if y + 1 < bounds.y1 { y + 1 } else { y };
    let sobel = |a: f32, b: f32, c: f32| 0.25 * (a + 2.0 * b + c);
    let nx = sobel(alpha(right, up), alpha(right, y), alpha(right, down))
        - sobel(alpha(left, up), alpha(left, y), alpha(left, down));
    let ny = sobel(alpha(left, down), alpha(x, down), alpha(right, down))
        - sobel(alpha(left, up), alpha(x, up), alpha(right, up));
    normalize([-surface_scale * nx, -surface_scale * ny, 1.0])
}

/// Unit vector from the surface to the light and the light's intensity factor.
fn light(source: &LightSource, surface: [f32; 3]) -> ([f32; 3], f32) {
    match source {
        LightSource::Distant { azimuth, elevation } => {
            let (az, el) = (azimuth.to_radians(), elevation.to_radians());
            ([az.cos() * el.cos(), az.sin() * el.cos(), el.sin()], 1.0)
        }
        LightSource::Point { x, y, z } => (
            normalize([x - surface[0], y - surface[1], z - surface[2]]),
            1.0,
        ),
        LightSource::Spot {
            x,
            y,
            z,
            points_at_x,
            points_at_y,
            points_at_z,
            specular_exponent,
            limiting_cone_angle,
        } => {
            let to_light = normalize([x - surface[0], y - surface[1], z - surface[2]]);
            let direction = normalize([points_at_x - x, points_at_y - y, points_at_z - z]);
            let cosine = -dot(to_light, direction);
            // Without a limiting cone only the hemisphere facing the target is lit.
            let cutoff =
                limiting_cone_angle.map_or(0.0, |angle| angle.abs().min(90.0).to_radians().cos());
            let mut intensity = 0.0;
            if cosine >= cutoff {
                intensity = cosine.powf(*specular_exponent);
                if cosine < cutoff + CONE_SMOOTHING {
                    intensity *= (cosine - cutoff) / CONE_SMOOTHING;
                }
            }
            (to_light, intensity)
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    if length > 0.0 {
        v.map(|c| c / length)
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::palette::css::WHITE;

    fn lit(reflection: Reflection, light: &LightSource) -> Pixmap {
        let mut pixels = Pixmap::new(3, 3);
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 3,
        };
        let lighting = Lighting {
            reflection,
            surface_scale: 1.0,
            color: WHITE,
            light,
        };
        apply(
            &mut pixels,
            &lighting,
            bounds,
            Point::ORIGIN,
            ColorSpace::Srgb,
        );
        pixels
    }

    #[test]
    fn flat_surfaces_follow_lambert_and_blinn_phong() {
        let overhead = LightSource::Distant {
            azimuth: 0.0,
            elevation: 30.0,
        };
        let diffuse = lit(Reflection::Diffuse { constant: 1.0 }, &overhead);
        assert!(diffuse.data().iter().all(|p| p.r == 128 && p.a == 255));
        let straight_down = LightSource::Distant {
            azimuth: 0.0,
            elevation: 90.0,
        };
        let specular = lit(
            Reflection::Specular {
                constant: 1.0,
                exponent: 4.0,
            },
            &straight_down,
        );
        assert!(specular.data().iter().all(|p| p.r == 255 && p.a == 255));
    }

    #[test]
    fn spotlights_cut_off_outside_the_cone() {
        let spot = |limiting_cone_angle| LightSource::Spot {
            x: 0.5,
            y: 0.5,
            z: 1.0,
            points_at_x: 0.5,
            points_at_y: 0.5,
            points_at_z: 0.0,
            specular_exponent: 1.0,
            limiting_cone_angle,
        };
        let pixels = lit(Reflection::Diffuse { constant: 1.0 }, &spot(Some(30.0)));
        assert_eq!(pixels.data()[0].r, 255);
        assert_eq!(pixels.data()[8].r, 0);
        assert_eq!(pixels.data()[8].a, 255);
        // Without a cone, the light still reaches the whole hemisphere it faces.
        let pixels = lit(Reflection::Diffuse { constant: 1.0 }, &spot(None));
        assert!(pixels.data()[8].r > 0);
    }

    #[test]
    fn edge_normals_repeat_the_nearest_inside_pixel() {
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 1,
        };
        let alpha = |x: usize, _: usize| if x == 1 { 1.0 } else { 0.0 };
        // Rows repeat at the edge: each Sobel column sums 4 * alpha / 4.
        let n = normal(&alpha, bounds, 0, 0, 1.0);
        let expected = normalize([-1.0, 0.0, 1.0]);
        assert!(n.iter().zip(expected).all(|(a, e)| (a - e).abs() < 1e-6));
        let single = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        };
        assert_eq!(normal(&alpha, single, 0, 0, 5.0), [0.0, 0.0, 1.0]);
    }
}
