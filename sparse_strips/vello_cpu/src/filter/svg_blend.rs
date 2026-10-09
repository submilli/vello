// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feBlend`: every Compositing and Blending Level 1 blend mode, source over.
//!
//! See: <https://drafts.fxtf.org/compositing-1/#blending>
use super::channels::{encode_stored, straight};
use vello_common::filter::graph::ColorSpace;
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::peniko::Mix;
use vello_common::pixmap::Pixmap;

/// Blend `pixels` (the `in` source) over `other` (the `in2` backdrop).
pub(super) fn blend(pixels: &mut Pixmap, other: &Pixmap, mode: Mix, space: ColorSpace) {
    for (pixel, other) in pixels.data_mut().iter_mut().zip(other.data()) {
        let s = straight(*pixel, space);
        let b = straight(*other, space);
        let alpha = s[3] + b[3] - s[3] * b[3];
        let mixed = mix(mode, [b[0], b[1], b[2]], [s[0], s[1], s[2]]);
        let mut output = [0.0, 0.0, 0.0, alpha];
        if alpha > 0.0 {
            for i in 0..3 {
                output[i] = ((1.0 - b[3]) * s[3] * s[i]
                    + (1.0 - s[3]) * b[3] * b[i]
                    + s[3] * b[3] * mixed[i])
                    / alpha;
            }
        }
        *pixel = encode_stored(output, space);
    }
    pixels.recompute_may_have_transparency();
}

type Rgb = [f32; 3];

fn mix(mode: Mix, backdrop: Rgb, source: Rgb) -> Rgb {
    let separable = |f: fn(f32, f32) -> f32| {
        [
            f(backdrop[0], source[0]),
            f(backdrop[1], source[1]),
            f(backdrop[2], source[2]),
        ]
    };
    match mode {
        Mix::Normal => source,
        Mix::Multiply => separable(|b, s| b * s),
        Mix::Screen => separable(screen),
        Mix::Overlay => separable(|b, s| hard_light(s, b)),
        Mix::Darken => separable(f32::min),
        Mix::Lighten => separable(f32::max),
        Mix::ColorDodge => separable(color_dodge),
        Mix::ColorBurn => separable(color_burn),
        Mix::HardLight => separable(hard_light),
        Mix::SoftLight => separable(soft_light),
        Mix::Difference => separable(|b, s| (b - s).abs()),
        Mix::Exclusion => separable(|b, s| b + s - 2.0 * b * s),
        Mix::Hue => set_lum(set_sat(source, sat(backdrop)), lum(backdrop)),
        Mix::Saturation => set_lum(set_sat(backdrop, sat(source)), lum(backdrop)),
        Mix::Color => set_lum(source, lum(backdrop)),
        Mix::Luminosity => set_lum(backdrop, lum(source)),
    }
}

fn screen(b: f32, s: f32) -> f32 {
    b + s - b * s
}

fn hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        b * 2.0 * s
    } else {
        screen(b, 2.0 * s - 1.0)
    }
}

fn color_dodge(b: f32, s: f32) -> f32 {
    if b == 0.0 {
        0.0
    } else if s >= 1.0 {
        1.0
    } else {
        (b / (1.0 - s)).min(1.0)
    }
}

fn color_burn(b: f32, s: f32) -> f32 {
    if b >= 1.0 {
        1.0
    } else if s <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - b) / s).min(1.0)
    }
}

fn soft_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    }
    let d = if b <= 0.25 {
        ((16.0 * b - 12.0) * b + 4.0) * b
    } else {
        b.sqrt()
    };
    b + (2.0 * s - 1.0) * (d - b)
}

fn lum(c: Rgb) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn sat(c: Rgb) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_lum(c: Rgb, l: f32) -> Rgb {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

fn clip_color(c: Rgb) -> Rgb {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut out = c;
    if n < 0.0 && l - n > 0.0 {
        for v in &mut out {
            *v = l + (*v - l) * l / (l - n);
        }
    }
    if x > 1.0 && x - l > 0.0 {
        for v in &mut out {
            *v = l + (*v - l) * (1.0 - l) / (x - l);
        }
    }
    out
}

/// Scale the channel range to `s`, keeping the order of the components.
fn set_sat(c: Rgb, s: f32) -> Rgb {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    if max <= min {
        return [0.0; 3];
    }
    c.map(|v| (v - min) * s / (max - min))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separable_modes_follow_the_compositing_formulas() {
        let (b, s) = ([0.2, 0.6, 1.0], [0.8, 0.4, 0.0]);
        let close = |a: Rgb, e: Rgb| a.iter().zip(e).all(|(a, e)| (a - e).abs() < 1e-5);
        assert!(close(mix(Mix::Overlay, b, s), [0.32, 0.52, 1.0]));
        assert!(close(mix(Mix::ColorDodge, b, s), [1.0, 1.0, 1.0]));
        assert!(close(mix(Mix::ColorBurn, b, s), [0.0, 0.0, 1.0]));
        assert!(close(mix(Mix::HardLight, b, s), [0.68, 0.48, 0.0]));
        assert!(close(mix(Mix::Difference, b, s), [0.6, 0.2, 1.0]));
        assert!(close(mix(Mix::Exclusion, b, s), [0.68, 0.52, 1.0]));
        let soft = mix(Mix::SoftLight, b, s);
        assert!(
            (soft[0] - (0.2 + 0.6 * (((16.0 * 0.2 - 12.0) * 0.2 + 4.0) * 0.2 - 0.2))).abs() < 1e-5
        );
    }

    #[test]
    fn non_separable_modes_keep_luminosity_and_saturation_in_gamut() {
        let (b, s) = ([0.9, 0.1, 0.1], [0.1, 0.1, 0.9]);
        for mode in [Mix::Hue, Mix::Saturation, Mix::Color, Mix::Luminosity] {
            let out = mix(mode, b, s);
            assert!(
                out.iter().all(|v| (-1e-5..=1.0 + 1e-5).contains(v)),
                "{mode:?} {out:?}"
            );
        }
        assert!((lum(mix(Mix::Color, b, s)) - lum(b)).abs() < 1e-5);
        assert!((lum(mix(Mix::Luminosity, b, s)) - lum(s)).abs() < 1e-5);
        assert_eq!(mix(Mix::Hue, [0.5; 3], s), [0.5; 3]);
    }
}
