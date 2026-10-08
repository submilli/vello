// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CSS filter conversion and the bounded sequential CPU subset.

use crate::filter_effects::{EdgeMode, FilterFunction, FilterPrimitive, matrices};
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

pub(crate) fn primitive(function: FilterFunction) -> FilterPrimitive {
    let mut matrix = matrices::IDENTITY;
    match function {
        FilterFunction::Blur { radius } => {
            return FilterPrimitive::GaussianBlur {
                std_deviation: radius,
                edge_mode: EdgeMode::None,
            };
        }
        FilterFunction::Brightness { amount } => rgb_linear(&mut matrix, amount, 0.0),
        FilterFunction::Contrast { amount } => {
            rgb_linear(&mut matrix, amount, (1.0 - amount) / 2.0);
        }
        FilterFunction::Invert { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            rgb_linear(&mut matrix, 1.0 - 2.0 * amount, amount);
        }
        FilterFunction::Opacity { amount } => matrix[18] = amount.clamp(0.0, 1.0),
        FilterFunction::Grayscale { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            for (output, grayscale) in matrix.iter_mut().zip(matrices::GRAYSCALE) {
                *output = *output * (1.0 - amount) + grayscale * amount;
            }
        }
        FilterFunction::Saturate { amount } => matrix = saturation(amount),
        FilterFunction::Sepia { amount } => {
            let amount = amount.clamp(0.0, 1.0);
            for (output, sepia) in matrix.iter_mut().zip(matrices::SEPIA) {
                *output = *output * (1.0 - amount) + sepia * amount;
            }
        }
        FilterFunction::HueRotate { angle } => {
            let (s, c) = angle.to_radians().sin_cos();
            matrix = [
                0.213 + 0.787 * c - 0.213 * s,
                0.715 - 0.715 * c - 0.715 * s,
                0.072 - 0.072 * c + 0.928 * s,
                0.0,
                0.0,
                0.213 - 0.213 * c + 0.143 * s,
                0.715 + 0.285 * c + 0.140 * s,
                0.072 - 0.072 * c - 0.283 * s,
                0.0,
                0.0,
                0.213 - 0.213 * c - 0.787 * s,
                0.715 - 0.715 * c + 0.715 * s,
                0.072 + 0.928 * c + 0.072 * s,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ];
        }
    }
    FilterPrimitive::ColorMatrix { matrix }
}

fn rgb_linear(matrix: &mut [f32; 20], slope: f32, intercept: f32) {
    for row in 0..3 {
        matrix[row * 6] = slope;
        matrix[row * 5 + 4] = intercept;
    }
}

fn saturation(amount: f32) -> [f32; 20] {
    let mut matrix = matrices::IDENTITY;
    for row in 0..3 {
        for (column, luma) in [0.213, 0.715, 0.072].into_iter().enumerate() {
            matrix[row * 5 + column] =
                luma * (1.0 - amount) + if row == column { amount } else { 0.0 };
        }
    }
    matrix
}

pub(crate) use super::parameters::valid_unary_parameters as supported;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::Filter;
    use crate::kurbo::{Affine, Rect};

    #[test]
    fn chain_accumulates_padding_and_rejects_unbounded_or_unsupported_input() {
        let blur = || primitive(FilterFunction::Blur { radius: 2.0 });
        let chain = Filter::from_chain([blur(), blur()]).unwrap();
        assert_eq!(
            chain.filter_expansion(&Affine::IDENTITY),
            Rect::new(-12.0, -12.0, 12.0, 12.0)
        );
        assert_eq!(
            chain.source_expansion(&Affine::IDENTITY),
            Rect::new(-12.0, -12.0, 12.0, 12.0)
        );
        assert!(Filter::from_chain(core::iter::repeat_with(blur)).is_none());
        assert!(Filter::from_chain([FilterPrimitive::Tile]).is_none());
        assert!(
            Filter::from_chain([primitive(FilterFunction::Blur { radius: f32::NAN })]).is_none()
        );
    }
}
