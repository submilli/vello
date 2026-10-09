// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SVG `feComposite`: Porter-Duff operators and the arithmetic combination.
//!
//! See: <https://drafts.fxtf.org/filter-effects/#feCompositeElement>
use super::channels::{encode_stored, straight};
use vello_common::filter::graph::ColorSpace;
use vello_common::filter_effects::CompositeOperator;
use vello_common::pixmap::Pixmap;

/// Composite `pixels` (`in`) with `other` (`in2`) in `space`.
pub(super) fn composite(
    pixels: &mut Pixmap,
    other: &Pixmap,
    operator: CompositeOperator,
    space: ColorSpace,
) {
    for (p, q) in pixels.data_mut().iter_mut().zip(other.data()) {
        let mut s = straight(*p, space);
        let mut d = straight(*q, space);
        for i in 0..3 {
            s[i] *= s[3];
            d[i] *= d[3];
        }
        let (fs, fd) = match operator {
            CompositeOperator::Over => (1.0, 1.0 - s[3]),
            CompositeOperator::In => (d[3], 0.0),
            CompositeOperator::Out => (1.0 - d[3], 0.0),
            CompositeOperator::Atop => (d[3], 1.0 - s[3]),
            CompositeOperator::Xor => (1.0 - d[3], 1.0 - s[3]),
            CompositeOperator::Arithmetic { .. } => (0.0, 0.0),
        };
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = match operator {
                CompositeOperator::Arithmetic { k1, k2, k3, k4 } => {
                    (k1 * s[i] * d[i] + k2 * s[i] + k3 * d[i] + k4).clamp(0.0, 1.0)
                }
                _ => (fs * s[i] + fd * d[i]).clamp(0.0, 1.0),
            };
        }
        for i in 0..3 {
            out[i] = if out[3] > 0.0 {
                out[i].min(out[3]) / out[3]
            } else {
                0.0
            };
        }
        *p = encode_stored(out, space);
    }
    pixels.recompute_may_have_transparency();
}
