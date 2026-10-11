// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rasters placed in an SVG graph's coordinate space: the source and each
//! node's result, held over its own frame.

use vello_common::color::PremulRgba8;
use vello_common::filter::graph::Input;
use vello_common::kurbo::Rect;
use vello_common::pixmap::Pixmap;

use super::channels::TRANSPARENT;

/// A raster and where it lies in graph coordinates, on the pixel grid.
pub(super) struct Framed {
    pub pixels: Pixmap,
    pub frame: Rect,
}

impl Framed {
    /// The pixel at graph position (`x`, `y`), transparent outside the frame.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Frames are whole pixels within u16 rasters, so offsets within them fit."
    )]
    pub(super) fn get(&self, x: f64, y: f64) -> PremulRgba8 {
        let (x0, y0) = (self.frame.x0, self.frame.y0);
        if x < x0 || y < y0 || x >= self.frame.x1 || y >= self.frame.y1 {
            return TRANSPARENT;
        }
        let width = usize::from(self.pixels.width());
        self.pixels.data()[(y - y0) as usize * width + (x - x0) as usize]
    }

    /// A raster over `frame` holding these pixels where the frames overlap.
    pub(super) fn reframe(&self, frame: Rect) -> Self {
        let mut pixels = raster(frame);
        self.copy_into(&mut pixels, frame);
        Self { pixels, frame }
    }

    /// Copy where this frame and `frame`, the frame of `target`, overlap.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "The overlap lies within both whole-pixel u16 frames."
    )]
    pub(super) fn copy_into(&self, target: &mut Pixmap, frame: Rect) {
        let overlap = self.frame.intersect(frame);
        if overlap.width() <= 0.0 || overlap.height() <= 0.0 {
            return;
        }
        let from_width = usize::from(self.pixels.width());
        let to_width = usize::from(target.width());
        let columns = overlap.width() as usize;
        let from_x = (overlap.x0 - self.frame.x0) as usize;
        let to_x = (overlap.x0 - frame.x0) as usize;
        for row in 0..overlap.height() as usize {
            let from_y = (overlap.y0 - self.frame.y0) as usize + row;
            let to_y = (overlap.y0 - frame.y0) as usize + row;
            let from = from_y * from_width + from_x;
            let to = to_y * to_width + to_x;
            target.data_mut()[to..to + columns]
                .copy_from_slice(&self.pixels.data()[from..from + columns]);
        }
        target.recompute_may_have_transparency();
    }
}

/// `SourceAlpha` keeps only the source's alpha.
pub(super) fn alpha_only(pixel: PremulRgba8, input: Input) -> PremulRgba8 {
    match input {
        Input::SourceAlpha => PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: pixel.a,
        },
        Input::SourceGraphic | Input::Result(_) => pixel,
    }
}

/// A transparent raster over a frame that admission bounded to u16 sides.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Admission bounds every frame's sides to u16."
)]
pub(super) fn raster(frame: Rect) -> Pixmap {
    Pixmap::new(frame.width() as u16, frame.height() as u16)
}
