// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image inputs of SVG graphs (`feImage`) and their `preserveAspectRatio` placement.
//!
//! The caller decodes or rasterizes the image; the graph only samples it.

use crate::kurbo::{Affine, Rect, Size};
use crate::pixmap::Pixmap;
use alloc::sync::Arc;

/// Premultiplied pixels shared between a caller and the graphs that sample them.
/// Equality is identity: graphs compare their plans, not pixel contents.
#[derive(Clone, Debug)]
pub struct FilterImage(Arc<Pixmap>);

impl FilterImage {
    /// Share `pixmap` as an image input.
    pub fn new(pixmap: Pixmap) -> Self {
        Self(Arc::new(pixmap))
    }

    /// The shared pixels.
    pub fn pixmap(&self) -> &Pixmap {
        &self.0
    }

    /// The image's size in pixels, as a rectangle at the origin.
    pub fn bounds(&self) -> Rect {
        Rect::new(
            0.0,
            0.0,
            f64::from(self.0.width()),
            f64::from(self.0.height()),
        )
    }
}

impl PartialEq for FilterImage {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Alignment of a `preserveAspectRatio` value.
///
/// See: <https://svgwg.org/svg2-draft/coords.html#PreserveAspectRatioAttribute>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// Scale non-uniformly so the source fills the destination.
    None,
    /// Scale uniformly, aligning the `[x, y]` axes.
    Uniform([AxisAlign; 2]),
}

/// Where a uniformly scaled source sits along one axis of its destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisAlign {
    /// `xMin` or `yMin`.
    Min,
    /// `xMid` or `yMid`.
    Mid,
    /// `xMax` or `yMax`.
    Max,
}

/// A parsed `preserveAspectRatio` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectRatio {
    /// How the source aligns in its destination.
    pub align: Align,
    /// `slice` covers the destination, cropping the source; `meet` fits inside it.
    pub slice: bool,
}

impl Default for AspectRatio {
    /// `xMidYMid meet`, the attribute's initial value.
    fn default() -> Self {
        Self {
            align: Align::Uniform([AxisAlign::Mid; 2]),
            slice: false,
        }
    }
}

impl AspectRatio {
    /// Map `source` (in image pixels) into `destination` as `feImage` draws a raster
    /// image: `meet` shrinks the destination around the scaled source, while
    /// `slice` crops the source to what covers the destination. Returns the
    /// adjusted `(source, destination)`. This matches Blink's
    /// `SVGPreserveAspectRatio::TransformRect`.
    pub fn fit(&self, source: Rect, destination: Rect) -> (Rect, Rect) {
        let Align::Uniform([align_x, align_y]) = self.align else {
            return (source, destination);
        };
        if source.width() <= 0.0 || source.height() <= 0.0 {
            return (source, destination);
        }
        let (mut source, mut destination) = (source, destination);
        let aspect = source.height() / source.width();
        if self.slice {
            // The source rows or columns that fall outside the destination are cropped.
            if destination.height() < destination.width() * aspect {
                let height = destination.height() * source.width() / destination.width();
                source = along_y(source, height, align_y);
            }
            if destination.width() < destination.height() / aspect {
                let width = destination.width() * source.height() / destination.height();
                source = along_x(source, width, align_x);
            }
        } else {
            let (width, height) = (destination.width(), destination.height());
            if height > width * aspect {
                destination = along_y(destination, width * aspect, align_y);
            }
            if width > height / aspect {
                destination = along_x(destination, height / aspect, align_x);
            }
        }
        (source, destination)
    }
}

impl AspectRatio {
    /// The transform fitting `content` (a box at the origin, such as an SVG
    /// image's viewBox) into a `container` at the origin, as a `viewBox` does.
    /// <https://svgwg.org/svg2-draft/coords.html#ComputingAViewportsTransform>
    pub fn view_box_transform(&self, content: Size, container: Size) -> Affine {
        if content.width <= 0.0 || content.height <= 0.0 {
            return Affine::IDENTITY;
        }
        let sx = container.width / content.width;
        let sy = container.height / content.height;
        let Align::Uniform([align_x, align_y]) = self.align else {
            return Affine::scale_non_uniform(sx, sy);
        };
        let scale = if self.slice { sx.max(sy) } else { sx.min(sy) };
        let dx = offset(container.width - content.width * scale, align_x);
        let dy = offset(container.height - content.height * scale, align_y);
        Affine::translate((dx, dy)) * Affine::scale(scale)
    }
}

/// `rect` narrowed to `width`, aligned within its old extent.
fn along_x(rect: Rect, width: f64, align: AxisAlign) -> Rect {
    let x0 = rect.x0 + offset(rect.width() - width, align);
    Rect::new(x0, rect.y0, x0 + width, rect.y1)
}

/// `rect` narrowed to `height`, aligned within its old extent.
fn along_y(rect: Rect, height: f64, align: AxisAlign) -> Rect {
    let y0 = rect.y0 + offset(rect.height() - height, align);
    Rect::new(rect.x0, y0, rect.x1, y0 + height)
}

fn offset(slack: f64, align: AxisAlign) -> f64 {
    match align {
        AxisAlign::Min => 0.0,
        AxisAlign::Mid => slack / 2.0,
        AxisAlign::Max => slack,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aspect(x: AxisAlign, y: AxisAlign, slice: bool) -> AspectRatio {
        AspectRatio {
            align: Align::Uniform([x, y]),
            slice,
        }
    }

    #[test]
    fn meet_shrinks_the_destination_around_the_scaled_source() {
        let source = Rect::new(0.0, 0.0, 2.0, 2.0);
        let destination = Rect::new(0.0, 0.0, 8.0, 4.0);
        let fitted = |x, y| aspect(x, y, false).fit(source, destination);
        assert_eq!(
            fitted(AxisAlign::Max, AxisAlign::Mid),
            (source, Rect::new(4.0, 0.0, 8.0, 4.0))
        );
        assert_eq!(
            fitted(AxisAlign::Mid, AxisAlign::Min),
            (source, Rect::new(2.0, 0.0, 6.0, 4.0))
        );
        let tall = Rect::new(1.5, 2.0, 6.5, 5.0);
        assert_eq!(
            AspectRatio::default().fit(source, tall),
            (source, Rect::new(2.5, 2.0, 5.5, 5.0))
        );
    }

    #[test]
    fn slice_crops_the_source_and_keeps_the_destination() {
        let source = Rect::new(0.0, 0.0, 2.0, 2.0);
        let destination = Rect::new(0.0, 0.0, 8.0, 4.0);
        assert_eq!(
            AspectRatio {
                slice: true,
                ..AspectRatio::default()
            }
            .fit(source, destination),
            (Rect::new(0.0, 0.5, 2.0, 1.5), destination)
        );
        assert_eq!(
            aspect(AxisAlign::Min, AxisAlign::Max, true).fit(source, destination),
            (Rect::new(0.0, 1.0, 2.0, 2.0), destination)
        );
    }

    #[test]
    fn view_boxes_fit_meet_slice_and_stretch() {
        let content = Size::new(4.0, 2.0);
        let container = Size::new(4.0, 4.0);
        let corner = |aspect: AspectRatio| {
            let t = aspect.view_box_transform(content, container);
            (t * crate::kurbo::Point::new(4.0, 2.0)).to_vec2()
        };
        assert_eq!(corner(aspect(AxisAlign::Max, AxisAlign::Max, false)).y, 4.0);
        assert_eq!(corner(aspect(AxisAlign::Min, AxisAlign::Min, true)).x, 8.0);
        let none = AspectRatio {
            align: Align::None,
            slice: false,
        };
        assert_eq!(corner(none), crate::kurbo::Vec2::new(4.0, 4.0));
        assert_eq!(
            none.view_box_transform(Size::ZERO, container),
            Affine::IDENTITY
        );
    }

    #[test]
    fn none_and_empty_sources_are_unchanged() {
        let source = Rect::new(0.0, 0.0, 2.0, 1.0);
        let destination = Rect::new(1.0, 1.0, 9.0, 3.0);
        let none = AspectRatio {
            align: Align::None,
            slice: false,
        };
        assert_eq!(none.fit(source, destination), (source, destination));
        let empty = Rect::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!(
            AspectRatio::default().fit(empty, destination),
            (empty, destination)
        );
    }
}
