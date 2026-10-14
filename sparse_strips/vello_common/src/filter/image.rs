// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image inputs of SVG graphs (`feImage`) and their `preserveAspectRatio` placement.
//!
//! The caller decodes or rasterizes the image. Planning a draw decides Chrome's
//! sampling ([`Sampling`]); the image then shares its texels with the graph,
//! which keeps only windows of the mip levels a downscale reads.

use super::image_sampling::{Sampling, TexelRect};
use super::mipmap::{self, Request};
use crate::color::PremulRgba8;
use crate::kurbo::{Affine, Rect, Size};
use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

/// Rows of premultiplied texels an image input is read from.
pub trait ImageRows: Send + Sync {
    /// Write the texels from column `x0` of row `y` into `row`, or return
    /// `false` when the image does not have them all.
    fn read(&self, x0: u32, y: u32, row: &mut [PremulRgba8]) -> bool;
}

impl ImageRows for Pixmap {
    fn read(&self, x0: u32, y: u32, row: &mut [PremulRgba8]) -> bool {
        let (width, height) = (u32::from(self.width()), u32::from(self.height()));
        let Ok(len) = u32::try_from(row.len()) else {
            return false;
        };
        if !x0.checked_add(len).is_some_and(|x1| x1 <= width) || y >= height {
            return false;
        }
        let start = (y * width + x0) as usize;
        row.copy_from_slice(&self.data()[start..start + row.len()]);
        true
    }
}

/// Texels of one image draw, shared between a caller and the graphs that
/// sample them. Equality is identity: graphs compare their plans, not pixel
/// contents.
#[derive(Clone, Debug)]
pub struct FilterImage(Arc<Sampled>);

#[derive(Debug)]
struct Sampled {
    sampling: Sampling,
    windows: Vec<Window>,
}

impl FilterImage {
    /// Plan drawing a `size` image whose `source` (in texels) is placed at
    /// `destination`, within the primitive's `crop` (its subregion within the
    /// filter region), for pixels overlapping `area`. `None` when nothing of
    /// the image shows there.
    pub fn plan(
        size: [u32; 2],
        source: Rect,
        destination: Rect,
        crop: Rect,
        area: Rect,
    ) -> Option<ImagePlan> {
        let sampling = Sampling::new(size, source, destination, crop)?;
        let requests = sampling
            .levels()
            .map(|(level, _)| {
                let window = sampling.window(level, area)?;
                Some(Request { level, window })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(ImagePlan { sampling, requests })
    }

    /// How the draw samples.
    pub fn sampling(&self) -> &Sampling {
        &self.0.sampling
    }

    /// The texels of mip `level` (0 for the subset itself).
    pub fn window(&self, level: u8) -> Option<&Window> {
        self.0.windows.iter().find(|window| window.level == level)
    }

    /// Texels the image keeps alive besides those it shares with its caller.
    pub fn retained_texels(&self) -> usize {
        self.0.windows.iter().map(Window::retained).sum()
    }
}

/// Texels retained by these draws, counting each shared allocation once.
pub(super) fn unique_retained<'a>(images: impl Iterator<Item = &'a FilterImage>) -> usize {
    let mut seen = Vec::new();
    let mut total = 0_usize;
    for image in images {
        for window in &image.0.windows {
            let retained = window.retained();
            if retained == 0 {
                continue;
            }
            let identity = match &window.texels {
                Texels::Kept(pixels) => Arc::as_ptr(pixels).cast::<()>(),
                Texels::Shared { image, .. } => Arc::as_ptr(image).cast::<()>(),
            };
            if !seen.contains(&identity) {
                seen.push(identity);
                total = total.saturating_add(retained);
            }
        }
    }
    total
}

impl PartialEq for FilterImage {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A planned draw whose texels are not read yet.
#[derive(Debug)]
pub struct ImagePlan {
    pub(super) sampling: Sampling,
    pub(super) requests: Vec<Request>,
}

impl ImagePlan {
    /// Texels of the mip windows a draw keeps, or `None` when one is too wide
    /// or tall for a pixmap, as building it would find.
    pub fn mip_texels(&self) -> Option<u64> {
        let mut total = 0;
        for request in self.mip_requests() {
            let window = request.window;
            u16::try_from(window.width().max(window.height())).ok()?;
            total += window.area();
        }
        Some(total)
    }

    /// Work to stream this draw's mip windows before `share` allocates them.
    /// Includes channel arithmetic, row copies and level/request scans;
    /// source conversion and subsequent drawing are charged by the caller.
    pub fn stream_work(&self) -> u64 {
        let subset = self.sampling.subset;
        let requests: Vec<Request> = self.mip_requests().copied().collect();
        mipmap::stream_work([subset.width(), subset.height()], &requests)
    }

    /// The image texels the draw reads, its mip levels included.
    pub fn base_texels(&self) -> TexelRect {
        let subset = self.sampling.subset;
        let size = [subset.width(), subset.height()];
        let base = mipmap::base_window(size, &self.requests);
        TexelRect {
            x0: subset.x0 + base.x0,
            y0: subset.y0 + base.y0,
            x1: subset.x0 + base.x1,
            y1: subset.y0 + base.y1,
        }
    }

    /// Sample `image`, the whole image, sharing its texels: only mip windows
    /// are copied. `None` when `image` lacks texels they need, or one is too
    /// large for a pixmap.
    pub fn share(self, image: Arc<dyn ImageRows>) -> Option<FilterImage> {
        self.finish(image, 0)
    }

    /// Sample `raster`, which holds the image's texels from `origin`, keeping
    /// it. `None` unless it covers [`base_texels`](Self::base_texels).
    pub fn keep(self, raster: Pixmap, origin: [u32; 2]) -> Option<FilterImage> {
        let base = self.base_texels();
        let [x, y] = origin.map(u64::from);
        let covers = x <= u64::from(base.x0)
            && y <= u64::from(base.y0)
            && x + u64::from(raster.width()) >= u64::from(base.x1)
            && y + u64::from(raster.height()) >= u64::from(base.y1);
        if !covers {
            return None;
        }
        let retained = usize::from(raster.width()) * usize::from(raster.height());
        self.finish(Arc::new(Offset(raster, origin)), retained)
    }

    fn finish(self, image: Arc<dyn ImageRows>, retained: usize) -> Option<FilterImage> {
        let subset = self.sampling.subset;
        let mips: Vec<Request> = self.mip_requests().copied().collect();
        let built = mipmap::build(&*image, subset, &mips)?;
        let windows: Vec<Window> = mips
            .iter()
            .zip(built)
            .map(|(request, pixels)| Window {
                level: request.level,
                size: self.sampling.level_size(request.level),
                origin: [request.window.x0, request.window.y0].map(i64::from),
                texels: Texels::Kept(Arc::new(pixels)),
            })
            .collect();
        self.with_windows(image, retained, windows)
    }

    pub(super) fn with_windows(
        self,
        image: Arc<dyn ImageRows>,
        retained: usize,
        mut windows: Vec<Window>,
    ) -> Option<FilterImage> {
        let subset = self.sampling.subset;
        if self.requests.iter().any(|request| request.level == 0) {
            windows.push(Window {
                level: 0,
                size: self.sampling.level_size(0),
                origin: [-i64::from(subset.x0), -i64::from(subset.y0)],
                texels: Texels::Shared { image, retained },
            });
        }
        Some(FilterImage(Arc::new(Sampled {
            sampling: self.sampling,
            windows,
        })))
    }

    pub(super) fn mip_requests(&self) -> impl Iterator<Item = &Request> {
        self.requests.iter().filter(|request| request.level > 0)
    }
}

/// A raster holding an image's texels from an origin.
struct Offset(Pixmap, [u32; 2]);

impl ImageRows for Offset {
    fn read(&self, x0: u32, y: u32, row: &mut [PremulRgba8]) -> bool {
        let [ox, oy] = self.1;
        match (x0.checked_sub(ox), y.checked_sub(oy)) {
            (Some(x0), Some(y)) => self.0.read(x0, y, row),
            _ => false,
        }
    }
}

/// Texels of one mip level that a draw reads.
#[derive(Clone, Debug)]
pub struct Window {
    pub(super) level: u8,
    /// The level's size.
    pub(super) size: [u32; 2],
    /// Where texel `(0, 0)` of `texels` lies in the level, relative to the subset.
    pub(super) origin: [i64; 2],
    pub(super) texels: Texels,
}

#[derive(Clone)]
pub(super) enum Texels {
    /// A window computed for the draw.
    Kept(Arc<Pixmap>),
    /// The whole image, `retained` texels of it owned by the draw.
    Shared {
        image: Arc<dyn ImageRows>,
        retained: usize,
    },
}

impl fmt::Debug for Texels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kept(pixmap) => f.debug_tuple("Kept").field(pixmap).finish(),
            Self::Shared { retained, .. } => f
                .debug_struct("Shared")
                .field("retained", retained)
                .finish_non_exhaustive(),
        }
    }
}

impl Window {
    /// Texel `(x, y)` of the level, relative to the subset. Reads clamp to the
    /// level, as Skia clamps to the subset. A plan keeps every texel its
    /// pixels read; past those, a kept window reads its nearest texel and a
    /// shared image transparent.
    pub fn texel(&self, x: i64, y: i64) -> PremulRgba8 {
        let x = x.clamp(0, i64::from(self.size[0].max(1)) - 1) - self.origin[0];
        let y = y.clamp(0, i64::from(self.size[1].max(1)) - 1) - self.origin[1];
        match &self.texels {
            Texels::Kept(pixmap) => kept(pixmap, x, y),
            Texels::Shared { image, .. } => {
                let mut texel = [PremulRgba8::from_u32(0)];
                match (u32::try_from(x), u32::try_from(y)) {
                    (Ok(x), Ok(y)) if image.read(x, y, &mut texel) => texel[0],
                    _ => PremulRgba8::from_u32(0),
                }
            }
        }
    }

    fn retained(&self) -> usize {
        match &self.texels {
            Texels::Kept(pixmap) => usize::from(pixmap.width()) * usize::from(pixmap.height()),
            Texels::Shared { retained, .. } => *retained,
        }
    }
}

#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "The index is clamped into the pixmap."
)]
fn kept(pixmap: &Pixmap, x: i64, y: i64) -> PremulRgba8 {
    let (width, height) = (pixmap.width(), pixmap.height());
    if width == 0 || height == 0 {
        return PremulRgba8::from_u32(0);
    }
    let x = x.clamp(0, i64::from(width) - 1) as usize;
    let y = y.clamp(0, i64::from(height) - 1) as usize;
    pixmap.data()[y * usize::from(width) + x]
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

    fn alphas(width: u16, height: u16) -> Pixmap {
        let texels = (0..u32::from(width) * u32::from(height))
            .map(|i| PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: (i % 251) as u8,
            })
            .collect();
        Pixmap::from_parts(texels, width, height)
    }

    #[test]
    fn shared_images_keep_only_mip_windows() {
        let image: Arc<dyn ImageRows> = Arc::new(alphas(64, 64));
        let whole = Rect::new(0.0, 0.0, 64.0, 64.0);
        let area = Rect::new(0.0, 0.0, 8.0, 8.0);
        // At its size, the image is sampled where it is.
        let plan = FilterImage::plan([64, 64], whole, whole, whole, area).unwrap();
        assert_eq!(plan.mip_texels(), Some(0));
        let image_at_size = plan.share(Arc::clone(&image)).unwrap();
        assert_eq!(image_at_size.retained_texels(), 0);
        assert_eq!(image_at_size.window(0).unwrap().texel(3, 2).a, 2 * 64 + 3);
        // An eighth of its size (level 2.5), levels 2 (16x16) and 3 (8x8) are
        // kept where the 8x8 area reads them, with a texel or two of taps.
        let small = Rect::new(0.0, 0.0, 8.0, 8.0);
        let plan = FilterImage::plan([64, 64], whole, small, small, area).unwrap();
        let kept = plan.mip_texels().unwrap();
        assert!(kept > 0 && kept <= 18 * 18 + 10 * 10, "{kept}");
        let downscaled = plan.share(image).unwrap();
        assert_eq!(downscaled.retained_texels() as u64, kept);
        assert!(downscaled.window(0).is_none());
    }

    #[test]
    fn kept_rasters_need_only_the_planned_texels() {
        // Pixels overlapping 2..4 in a 1:1 draw read texels 1 to 5.
        let row = Rect::new(0.0, 0.0, 16.0, 1.0);
        let plan =
            FilterImage::plan([16, 1], row, row, row, Rect::new(2.0, 0.0, 4.0, 1.0)).unwrap();
        let texels = plan.base_texels();
        assert_eq!((texels.x0, texels.x1), (1, 6));
        let raster = alphas(5, 1);
        let image = plan.keep(raster, [1, 0]).unwrap();
        assert_eq!(image.retained_texels(), 5);
        let level = image.window(0).unwrap();
        assert_eq!(level.texel(2, 0).a, 1);
        // Texels outside the raster read as transparent.
        assert_eq!(level.texel(9, 0).a, 0);
        // A raster short of the planned texels is refused.
        let plan =
            FilterImage::plan([16, 1], row, row, row, Rect::new(2.0, 0.0, 4.0, 1.0)).unwrap();
        assert!(plan.keep(alphas(4, 1), [1, 0]).is_none());
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
