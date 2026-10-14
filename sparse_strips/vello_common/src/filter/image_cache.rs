// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw-local mip windows, shared by image identity and sampled subset.
//!
//! Collect every primitive's plan before streaming. Windows of each level are
//! unioned, so one pass supplies differently placed or scaled uses of an image.
//! Retain the cache across planning retries: a narrower read reuses the windows
//! already built. Sources must remain immutable until the cache is dropped.

use super::image::{FilterImage, ImagePlan, ImageRows, Texels, Window};
use super::image_sampling::TexelRect;
use super::mipmap::{self, Request};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// A draw's bounded image cache. Shared source identity is `Arc` identity, not
/// an address supplied by the page. Different subsets have different mipmaps.
#[derive(Default)]
pub struct ImageCache {
    entries: Vec<Entry>,
    streamed: u64,
}

struct Entry {
    image: Arc<dyn ImageRows>,
    subset: TexelRect,
    requests: Vec<Request>,
    windows: Vec<Window>,
}

/// Why image plans cannot be completed within the caller's draw limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// Unioned mip windows or streamed source texels exceed a supplied limit.
    Limit,
    /// A source does not supply all its declared texels.
    MissingTexels,
}

impl ImageCache {
    /// Complete all `plans` together. Limits apply to the entire cache, including
    /// earlier planning retries. No streaming starts until every new union is
    /// admitted. A failed source read still consumes its stream allowance.
    ///
    /// Plans with no mipmaps read their source in place and consume no stream
    /// allowance. New windows outside an earlier union require another build,
    /// charged in full; callers should collect the widest reads first.
    pub fn share_batch(
        &mut self,
        plans: Vec<(ImagePlan, Arc<dyn ImageRows>)>,
        max_texels: u64,
        max_streamed: u64,
    ) -> Result<Vec<FilterImage>, CacheError> {
        let mut unions: Vec<(usize, Vec<Request>)> = Vec::new();
        let mut indices = Vec::with_capacity(plans.len());
        for (plan, image) in &plans {
            let index = self.entry(image, plan.sampling.subset);
            indices.push(index);
            let requests = match unions.iter_mut().find(|(i, _)| *i == index) {
                Some((_, requests)) => requests,
                None => {
                    unions.push((index, self.entries[index].requests.clone()));
                    &mut unions.last_mut().expect("the union was just pushed").1
                }
            };
            for request in plan.mip_requests() {
                merge(requests, *request);
            }
        }
        let mut kept = self.retained_texels();
        let mut stream = 0_u64;
        for (index, requests) in &unions {
            let entry = &self.entries[*index];
            if *requests == entry.requests {
                continue;
            }
            let pixels = texels(requests).ok_or(CacheError::Limit)?;
            kept = kept - texels(&entry.requests).ok_or(CacheError::Limit)? + pixels;
            let size = [entry.subset.width(), entry.subset.height()];
            stream = stream
                .checked_add(mipmap::base_window(size, requests).area())
                .ok_or(CacheError::Limit)?;
        }
        if kept > max_texels
            || self
                .streamed
                .checked_add(stream)
                .is_none_or(|n| n > max_streamed)
        {
            return Err(CacheError::Limit);
        }
        for (index, requests) in unions {
            let entry = &mut self.entries[index];
            if requests == entry.requests {
                continue;
            }
            let size = [entry.subset.width(), entry.subset.height()];
            self.streamed += mipmap::base_window(size, &requests).area();
            let built = mipmap::build(&*entry.image, entry.subset, &requests)
                .ok_or(CacheError::MissingTexels)?;
            entry.windows = requests
                .iter()
                .zip(built)
                .map(|(request, pixels)| Window {
                    level: request.level,
                    size: super::image_sampling::level_size(size, request.level),
                    origin: [request.window.x0, request.window.y0].map(i64::from),
                    texels: Texels::Kept(Arc::new(pixels)),
                })
                .collect();
            entry.requests = requests;
        }
        plans
            .into_iter()
            .zip(indices)
            .map(|((plan, image), index)| {
                let windows = self.entries[index]
                    .windows
                    .iter()
                    .filter(|window| plan.mip_requests().any(|r| r.level == window.level))
                    .cloned()
                    .collect();
                plan.with_windows(image, 0, windows)
                    .ok_or(CacheError::MissingTexels)
            })
            .collect()
    }

    /// Source texels streamed so far, including unsuccessful reads.
    pub fn streamed_texels(&self) -> u64 {
        self.streamed
    }

    /// Mip texels retained by the cache, counting a shared window once.
    pub fn retained_texels(&self) -> u64 {
        self.entries
            .iter()
            .flat_map(|e| &e.requests)
            .map(|r| r.window.area())
            .sum()
    }

    fn entry(&mut self, image: &Arc<dyn ImageRows>, subset: TexelRect) -> usize {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.image, image) && entry.subset == subset)
        {
            return index;
        }
        let index = self.entries.len();
        self.entries.push(Entry {
            image: Arc::clone(image),
            subset,
            requests: Vec::new(),
            windows: Vec::new(),
        });
        index
    }
}

fn merge(requests: &mut Vec<Request>, request: Request) {
    match requests.iter_mut().find(|r| r.level == request.level) {
        Some(existing) => existing.window = existing.window.union(request.window),
        None => requests.push(request),
    }
}

fn texels(requests: &[Request]) -> Option<u64> {
    requests.iter().try_fold(0_u64, |sum, request| {
        u16::try_from(request.window.width().max(request.window.height())).ok()?;
        sum.checked_add(request.window.area())
    })
}

impl core::fmt::Debug for ImageCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImageCache")
            .field("entries", &self.entries.len())
            .field("streamed", &self.streamed)
            .field("retained", &self.retained_texels())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::PremulRgba8;
    use crate::kurbo::Rect;
    use alloc::vec;
    use core::sync::atomic::{AtomicU64, Ordering};

    struct Rows(AtomicU64);
    impl ImageRows for Rows {
        fn read(&self, x0: u32, y: u32, row: &mut [PremulRgba8]) -> bool {
            self.0.fetch_add(row.len() as u64, Ordering::Relaxed);
            for (i, pixel) in row.iter_mut().enumerate() {
                *pixel = PremulRgba8 {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: ((x0 as usize + i + y as usize) % 251) as u8,
                };
            }
            true
        }
    }

    fn plan(source: Rect, destination: Rect, area: Rect) -> ImagePlan {
        FilterImage::plan([64, 64], source, destination, destination, area).unwrap()
    }

    #[test]
    fn unioned_windows_stream_once_and_narrower_retries_reuse_them() {
        let rows = Arc::new(Rows(AtomicU64::new(0)));
        let source: Arc<dyn ImageRows> = rows.clone();
        let whole = Rect::new(0., 0., 64., 64.);
        let small = Rect::new(0., 0., 8., 8.);
        let left = Rect::new(0., 0., 2., 8.);
        let right = Rect::new(6., 0., 8., 8.);
        let mut cache = ImageCache::default();
        let images = cache
            .share_batch(
                vec![
                    (plan(whole, small, left), source.clone()),
                    (plan(whole, small, right), source.clone()),
                ],
                4096,
                4096,
            )
            .unwrap();
        assert_eq!(cache.streamed_texels(), 4096);
        assert_eq!(
            super::super::image::unique_retained(images.iter()) as u64,
            cache.retained_texels()
        );
        assert_eq!(rows.0.load(Ordering::Relaxed), 4096);
        let left_level = images[0].window(2).unwrap();
        let right_level = images[1].window(2).unwrap();
        // The two draws hold the same union, including its middle texels.
        assert_eq!(left_level.texel(7, 3), right_level.texel(7, 3));
        let independent = plan(whole, small, small).share(source.clone()).unwrap();
        for level in [2, 3] {
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(
                        images[0].window(level).unwrap().texel(x, y),
                        independent.window(level).unwrap().texel(x, y)
                    );
                }
            }
        }
        let before = rows.0.load(Ordering::Relaxed);
        cache
            .share_batch(vec![(plan(whole, small, left), source)], 4096, 4096)
            .unwrap();
        assert_eq!(rows.0.load(Ordering::Relaxed), before);
        assert_eq!(cache.streamed_texels(), 4096);
    }

    #[test]
    fn identities_subsets_levels_and_limits_are_independent() {
        let a: Arc<dyn ImageRows> = Arc::new(Rows(AtomicU64::new(0)));
        let b: Arc<dyn ImageRows> = Arc::new(Rows(AtomicU64::new(0)));
        let whole = Rect::new(0., 0., 64., 64.);
        let small = Rect::new(0., 0., 8., 8.);
        let smaller = Rect::new(0., 0., 4., 4.);
        let mut cache = ImageCache::default();
        // Different scales union levels and are one source stream.
        cache
            .share_batch(
                vec![
                    (plan(whole, small, small), a.clone()),
                    (plan(whole, smaller, smaller), a.clone()),
                ],
                4096,
                4096,
            )
            .unwrap();
        assert_eq!(cache.streamed_texels(), 4096);
        assert!(
            cache
                .share_batch(vec![(plan(whole, small, small), b.clone())], 4096, 4096)
                .is_err()
        );
        assert_eq!(cache.streamed_texels(), 4096);
        cache
            .share_batch(vec![(plan(whole, small, small), b)], 4096, 8192)
            .unwrap();
        assert_eq!(cache.streamed_texels(), 8192);
        // A source subset has its own mip chain and cannot use the full image's.
        let subset = Rect::new(16., 0., 48., 64.);
        let expected = plan(subset, small, small).share(a.clone()).unwrap();
        let actual = cache
            .share_batch(vec![(plan(subset, small, small), a)], 4096, 12288)
            .unwrap();
        for (level, _) in expected.sampling().levels() {
            assert_eq!(
                actual[0].window(level).unwrap().texel(0, 0),
                expected.window(level).unwrap().texel(0, 0)
            );
        }
        // Window admission happens before streaming any source.
        let rows = Arc::new(Rows(AtomicU64::new(0)));
        let source: Arc<dyn ImageRows> = rows.clone();
        let mut cache = ImageCache::default();
        assert!(
            cache
                .share_batch(vec![(plan(whole, small, small), source)], 1, 8192)
                .is_err()
        );
        assert_eq!(rows.0.load(Ordering::Relaxed), 0);
    }
}
