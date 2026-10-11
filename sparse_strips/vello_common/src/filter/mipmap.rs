// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Windows of Skia's mip levels of an image subset, as `SkMipmap::Build` makes them.
//!
//! Each level halves the one before, rounding down (at least one texel). A
//! texel averages a 2×2 block; along an odd side it takes three texels weighted
//! 1, 2, 1, so the level still spans the whole side. The weighted sum of the
//! premultiplied 8-bit channels is shifted down, truncating.
//!
//! Only the requested windows are kept. Rows stream through the levels, so
//! memory stays at three rows per level whatever the subset's size, and each
//! level computes only the texels its successors or a window read.

use super::image::ImageRows;
use super::image_sampling::{TexelRect, level_size};
use crate::color::PremulRgba8;
use crate::pixmap::Pixmap;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

/// Texels of mip `level` of a subset, relative to the subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Request {
    pub level: u8,
    pub window: TexelRect,
}

/// The requested windows of the mip levels of `subset` (in `rows`' texels),
/// in request order. `None` when `rows` cannot supply a texel, or a window is
/// wider or taller than a pixmap.
pub(crate) fn build(
    rows: &dyn ImageRows,
    subset: TexelRect,
    requests: &[Request],
) -> Option<Vec<Pixmap>> {
    let mut outputs = requests
        .iter()
        .map(|request| {
            let width = u16::try_from(request.window.width()).ok()?;
            let height = u16::try_from(request.window.height()).ok()?;
            Some(Pixmap::new(width, height))
        })
        .collect::<Option<Vec<_>>>()?;
    let size = [subset.width(), subset.height()];
    let windows = plan_windows(size, requests);
    let Some(&base) = windows.first() else {
        return Some(outputs);
    };
    let mut levels: Vec<Level> = windows
        .into_iter()
        .enumerate()
        .map(|(index, window)| Level::new(index, size, window))
        .collect();
    let mut row = vec![PremulRgba8::from_u32(0); base.width() as usize];
    for y in base.y0..base.y1 {
        if !rows.read(subset.x0 + base.x0, subset.y0 + y, &mut row) {
            return None;
        }
        levels[0].push(y, &row, requests, &mut outputs);
        cascade(&mut levels, requests, &mut outputs)?;
    }
    Some(outputs)
}

/// The texels of the subset (of `size`) that building `requests` reads.
pub(crate) fn base_window(size: [u32; 2], requests: &[Request]) -> TexelRect {
    plan_windows(size, requests)
        .first()
        .copied()
        .unwrap_or(EMPTY)
}

const EMPTY: TexelRect = TexelRect {
    x0: 0,
    y0: 0,
    x1: 0,
    y1: 0,
};

/// Each level's window, finest first, up to the coarsest request: its own
/// requests plus what the next level averages.
fn plan_windows(size: [u32; 2], requests: &[Request]) -> Vec<TexelRect> {
    let Some(coarsest) = requests.iter().map(|r| r.level).max() else {
        return Vec::new();
    };
    let mut windows = vec![EMPTY; usize::from(coarsest) + 1];
    for level in (0..=coarsest).rev() {
        let mut window = requests
            .iter()
            .filter(|r| r.level == level)
            .fold(EMPTY, |window, r| window.union(r.window));
        if let Some(coarser) = windows.get(usize::from(level) + 1)
            && coarser.area() > 0
        {
            window = window.union(sources(*coarser, level_size(size, level)));
        }
        windows[usize::from(level)] = window;
    }
    windows
}

/// The texels of a level of `size` that `window` of the next level averages.
fn sources(window: TexelRect, size: [u32; 2]) -> TexelRect {
    let [tx, ty] = size.map(|side| taps(side).len() as u32);
    TexelRect {
        x0: 2 * window.x0,
        y0: 2 * window.y0,
        x1: (2 * (window.x1 - 1) + tx).min(size[0]),
        y1: (2 * (window.y1 - 1) + ty).min(size[1]),
    }
}

/// Weights along a side of `side` texels: one texel, a pair, or three.
fn taps(side: u32) -> &'static [u32] {
    match side {
        1 => &[1],
        side if side % 2 == 1 => &[1, 2, 1],
        _ => &[1, 1],
    }
}

/// One level's streaming state.
struct Level {
    /// The level's index; 0 is the subset.
    level: u8,
    /// The level's size.
    size: [u32; 2],
    /// The texels computed: what windows read or coarser levels need.
    window: TexelRect,
    /// The next row to compute.
    next: u32,
    /// The latest rows, at most three, with their indices.
    rows: VecDeque<(u32, Vec<PremulRgba8>)>,
}

impl Level {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Levels come from u8 requests."
    )]
    fn new(index: usize, subset: [u32; 2], window: TexelRect) -> Self {
        let level = index as u8;
        Self {
            level,
            size: level_size(subset, level),
            window,
            next: window.y0,
            rows: VecDeque::with_capacity(3),
        }
    }

    /// Keep row `y` and copy it into the windows requesting it.
    fn push(&mut self, y: u32, row: &[PremulRgba8], requests: &[Request], outputs: &mut [Pixmap]) {
        for (request, output) in requests.iter().zip(outputs.iter_mut()) {
            let window = request.window;
            if request.level != self.level || !(window.y0..window.y1).contains(&y) {
                continue;
            }
            let from = (window.x0 - self.window.x0) as usize;
            let width = window.width() as usize;
            let start = (y - window.y0) as usize * width;
            output.data_mut()[start..start + width].copy_from_slice(&row[from..from + width]);
        }
        if self.rows.len() == 3 {
            self.rows.pop_front();
        }
        self.rows.push_back((y, row.to_vec()));
        self.next = y + 1;
    }

    fn row(&self, y: u32) -> Option<&[PremulRgba8]> {
        self.rows
            .iter()
            .find(|(index, _)| *index == y)
            .map(|(_, row)| row.as_slice())
    }
}

/// Compute every coarser row whose sources have arrived. `None` when a level
/// lacks rows or texels the next one averages, which a consistent plan rules
/// out.
fn cascade(levels: &mut [Level], requests: &[Request], outputs: &mut [Pixmap]) -> Option<()> {
    for index in 1..levels.len() {
        let (finer, coarser) = levels.split_at_mut(index);
        let (finer, coarser) = (&finer[index - 1], &mut coarser[0]);
        let ty = taps(finer.size[1]);
        while coarser.next < coarser.window.y1 {
            let y = coarser.next;
            let last = 2 * y + ty.len() as u32 - 1;
            if finer.next <= last {
                break;
            }
            let row = downsample(finer, coarser.window, y)?;
            coarser.push(y, &row, requests, outputs);
        }
    }
    Some(())
}

/// Row `y` of the next level, over `window`'s columns, from `finer`'s rows.
fn downsample(finer: &Level, window: TexelRect, y: u32) -> Option<Vec<PremulRgba8>> {
    let tx = taps(finer.size[0]);
    let ty = taps(finer.size[1]);
    let divisor: u32 = tx.iter().sum::<u32>() * ty.iter().sum::<u32>();
    let sources = ty
        .iter()
        .enumerate()
        .map(|(dy, &weight)| Some((weight, finer.row(2 * y + dy as u32)?)))
        .collect::<Option<Vec<_>>>()?;
    let first = (2 * window.x0).checked_sub(finer.window.x0)? as usize;
    let mut row = Vec::with_capacity(window.width() as usize);
    for column in (first..).step_by(2).take(window.width() as usize) {
        let mut sum = [0_u32; 4];
        for &(wy, source) in &sources {
            let texels = source.get(column..column + tx.len())?;
            for (texel, &wx) in texels.iter().zip(tx) {
                let weight = wx * wy;
                for (s, c) in sum.iter_mut().zip([texel.r, texel.g, texel.b, texel.a]) {
                    *s += u32::from(c) * weight;
                }
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "A weighted average of bytes fits a byte."
        )]
        let [r, g, b, a] = sum.map(|s| (s / divisor) as u8);
        row.push(PremulRgba8 { r, g, b, a });
    }
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-row alpha image read from a slice.
    struct Alphas<'a>(&'a [u8], u32);

    impl ImageRows for Alphas<'_> {
        fn read(&self, x0: u32, y: u32, row: &mut [PremulRgba8]) -> bool {
            for (i, texel) in row.iter_mut().enumerate() {
                let index = (y * self.1 + x0) as usize + i;
                let Some(&a) = self.0.get(index) else {
                    return false;
                };
                *texel = PremulRgba8 {
                    r: 0,
                    g: 0,
                    b: 0,
                    a,
                };
            }
            true
        }
    }

    fn whole(width: u32, height: u32) -> TexelRect {
        TexelRect {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    fn alphas(pixmap: &Pixmap) -> Vec<u8> {
        pixmap.data().iter().map(|p| p.a).collect()
    }

    #[test]
    fn even_sides_average_pairs_and_odd_sides_weight_three() {
        let image = [255, 0, 0, 0, 7, 9, 0];
        let rows = Alphas(&image, 7);
        // Seven texels make three, weighting 1, 2, 1 over texels 0..3, 2..5
        // and 4..7, truncated: 255 / 4, 7 / 4 and (7 + 2 * 9) / 4.
        let levels = build(
            &rows,
            whole(7, 1),
            &[Request {
                level: 1,
                window: whole(3, 1),
            }],
        )
        .unwrap();
        assert_eq!(alphas(&levels[0]), [63, 1, 6]);
        // Four texels make two, then one: pairs, truncated.
        let rows = Alphas(&[255, 0, 3, 0], 4);
        let request = |level, width| Request {
            level,
            window: whole(width, 1),
        };
        let levels = build(&rows, whole(4, 1), &[request(1, 2), request(2, 1)]).unwrap();
        assert_eq!(alphas(&levels[0]), [127, 1]);
        assert_eq!(alphas(&levels[1]), [64]);
    }

    #[test]
    fn windows_match_whole_levels() {
        // A 9x6 image, odd by even, and its levels 1 (4x3) and 2 (2x1).
        let image: Vec<u8> = (0..54).map(|i| ((i * 37) % 251) as u8).collect();
        let rows = Alphas(&image, 9);
        let all = build(
            &rows,
            whole(9, 6),
            &[
                Request {
                    level: 1,
                    window: whole(4, 3),
                },
                Request {
                    level: 2,
                    window: whole(2, 1),
                },
            ],
        )
        .unwrap();
        let window = TexelRect {
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
        };
        let part = build(&rows, whole(9, 6), &[Request { level: 1, window }]).unwrap();
        let whole_level = alphas(&all[0]);
        let expected: Vec<u8> = (1..3)
            .flat_map(|y| (1..3).map(move |x| (x, y)))
            .map(|(x, y)| whole_level[y * 4 + x])
            .collect();
        assert_eq!(alphas(&part[0]), expected);
        // A window of an offset subset reads the texels after its origin.
        let shifted = build(
            &rows,
            TexelRect {
                x0: 1,
                y0: 0,
                x1: 9,
                y1: 6,
            },
            &[Request {
                level: 0,
                window: whole(1, 1),
            }],
        )
        .unwrap();
        assert_eq!(alphas(&shifted[0]), [image[1]]);
    }

    #[test]
    fn short_rows_and_oversized_windows_build_nothing() {
        let rows = Alphas(&[1, 2, 3], 4);
        let request = Request {
            level: 1,
            window: whole(2, 1),
        };
        assert!(build(&rows, whole(4, 1), &[request]).is_none());
        let wide = Request {
            level: 0,
            window: whole(70_000, 1),
        };
        assert!(build(&rows, whole(70_000, 1), &[wide]).is_none());
    }
}
