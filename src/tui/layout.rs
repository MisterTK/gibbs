//! Terminal-size tiers, 1D `flex()` solver, and the sidebar `flow()` engine: regions and
//! columns resolve `min`/`basis`/`max` with `grow`/`shrink` weights instead of raw percentages.

use ratatui::layout::Rect;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comp {
    Avatar,
    Roofline,
    Gap,
    Tokens,
    Latency,
    Logprobs,
    Weights,
    Memory,
    Device,
    Kernels,
    Load,
}

#[derive(Clone, Copy, Debug)]
pub struct Flex {
    pub min: u16,
    pub basis: u16,
    pub max: u16,
    pub grow: u16,
    pub shrink: u16,
}

impl Flex {
    pub const fn new(min: u16, basis: u16, max: u16, grow: u16, shrink: u16) -> Self {
        Self {
            min,
            basis,
            max,
            grow,
            shrink,
        }
    }
}

/// Resolves 1D flexible lengths (CSS Flexbox §9.7) within `avail` with `gap` between active items.
/// Returns `(offset, size)` in `0..=avail` for each item in `items`.
pub fn flex(avail: u16, gap: u16, items: &[Flex]) -> Vec<(u16, u16)> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut sz: Vec<u16> = items
        .iter()
        .map(|it| it.basis.clamp(it.min, it.max.max(it.min)))
        .collect();
    let active_gaps = |sz: &[u16]| -> u16 {
        let n = sz.iter().filter(|&&s| s > 0).count();
        gap.saturating_mul(n.saturating_sub(1) as u16)
    };
    let space = avail.saturating_sub(gap.saturating_mul(items.len().saturating_sub(1) as u16));

    // Grow from basis toward max proportionally to `grow`.
    loop {
        let used: u16 = sz.iter().sum();
        if used >= space {
            break;
        }
        let rem = space - used;
        let total_grow: u32 = items
            .iter()
            .enumerate()
            .filter(|&(i, it)| sz[i] < it.max.max(it.min) && it.grow > 0)
            .map(|(_, it)| u32::from(it.grow))
            .sum();
        if total_grow == 0 {
            break;
        }
        let mut hit_cap = false;
        let mut rems: Vec<(u32, u16, usize)> = Vec::new();
        let mut used_step = 0u16;
        for (i, it) in items.iter().enumerate() {
            let cap = it.max.max(it.min).saturating_sub(sz[i]);
            if cap == 0 || it.grow == 0 {
                continue;
            }
            let num = u32::from(rem) * u32::from(it.grow);
            let add = (num / total_grow) as u16;
            if add >= cap {
                sz[i] += cap;
                used_step += cap;
                hit_cap = true;
            } else {
                sz[i] += add;
                used_step += add;
                rems.push((num % total_grow, it.grow, i));
            }
        }
        if !hit_cap {
            rems.sort_by_key(|&(r, g, i)| (std::cmp::Reverse(r), std::cmp::Reverse(g), i));
            let left = usize::from(rem.saturating_sub(used_step));
            for (_, _, i) in rems.into_iter().take(left) {
                sz[i] += 1;
            }
            break;
        }
    }

    // Shrink from basis toward min proportionally to `shrink`.
    loop {
        let used: u16 = sz.iter().sum();
        if used <= space {
            break;
        }
        let over = used - space;
        let total_shrink: u32 = items
            .iter()
            .enumerate()
            .filter(|&(i, it)| sz[i] > it.min && it.shrink > 0)
            .map(|(_, it)| u32::from(it.shrink))
            .sum();
        if total_shrink == 0 {
            break;
        }
        let mut hit_floor = false;
        let mut rems: Vec<(u32, u16, usize)> = Vec::new();
        let mut cut_step = 0u16;
        for (i, it) in items.iter().enumerate() {
            let room = sz[i].saturating_sub(it.min);
            if room == 0 || it.shrink == 0 {
                continue;
            }
            let num = u32::from(over) * u32::from(it.shrink);
            let cut = (num / total_shrink) as u16;
            if cut >= room {
                sz[i] -= room;
                cut_step += room;
                hit_floor = true;
            } else {
                sz[i] -= cut;
                cut_step += cut;
                rems.push((num % total_shrink, it.shrink, i));
            }
        }
        if !hit_floor {
            rems.sort_by_key(|&(r, s, i)| {
                (
                    std::cmp::Reverse(r),
                    std::cmp::Reverse(s),
                    std::cmp::Reverse(i),
                )
            });
            let left = usize::from(over.saturating_sub(cut_step));
            for (_, _, i) in rems.into_iter().take(left) {
                sz[i] -= 1;
            }
            break;
        }
    }
    // Hard clamp from the tail if sum(min) + gaps still exceeds `avail`.
    for i in (0..sz.len()).rev() {
        let total = sz.iter().sum::<u16>().saturating_add(active_gaps(&sz));
        if total <= avail {
            break;
        }
        let cut = (total - avail).min(sz[i]);
        sz[i] -= cut;
    }

    let mut pos = 0u16;
    let mut seen_nonzero = false;
    sz.into_iter()
        .map(|s| {
            if s > 0 && seen_nonzero {
                pos = pos.saturating_add(gap).min(avail);
            }
            let start = pos;
            let len = s.min(avail.saturating_sub(start));
            pos = start + len;
            if len > 0 {
                seen_nonzero = true;
            }
            (start, len)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub comp: Comp,
    pub min_h: u16,
    pub pref_h: u16,
    pub max_h: u16,
    pub priority: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub placed: Vec<(Comp, Rect)>,
    pub hidden: usize,
}

impl Plan {
    #[cfg(test)]
    pub fn rect(&self, c: Comp) -> Option<Rect> {
        self.placed.iter().find(|(k, _)| *k == c).map(|(_, r)| *r)
    }
}

const fn spec(comp: Comp, min_h: u16, pref_h: u16, max_h: u16, priority: u8) -> Spec {
    Spec {
        comp,
        min_h,
        pref_h,
        max_h,
        priority,
    }
}

/// The fixed spec table. `Load` is present only while loading; `Avatar` only when the column
/// is at least 24 wide and the terminal at least 44 rows. A `focused` component becomes
/// priority 0 and prefers up to 24 rows.
pub fn default_specs(col_w: u16, area_h: u16, loading: bool, focused: Option<Comp>) -> Vec<Spec> {
    let mut v = vec![
        spec(Comp::Roofline, 6, 8, 12, 1),
        spec(Comp::Gap, 4, 6, 7, 2),
        spec(Comp::Tokens, 5, 6, 8, 3),
        spec(Comp::Latency, 5, 6, 8, 4),
        spec(Comp::Logprobs, 6, 8, 12, 5),
        spec(Comp::Weights, 5, 7, 12, 6),
        spec(Comp::Memory, 4, 4, 6, 8),
        spec(Comp::Device, 4, 4, 6, 9),
        spec(Comp::Kernels, 6, 10, 14, 10),
    ];
    if loading {
        v.push(spec(Comp::Load, 7, 7, 7, 0));
    }
    if col_w >= 24 && area_h >= 44 {
        v.push(spec(Comp::Avatar, 12, 16, 30, 7));
    }
    if let Some(f) = focused {
        for s in v.iter_mut().filter(|s| s.comp == f) {
            s.priority = 0;
            s.pref_h = s.pref_h.max(area_h.min(24));
            s.max_h = s.max_h.max(s.pref_h);
        }
    }
    v
}

/// Places `specs` into `columns` equal columns (1 cell apart) of `area`.
pub fn flow(specs: &[Spec], area: Rect, columns: u16) -> Plan {
    let n = usize::from(columns.max(1));
    let items: Vec<Flex> = vec![Flex::new(0, 1, u16::MAX, 1, 1); n];
    let rects: Vec<Rect> = flex(area.width, 1, &items)
        .into_iter()
        .map(|(dx, cw)| Rect::new(area.x + dx, area.y, cw, area.height))
        .collect();
    flow_in(specs, &rects)
}

/// [`flow`] over explicit column rectangles (same y and height).
fn flow_in(specs: &[Spec], cols: &[Rect]) -> Plan {
    let Some(first) = cols.first() else {
        return Plan {
            placed: vec![],
            hidden: specs.len(),
        };
    };
    let h = first.height;
    let mut order: Vec<&Spec> = specs.iter().collect();
    order.sort_by_key(|s| s.priority); // stable
    let mut placed: Vec<(&Spec, usize, u16)> = Vec::new(); // (spec, column, height)
    let mut used = vec![0u16; cols.len()];
    for s in order {
        // The emptiest column the component fits in (the leftmost on ties); with one column this
        // is plain first-fit, with two it keeps both populated instead of filling the first.
        let fits = |c: &usize| cols[*c].width > 0 && used[*c] + s.min_h <= h;
        if let Some(c) = (0..cols.len()).filter(fits).min_by_key(|&c| used[c]) {
            used[c] += s.min_h;
            placed.push((s, c, s.min_h));
        }
    }
    let hidden = specs.len() - placed.len();
    // Pass 1: priority-ordered shrink/grow between `min_h` and `pref_h` (flex-basis).
    for (s, c, cur) in &mut placed {
        let want = s.pref_h.max(*cur);
        let grow = (want - *cur).min(h - used[*c]);
        used[*c] += grow;
        *cur += grow;
    }
    // Pass 2: proportional flexbox grow from `pref_h` (`basis`) toward `max_h` per column.
    // Pass 3: when the column still has spare height (`col_used < h`), distribute 100% of the
    // remaining vertical space across the column's placed items (`flex: 1 1 auto` to 100% height).
    for col_idx in 0..cols.len() {
        let indices: Vec<usize> = (0..placed.len())
            .filter(|&i| placed[i].1 == col_idx)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let items: Vec<Flex> = indices
            .iter()
            .map(|&i| {
                let (s, _, cur) = placed[i];
                let max_h = s.max_h.max(cur);
                let grow = max_h.saturating_sub(cur).max(1);
                Flex::new(cur, cur, max_h, grow, 1)
            })
            .collect();
        for (&idx, (_, sz)) in indices.iter().zip(flex(h, 0, &items)) {
            placed[idx].2 = sz;
        }
        let col_used: u16 = indices.iter().map(|&i| placed[i].2).sum();
        if col_used < h {
            let uncapped: Vec<Flex> = indices
                .iter()
                .map(|&i| {
                    let (s, _, cur) = placed[i];
                    Flex::new(cur, cur, u16::MAX, s.pref_h.max(1), 1)
                })
                .collect();
            for (idx, (_, sz)) in indices.into_iter().zip(flex(h, 0, &uncapped)) {
                placed[idx].2 = sz;
            }
        }
    }
    let mut y = vec![first.y; cols.len()];
    let mut out = Vec::with_capacity(placed.len());
    for (s, c, ph) in placed {
        out.push((s.comp, Rect::new(cols[c].x, y[c], cols[c].width, ph)));
        y[c] += ph;
    }
    Plan {
        placed: out,
        hidden,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Compact,
    Standard,
    Wide,
}

#[derive(Clone, Debug)]
pub struct Regions {
    pub tier: Tier,
    pub status: Rect,
    pub main: Rect,
    /// Flow columns (empty in the Compact tier).
    pub sidebar: Vec<Rect>,
    pub dock: Option<Rect>,
    pub keys: Rect,
}

impl Regions {
    /// The sidebar columns and the gap between them as one rectangle, as `flow` takes them.
    pub fn sidebar_area(&self) -> Option<Rect> {
        self.sidebar.iter().copied().reduce(|a, b| a.union(b))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MainStack {
    pub detail: Option<Rect>,
    pub requests: Option<Rect>,
    pub console: Rect,
    pub log: Rect,
}

/// Splits the main column vertically with `flex()` instead of fixed height thresholds.
pub fn main_stack(
    area: Rect,
    show_requests: bool,
    loading_compact: bool,
    detail_compact: bool,
) -> MainStack {
    let want_detail = detail_compact && area.height >= 9;
    let detail_flex = if want_detail {
        Flex::new(8, 12, 24, 2, 1)
    } else {
        Flex::new(0, 0, 0, 0, 0)
    };
    let req_flex = if !show_requests || want_detail {
        Flex::new(0, 0, 0, 0, 0)
    } else if loading_compact {
        Flex::new(5, 7, 7, 0, 1)
    } else {
        Flex::new(4, 5, 8, 1, 1)
    };
    let console_flex = Flex::new(3, 12, u16::MAX, 6, 1);
    let log_flex = Flex::new(2, 3, 5, 1, 1);
    let spans = flex(
        area.height,
        0,
        &[detail_flex, req_flex, console_flex, log_flex],
    );
    let rect = |(dy, dh): (u16, u16)| Rect::new(area.x, area.y + dy, area.width, dh);
    MainStack {
        detail: (spans[0].1 > 0).then(|| rect(spans[0])),
        requests: (spans[1].1 > 0).then(|| rect(spans[1])),
        console: rect(spans[2]),
        log: rect(spans[3]),
    }
}

pub fn regions(area: Rect) -> Regions {
    let (x, y, w, h) = (area.x, area.y, area.width, area.height);
    // Short or narrow terminals use the single-column Compact flex layout with a bottom dock.
    let tier = if w < 100 || h < 20 {
        Tier::Compact
    } else if w < 160 {
        Tier::Standard
    } else {
        Tier::Wide
    };

    let status_min = h.min(1);
    let keys_min = u16::from(h >= 2);
    if tier == Tier::Compact {
        let v = flex(
            h,
            0,
            &[
                Flex::new(status_min, status_min, 1, 0, 0),
                Flex::new(0, 12, u16::MAX, 2, 1),
                Flex::new(0, 3, 8, 1, 2),
                Flex::new(keys_min, keys_min, 1, 0, 0),
            ],
        );
        let vrect = |(dy, dh): (u16, u16)| Rect::new(x, y + dy, w, dh);
        let dock = (v[2].1 > 0).then(|| vrect(v[2]));
        return Regions {
            tier,
            status: vrect(v[0]),
            main: vrect(v[1]),
            sidebar: vec![],
            dock,
            keys: vrect(v[3]),
        };
    }

    let v = flex(
        h,
        0,
        &[
            Flex::new(status_min, status_min, 1, 0, 0),
            Flex::new(0, 18, u16::MAX, 1, 1),
            Flex::new(keys_min, keys_min, 1, 0, 0),
        ],
    );
    let status = Rect::new(x, y + v[0].0, w, v[0].1);
    let (mid_y, mid_h) = (y + v[1].0, v[1].1);
    let keys = Rect::new(x, y + v[2].0, w, v[2].1);

    let sidebar = if tier == Tier::Wide {
        // Two sidebar columns (min 41, max 60 each, 1-cell gap) and main column solved via flex.
        let h_spans = flex(
            w,
            0,
            &[
                Flex::new(76, 76, u16::MAX, 15, 1),
                Flex::new(83, 84, 120, 10, 1),
            ],
        );
        let c0 = (h_spans[1].1.saturating_sub(1)) / 2;
        let side_w = 2 * c0 + 1;
        let sx = x + w - side_w;
        let cols = flex(
            side_w,
            1,
            &[Flex::new(c0, c0, c0, 1, 1), Flex::new(c0, c0, c0, 1, 1)],
        );
        vec![
            Rect::new(sx + cols[0].0, mid_y, cols[0].1, mid_h),
            Rect::new(sx + cols[1].0, mid_y, cols[1].1, mid_h),
        ]
    } else {
        let h_spans = flex(
            w,
            0,
            &[
                Flex::new(60, 60, u16::MAX, 16, 1),
                Flex::new(40, 40, 64, 9, 1),
            ],
        );
        vec![Rect::new(x + h_spans[1].0, mid_y, h_spans[1].1, mid_h)]
    };
    let side_w = sidebar
        .iter()
        .copied()
        .reduce(|a, b| a.union(b))
        .map_or(0, |r| r.width);
    let main = Rect::new(x, mid_y, w - side_w, mid_h);
    Regions {
        tier,
        status,
        main,
        sidebar,
        dock: None,
        keys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside(r: Rect, a: Rect) -> bool {
        r.x >= a.x && r.y >= a.y && r.right() <= a.right() && r.bottom() <= a.bottom()
    }

    fn disjoint(rs: &[Rect]) -> bool {
        rs.iter().enumerate().all(|(i, a)| {
            rs[i + 1..].iter().all(|b| {
                let i = a.intersection(*b);
                i.width == 0 || i.height == 0
            })
        })
    }

    #[test]
    fn regions_and_flow_hold_for_every_size() {
        for w in 20..=220u16 {
            for h in 6..=80u16 {
                let area = Rect::new(0, 0, w, h);
                let r = regions(area);
                let mut all = vec![r.status, r.main, r.keys];
                all.extend(r.sidebar.iter().copied());
                all.extend(r.dock);
                all.retain(|a| a.width > 0 && a.height > 0);
                for a in &all {
                    assert!(inside(*a, area), "{w}x{h}: {a:?}");
                }
                assert!(disjoint(&all), "{w}x{h}: {all:?}");
                if h < 20 {
                    assert!(r.sidebar.is_empty(), "{w}x{h}");
                }
                for focused in [None, Some(Comp::Logprobs), Some(Comp::Kernels)] {
                    for loading in [false, true] {
                        let cw = r.sidebar.first().map_or(0, |c| c.width);
                        let specs = default_specs(cw, h, loading, focused);
                        let plan = flow_in(&specs, &r.sidebar);
                        assert_eq!(plan.placed.len() + plan.hidden, specs.len());
                        let rects: Vec<Rect> = plan.placed.iter().map(|p| p.1).collect();
                        for (c, rc) in &plan.placed {
                            let s = specs.iter().find(|s| s.comp == *c).unwrap();
                            assert!(rc.height >= s.min_h, "{c:?} {w}x{h}");
                            assert!(
                                r.sidebar.iter().any(|col| inside(*rc, *col)),
                                "{c:?} {w}x{h}"
                            );
                        }
                        for col in &r.sidebar {
                            let col_rects: Vec<Rect> = plan
                                .placed
                                .iter()
                                .map(|p| p.1)
                                .filter(|rc| rc.x == col.x && rc.width == col.width)
                                .collect();
                            if !col_rects.is_empty() {
                                let sum_h: u16 = col_rects.iter().map(|rc| rc.height).sum();
                                assert_eq!(
                                    sum_h, col.height,
                                    "sidebar column in {w}x{h} did not fill 100% height"
                                );
                                assert_eq!(
                                    col_rects.last().unwrap().bottom(),
                                    col.bottom(),
                                    "sidebar column bottom mismatch in {w}x{h}"
                                );
                            }
                        }
                        assert!(disjoint(&rects), "{w}x{h}: {rects:?}");
                        // The one-call form over a single area obeys the same invariants.
                        let p2 = flow(&specs, area, 1 + (w % 2));
                        assert_eq!(p2.placed.len() + p2.hidden, specs.len());
                        let r2: Vec<Rect> = p2.placed.iter().map(|p| p.1).collect();
                        assert!(r2.iter().all(|a| inside(*a, area)));
                        assert!(disjoint(&r2));
                    }
                }
            }
        }
    }

    #[test]
    fn short_terminal_has_no_sidebar_and_priority_order_wins() {
        let r = regions(Rect::new(0, 0, 200, 19));
        assert!(r.sidebar.is_empty());
        // 17 rows: Roofline (6), Gap (4), Tokens (5) fit at min; nothing else fits.
        let specs = default_specs(40, 30, false, None);
        let plan = flow(&specs, Rect::new(0, 0, 40, 17), 1);
        assert!(plan.rect(Comp::Gap).is_some() && plan.rect(Comp::Tokens).is_some());
        assert!(plan.rect(Comp::Latency).is_none());
        // The two spare rows go to Roofline first (min 6 → 8).
        assert_eq!(plan.rect(Comp::Roofline).unwrap().height, 8);
        assert_eq!(plan.rect(Comp::Gap).unwrap().height, 4);
        assert_eq!(plan.rect(Comp::Tokens).unwrap().height, 5);
        // Kernels ranks last: it only appears once everything else has its minimum.
        let tall = flow(&specs, Rect::new(0, 0, 40, 60), 1);
        assert!(tall.rect(Comp::Kernels).is_some());
        let tight = flow(&specs, Rect::new(0, 0, 40, 39), 1);
        assert!(tight.rect(Comp::Kernels).is_none() && tight.rect(Comp::Device).is_some());
    }

    #[test]
    fn load_is_priority_zero_and_avatar_needs_room() {
        assert_eq!(
            default_specs(40, 50, true, None)
                .iter()
                .min_by_key(|s| s.priority)
                .unwrap()
                .comp,
            Comp::Load
        );
        assert!(default_specs(40, 43, false, None)
            .iter()
            .all(|s| s.comp != Comp::Avatar));
        assert!(default_specs(23, 60, false, None)
            .iter()
            .all(|s| s.comp != Comp::Avatar));
        assert!(default_specs(24, 44, false, None)
            .iter()
            .any(|s| s.comp == Comp::Avatar));
    }

    #[test]
    fn flex_grows_and_shrinks_by_weights_and_caps() {
        // Surplus distributes by `grow`, freezes at `max`, and spills remainder to uncapped items.
        let grown = flex(
            100,
            0,
            &[
                Flex::new(40, 40, 50, 1, 1),
                Flex::new(40, 40, u16::MAX, 1, 1),
            ],
        );
        assert_eq!(grown, vec![(0, 50), (50, 50)]);
        // Deficit shrinks proportionally by `shrink`, respecting `min`.
        let shrunk = flex(
            20,
            0,
            &[Flex::new(6, 12, 20, 1, 1), Flex::new(4, 12, 20, 1, 3)],
        );
        assert_eq!(shrunk, vec![(0, 11), (11, 9)]);
        // Sidebar pass 2 distributes spare height across multiple components instead of first-wins.
        let specs = default_specs(40, 30, false, None);
        let roomy = flow(&specs[..2], Rect::new(0, 0, 40, 17), 1);
        assert_eq!(roomy.rect(Comp::Roofline).unwrap().height, 10);
        assert_eq!(roomy.rect(Comp::Gap).unwrap().height, 7);
    }
}
