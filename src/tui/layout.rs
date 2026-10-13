//! Terminal-size tiers and the sidebar `flow()` engine: components are placed by priority at
//! their minimum height, then grown toward their preferred and maximum heights.

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
    let n = columns.max(1);
    let cw = area.width.saturating_sub(n - 1) / n;
    let rects: Vec<Rect> = (0..n)
        .map(|i| Rect::new(area.x + i * (cw + 1), area.y, cw, area.height))
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
    for target in [|s: &Spec| s.pref_h, |s: &Spec| s.max_h] {
        for i in 0..placed.len() {
            let (s, c, cur) = placed[i];
            let want = target(s).max(cur);
            let grow = (want - cur).min(h - used[c]);
            used[c] += grow;
            placed[i].2 = cur + grow;
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

pub fn regions(area: Rect) -> Regions {
    let (x, y, w, h) = (area.x, area.y, area.width, area.height);
    let status = Rect::new(x, y, w, h.min(1));
    let keys_h = if h >= 2 { 1 } else { 0 };
    let keys = Rect::new(x, y + h - keys_h, w, keys_h);
    let mid_y = y + status.height;
    let mid_h = h - status.height - keys_h;
    // Short terminals get the dock layout whatever their width.
    let tier = if w < 100 || h < 20 {
        Tier::Compact
    } else if w < 160 {
        Tier::Standard
    } else {
        Tier::Wide
    };
    let side_w = match tier {
        Tier::Compact => 0,
        Tier::Standard => (w * 36 / 100).clamp(40, 64),
        // Two columns of (total − 1) / 2 and a one-cell gap, so `flow` splits the same way.
        Tier::Wide => {
            let total = (w * 40 / 100).clamp(84, 120);
            2 * ((total - 1) / 2) + 1
        }
    };
    if tier == Tier::Compact {
        let dock_h = (h / 3).min(8).min(mid_h);
        let main = Rect::new(x, mid_y, w, mid_h - dock_h);
        let dock = (dock_h > 0).then(|| Rect::new(x, mid_y + main.height, w, dock_h));
        return Regions {
            tier,
            status,
            main,
            sidebar: vec![],
            dock,
            keys,
        };
    }
    let main_w = w - side_w;
    let main = Rect::new(x, mid_y, main_w, mid_h);
    let sx = x + main_w;
    let sidebar = if tier == Tier::Wide {
        let c0 = (side_w - 1) / 2;
        vec![
            Rect::new(sx, mid_y, c0, mid_h),
            Rect::new(sx + c0 + 1, mid_y, c0, mid_h),
        ]
    } else {
        vec![Rect::new(sx, mid_y, side_w, mid_h)]
    };
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
                            assert!(rc.height <= s.max_h.max(s.pref_h));
                            assert!(
                                r.sidebar.iter().any(|col| inside(*rc, *col)),
                                "{c:?} {w}x{h}"
                            );
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
}
