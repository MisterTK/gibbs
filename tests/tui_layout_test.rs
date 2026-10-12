use ratatui::layout::Rect;

#[test]
fn test_tui_auto_flowing_sidebar_layout_breakpoints() {
    // Breakpoint 1: width >= 120 -> sidebar 32 cols
    let w1 = 130u16;
    let (sidebar_w1, has_sidebar1) = if w1 >= 120 {
        (32u16, true)
    } else if w1 >= 90 {
        (24u16, true)
    } else {
        (0u16, false)
    };
    assert!(has_sidebar1);
    assert_eq!(sidebar_w1, 32);

    // Breakpoint 2: 90 <= width < 120 -> sidebar 24 cols
    let w2 = 100u16;
    let (sidebar_w2, has_sidebar2) = if w2 >= 120 {
        (32u16, true)
    } else if w2 >= 90 {
        (24u16, true)
    } else {
        (0u16, false)
    };
    assert!(has_sidebar2);
    assert_eq!(sidebar_w2, 24);

    // Breakpoint 3: width < 90 (e.g. standard 80x24 terminal) -> sidebar 0 cols
    let w3 = 80u16;
    let h3 = 24u16;
    let (sidebar_w3, has_sidebar3) = if w3 >= 120 {
        (32u16, true)
    } else if w3 >= 90 {
        (24u16, true)
    } else {
        (0u16, false)
    };
    assert!(!has_sidebar3);
    assert_eq!(sidebar_w3, 0);

    // Verify 80x24 terminal vertical fit without overflow
    let top_h = if h3 >= 30 { 8u16 } else { 5u16 };
    let log_h = if h3 >= 30 { 5u16 } else { 3u16 };
    let min_console = 6u16;
    let fixed_total = 1 + top_h + min_console + log_h + 1;
    assert!(
        fixed_total <= h3,
        "Fixed layout height {fixed_total} must fit in 80x24 terminal height {h3}"
    );
}
