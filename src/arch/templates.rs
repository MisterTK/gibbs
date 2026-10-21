//! Fallback chat templates (`arch/templates/*.jinja`) for tokenizers that declare none.

const TEMPLATES: [(&str, &str); 5] = [
    ("chatml", include_str!("../../arch/templates/chatml.jinja")),
    ("header", include_str!("../../arch/templates/header.jinja")),
    ("turn", include_str!("../../arch/templates/turn.jinja")),
    ("plain", include_str!("../../arch/templates/plain.jinja")),
    ("inst", include_str!("../../arch/templates/inst.jinja")),
];

const NAMES: [&str; 5] = ["chatml", "header", "turn", "plain", "inst"];

/// The Jinja source of fallback template `name`.
pub fn template(name: &str) -> Option<&'static str> {
    TEMPLATES.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Every fallback template name, in picker order.
pub fn names() -> &'static [&'static str] {
    &NAMES
}
