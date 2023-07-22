use lazy_static::lazy_static;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

lazy_static! {
    // Load these once at the start of your program
    static ref SS: SyntaxSet = SyntaxSet::load_defaults_newlines();
    static ref TS: ThemeSet = ThemeSet::load_defaults();
}

const THEME: &str = "base16-ocean.dark";

/// Highlight and print the string.
/// Returns true if highlighting was performed, false otherwise.
fn highlight(s: &str, syntax: &str) -> bool {
    // Syntax highlight and print the table.
    let syntax = SS.find_syntax_by_name(syntax).unwrap();
    let mut h = HighlightLines::new(syntax, &TS.themes[THEME]);
    for line in LinesWithEndings::from(s) {
        let ranges: Vec<(Style, &str)> = h.highlight_line(line, &SS).unwrap();
        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
        print!("{}", escaped);
    }
    true
}

/// Colorize Morello IR.
pub fn morello(table: &str) -> bool {
    highlight(table, "Python")
}

/// Colorize Generated C code.
pub fn c(table: &str) -> bool {
    highlight(table, "C")
}
