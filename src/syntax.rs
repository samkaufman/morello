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

const SYNTAX: &str = "Python";
const THEME: &str = "base16-ocean.dark";

#[derive(Clone, PartialEq, clap::ValueEnum)]
pub enum ColorMode {
    Never,
    Auto,
    Always,
}

/// Highlight and print the table.
/// Returns true if highlighting was performed, false otherwise.
pub fn highlight(table: &str, mut color: ColorMode) -> bool {
    // Decide whether to print in color.
    if color == ColorMode::Auto && !atty::is(atty::Stream::Stdout) {
        color = ColorMode::Never;
    }

    // Print the table without syntax highlighting.
    if color == ColorMode::Never {
        return false;
    }

    // Syntax highlight and print the table.
    let syntax = SS.find_syntax_by_name(SYNTAX).unwrap();
    let mut h = HighlightLines::new(syntax, &TS.themes[THEME]);
    for line in LinesWithEndings::from(table) {
        let ranges: Vec<(Style, &str)> = h.highlight_line(line, &SS).unwrap();
        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
        print!("{}", escaped);
    }
    true
}
