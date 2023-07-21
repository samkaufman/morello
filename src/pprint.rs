use crate::table::DbImpl;
use crate::target::Target;
use crate::views::View;
use crate::{imp::Impl, views::Param};

use by_address::ByThinAddress;
use lazy_static::lazy_static;
use prettytable::{self, format, row, Cell};
use std::collections::HashMap;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

lazy_static! {
    // Load these once at the start of your program
    static ref SS: SyntaxSet = SyntaxSet::load_defaults_newlines();
    static ref TS: ThemeSet = ThemeSet::load_defaults();
}

const fn ascii_lower() -> [char; 26] {
    let mut result = ['a'; 26];

    let mut c: u8 = 'a' as u8;
    while c <= 'z' as u8 {
        result[(c - 97) as usize] = c as char;
        c += 1;
    }
    result
}
static ASCII_LOWER: [char; 26] = ascii_lower();

pub struct NameEnv<'t, Tgt: Target> {
    names: HashMap<ByThinAddress<&'t dyn View<Tgt = Tgt>>, String>,
}

impl<'t, Tgt: Target> NameEnv<'t, Tgt> {
    pub fn name(&mut self, view: &'t dyn View<Tgt = Tgt>) -> &str {
        let view_by_address = ByThinAddress(view);
        let cnt = self.names.len();
        let name = self
            .names
            .entry(view_by_address)
            .or_insert_with(|| String::from(ASCII_LOWER[cnt]));
        name
    }

    pub fn get_name(&self, view: &'t dyn View<Tgt = Tgt>) -> Option<&str> {
        let view_by_address = ByThinAddress(view);
        self.names.get(&view_by_address).map(|s| s.as_str())
    }

    pub fn get_name_or_display(&self, view: &'t dyn View<Tgt = Tgt>) -> String {
        if let Some(present_name) = self.get_name(view) {
            present_name.to_owned()
        } else if let Some(param) = view.to_param() {
            param.to_string()
        } else {
            panic!("No name for non-Param view: {:?}", view);
        }
    }
}

#[derive(Clone, PartialEq, clap::ValueEnum)]
pub enum ColorMode {
    Never,
    Auto,
    Always,
}

pub fn pprint<Tgt: Target>(root: &DbImpl<Tgt>, mut color: ColorMode) {
    let mut name_env = NameEnv {
        names: HashMap::new(),
    };

    // Set up table
    let mut table = prettytable::Table::new();
    let mut titles = row!["", "Logical Spec"];
    for level in Tgt::levels() {
        titles.add_cell(Cell::new(&level.to_string()));
    }
    titles.add_cell(Cell::new("Cost"));

    table.set_titles(titles);

    // Traverse the Impl.
    let problem = &root.aux().as_ref().unwrap().0;
    let args = problem
        .0
        .parameters()
        .iter()
        .enumerate()
        .map(|(i, s)| Param(i.try_into().unwrap(), s.clone()))
        .collect::<Vec<_>>();
    let args_ptrs = args
        .iter()
        .map(|p| p as &dyn View<Tgt = Tgt>)
        .collect::<Vec<_>>();
    root.traverse(
        &args_ptrs,
        0,
        &mut |imp, args: &[&(dyn View<Tgt = Tgt>)], depth| {
            if let Some(line_top) = imp.line_strs(&mut name_env, args) {
                let main_str = format!("{}{}", " ".repeat(depth), line_top);
                let mut r = row![main_str, "", ""];
                if let Some((problem, cost)) = imp.aux() {
                    r = row![main_str, format!("{}", &problem.0),];
                    for level_peak in cost.peaks.iter() {
                        r.add_cell(Cell::new(&format!("{: >4}", level_peak)));
                    }
                    r.add_cell(Cell::new(&format!("{:?}", cost.main)));
                }
                table.add_row(r);
            }
        },
    );

    // Format the table.
    let format = format::FormatBuilder::new()
        .separator(
            format::LinePosition::Title,
            format::LineSeparator::new('-', ' ', ' ', ' '),
        )
        .column_separator(' ')
        .build();
    table.set_format(format);

    // Decide whether to print in color.
    if color == ColorMode::Auto && !atty::is(atty::Stream::Stdout) {
        color = ColorMode::Never;
    }

    // Print the table without syntax highlighting.
    if color == ColorMode::Never {
        table.printstd();
        return;
    }

    // Syntax highlight and print the table.
    let syntax = SS.find_syntax_by_name("Python").unwrap();
    let mut h = HighlightLines::new(syntax, &TS.themes["base16-ocean.dark"]);
    let s = table.to_string();
    for line in LinesWithEndings::from(&s) {
        let ranges: Vec<(Style, &str)> = h.highlight_line(line, &SS).unwrap();
        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
        print!("{}", escaped);
    }
}
