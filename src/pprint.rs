use std::collections::HashMap;

use crate::nameenv::NameEnv;
use crate::table::DbImpl;
use crate::target::Target;
use crate::utils::indent;
use crate::views::View;
use crate::{imp::Impl, views::Param};

use clap::ValueEnum;
use prettytable::{self, format, row, Cell};

#[derive(Copy, Clone, PartialEq, ValueEnum)]
pub enum PrintMode {
    Full,
    Compact,
}

pub fn pprint<Tgt: Target>(root: &DbImpl<Tgt>, print_mode: PrintMode) {
    let mut name_env: NameEnv<'_, dyn View<Tgt = Tgt>> = NameEnv::new();

    // Set up table
    let mut table = prettytable::Table::new();
    let mut titles = row![""];
    if print_mode == PrintMode::Full {
        titles.add_cell(Cell::new("Logical Spec"));
    }
    if print_mode != PrintMode::Compact {
        for level in Tgt::levels() {
            titles.add_cell(Cell::new(&level.to_string()));
        }
        titles.add_cell(Cell::new("Cost"));
    }
    table.set_titles(titles);

    let spec = &root.aux().as_ref().unwrap().0;
    let args = spec
        .0
        .parameters()
        .iter()
        .enumerate()
        .map(|(i, s)| Param::new(i.try_into().unwrap(), s.clone()))
        .collect::<Vec<_>>();
    let args_ptrs = args
        .iter()
        .map(|p| p as &dyn View<Tgt = Tgt>)
        .collect::<Vec<_>>();

    let mut param_bindings = HashMap::new();
    root.bind(&args_ptrs, &mut param_bindings);
    pprint_inner(
        &mut table,
        root,
        &param_bindings,
        &mut name_env,
        0,
        print_mode,
    );

    // Format and print the table.
    let format = format::FormatBuilder::new()
        .separator(
            format::LinePosition::Title,
            format::LineSeparator::new('-', ' ', ' ', ' '),
        )
        .column_separator(' ')
        .build();
    table.set_format(format);
    table.printstd();
}

fn pprint_inner<'a, Tgt: Target>(
    table: &mut prettytable::Table,
    imp: &'a DbImpl<Tgt>,
    param_bindings: &'a HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    name_env: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
    depth: usize,
    print_mode: PrintMode,
) {
    if let Some(line_top) = imp.line_strs(name_env, param_bindings) {
        let indent_str = indent(depth);
        let main_str = format!("{indent_str}{line_top}");
        let mut r = row![main_str, "", ""];
        if let Some((spec, cost)) = imp.aux() {
            match print_mode {
                PrintMode::Full => {
                    r = row![main_str, format!("{}", &spec.0)];
                    for level_peak in cost.peaks.iter() {
                        r.add_cell(Cell::new(&format!("{: >4}", level_peak)));
                    }
                    r.add_cell(Cell::new(&format!("{:?}", cost.main)));
                }
                PrintMode::Compact => {
                    r = row![format!("{indent_str}/* {} */\n{main_str}\n", &spec.0)];
                }
            }
        }
        table.add_row(r);
    }

    for child in imp.children() {
        pprint_inner(
            table,
            child,
            param_bindings,
            name_env,
            depth + 1,
            print_mode,
        );
    }
}
