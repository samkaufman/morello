use crate::color::do_color;
use crate::highlight;
use crate::nameenv::NameEnv;
use crate::table::DbImpl;
use crate::target::Target;
use crate::utils::indent;
use crate::views::View;
use crate::{imp::Impl, views::Param};

use clap::ValueEnum;
use prettytable::{self, format, row, Cell};

#[derive(Clone, PartialEq, ValueEnum)]
pub enum PrintMode {
    Full,
    Comfort,
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

    // Traverse the Impl.
    let problem = &root.aux().as_ref().unwrap().0;
    let args = problem
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
    root.traverse(
        &args_ptrs,
        0,
        &mut |imp, args: &[&(dyn View<Tgt = Tgt>)], depth| {
            if let Some(line_top) = imp.line_strs(&mut name_env, args) {
                let indent = indent(depth);
                let morello_ir = format!("{indent}{line_top}");
                let mut r = row![morello_ir, "", ""];
                if let Some((problem, cost)) = imp.aux() {
                    if print_mode == PrintMode::Full {
                        r = row![morello_ir, format!("{}", &problem.0)];
                    } else {
                        r = row![format!("{indent}/* {} */\n{morello_ir}\n", &problem.0)];
                    }

                    if print_mode != PrintMode::Compact {
                        for level_peak in cost.peaks.iter() {
                            r.add_cell(Cell::new(&format!("{: >4}", level_peak)));
                        }
                        r.add_cell(Cell::new(&format!("{:?}", cost.main)));
                    }
                }
                table.add_row(r);
            }
        },
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
    if do_color() {
        highlight::morello(&table.to_string());
    } else {
        table.printstd();
    }
}
