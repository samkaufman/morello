use crate::imp::ImplNode;
use crate::nameenv::NameEnv;
use crate::target::Target;
use crate::utils::indent;
use crate::views::View;
use crate::{imp::Impl, views::Param};

use prettytable::{self, format, row, Cell};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum ImplPrintStyle {
    Full,
    Compact,
}

pub trait PrintableAux: Clone {
    fn extra_column_titles(&self) -> Vec<String>;
    fn extra_column_values(&self) -> Vec<String>;
    fn c_header(&self) -> Option<String>;
}

impl PrintableAux for () {
    fn extra_column_titles(&self) -> Vec<String> {
        vec![]
    }

    fn extra_column_values(&self) -> Vec<String> {
        vec![]
    }

    fn c_header(&self) -> Option<String> {
        None
    }
}

/// Pretty-print an [ImplNode] to stdout.
pub fn pprint<Tgt, Aux>(root: &ImplNode<Tgt, Aux>, style: ImplPrintStyle)
where
    Tgt: Target,
    Aux: PrintableAux,
{
    pprint_table(root, style).printstd()
}

/// Pretty-print an [ImplNode] to a given [fmt::Write].
pub fn pprint_write<Tgt, W, Aux>(
    out: &mut W,
    root: &ImplNode<Tgt, Aux>,
    style: ImplPrintStyle,
) -> fmt::Result
where
    Tgt: Target,
    W: fmt::Write,
    Aux: PrintableAux,
{
    write!(out, "{}", pprint_table(root, style))
}

pub fn pprint_string<Tgt, Aux>(root: &ImplNode<Tgt, Aux>, style: ImplPrintStyle) -> String
where
    Tgt: Target,
    Aux: PrintableAux,
{
    // TODO: Avoid final newline
    format!("{}", pprint_table(root, style))
}

fn pprint_table<Tgt, Aux>(root: &ImplNode<Tgt, Aux>, style: ImplPrintStyle) -> prettytable::Table
where
    Tgt: Target,
    Aux: PrintableAux,
{
    let mut name_env: NameEnv<'_, dyn View<Tgt = Tgt>> = NameEnv::new();

    // Set up table
    let mut table = prettytable::Table::new();
    let mut titles = row!["Impl"];
    match style {
        ImplPrintStyle::Full => {
            for col_name in root.aux().extra_column_titles() {
                titles.add_cell(Cell::new(&col_name));
            }
        }
        ImplPrintStyle::Compact => {}
    }
    table.set_titles(titles);

    let args = root
        .parameters()
        .enumerate()
        .map(|(i, s)| Param::new(i.try_into().unwrap(), s.clone()))
        .collect::<Vec<_>>();
    let args_ptrs = args
        .iter()
        .map(|p| p as &dyn View<Tgt = Tgt>)
        .collect::<Vec<_>>();

    let mut param_bindings = HashMap::new();
    root.bind(&args_ptrs, &mut param_bindings);
    pprint_inner(&mut table, root, &param_bindings, &mut name_env, 0, style);

    // Format and print the table.
    let format = format::FormatBuilder::new()
        .separator(
            format::LinePosition::Title,
            format::LineSeparator::new('-', ' ', ' ', ' '),
        )
        .column_separator(' ')
        .build();
    table.set_format(format);
    table
}

fn pprint_inner<'a, Tgt, Aux>(
    table: &mut prettytable::Table,
    imp: &'a ImplNode<Tgt, Aux>,
    param_bindings: &'a HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    name_env: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
    depth: usize,
    style: ImplPrintStyle,
) where
    Tgt: Target,
    Aux: PrintableAux,
{
    if let Some(line_top) = imp.pprint_line(name_env, param_bindings) {
        let indent_str = indent(depth);
        let main_str = format!("{indent_str}{line_top}");
        let mut r;
        match style {
            ImplPrintStyle::Full => {
                r = row![main_str];
                for column_value in imp.aux().extra_column_values() {
                    r.add_cell(Cell::new(&column_value));
                }
            }
            ImplPrintStyle::Compact => {
                let joined = imp.aux().extra_column_values().join(", ");
                if !joined.is_empty() {
                    r = row![format!("{indent_str}/* {joined} */\n{main_str}\n")];
                } else {
                    r = row![main_str];
                }
            }
        }
        table.add_row(r);
    }

    for child in imp.children() {
        pprint_inner(table, child, param_bindings, name_env, depth + 1, style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::subspecs::SpecApp;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
    use crate::target::{
        CpuMemoryLevel::{GL, L1},
        X86Target,
    };
    use crate::tensorspec::TensorSpecAux;

    #[test]
    fn test_can_pprint_a_specapp_with_no_aux() {
        let rm1 = row_major(1);
        let logical_spec: LogicalSpec<X86Target> =
            lspec!(Move([4], (u8, GL, rm1.clone()), (u8, L1, rm1)));
        let args = logical_spec
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, ts)| Param::new(i.try_into().unwrap(), ts));
        let spec_app: ImplNode<X86Target, ()> =
            SpecApp::new(Spec(logical_spec, X86Target::max_mem()), args).into();
        pprint(&spec_app, ImplPrintStyle::Full);
        pprint(&spec_app, ImplPrintStyle::Compact);
    }
}
