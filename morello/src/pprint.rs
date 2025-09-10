use crate::cost::Cost;
use crate::imp::{Impl, ImplNode};
use crate::nameenv::NameEnv;
use crate::target::Target;
use crate::utils::indent;

use by_address::ByThinAddress;
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

/// Pretty-print an [ImplNode] to stdout.
pub fn pprint<Tgt>(root: &ImplNode<Tgt>, style: ImplPrintStyle)
where
    Tgt: Target,
{
    pprint_table(root, style).printstd()
}

/// Pretty-print an [ImplNode] to a given [fmt::Write].
pub fn pprint_write<Tgt, W>(out: &mut W, root: &ImplNode<Tgt>, style: ImplPrintStyle) -> fmt::Result
where
    Tgt: Target,
    W: fmt::Write,
{
    write!(out, "{}", pprint_table(root, style))
}

pub fn pprint_string<Tgt>(root: &ImplNode<Tgt>, style: ImplPrintStyle) -> String
where
    Tgt: Target,
{
    // TODO: Avoid final newline
    format!("{}", pprint_table(root, style))
}

fn pprint_table<Tgt>(root: &ImplNode<Tgt>, style: ImplPrintStyle) -> prettytable::Table
where
    Tgt: Target,
{
    let mut name_env = NameEnv::new();
    let mut costs_table = HashMap::new();

    // Set up table
    let mut table = prettytable::Table::new();
    let titles = match style {
        ImplPrintStyle::Full => row!["Impl", "Logical Spec", "Cost", "Peaks", "Depth", "Path"],
        ImplPrintStyle::Compact => row!["Impl"],
    };
    table.set_titles(titles);

    let beta_subbed = root.clone().bind(&mut |_| None);
    pprint_inner(
        &mut table,
        &beta_subbed,
        &mut name_env,
        &mut costs_table,
        0,
        style,
        &[],
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
    table
}

fn pprint_inner<'a, Tgt>(
    table: &mut prettytable::Table,
    imp: &'a ImplNode<Tgt>,
    name_env: &mut NameEnv,
    costs_table: &mut HashMap<ByThinAddress<&'a ImplNode<Tgt>>, Cost>,
    depth: usize,
    style: ImplPrintStyle,
    path: &[usize],
) where
    Tgt: Target,
{
    if let Some(line_top) = imp.pprint_line(name_env) {
        let indent_str = indent(depth);
        let main_str = format!("{indent_str}{line_top}");
        let mut r;

        let cost = fill_costs_table_entry(costs_table, imp);
        let path_str = format!(
            "[{}]",
            path.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut extra_column_values = vec![
            "".to_owned(),
            cost.main.to_string(),
            cost.peaks.to_string(),
            cost.depth.to_string(),
            path_str,
        ];
        if let Some(spec) = imp.spec() {
            extra_column_values[0] = spec.to_string();
        }

        match style {
            ImplPrintStyle::Full => {
                r = row![main_str];
                for v in extra_column_values {
                    r.add_cell(Cell::new(&v));
                }
            }
            ImplPrintStyle::Compact => {
                let path_info = extra_column_values.pop().unwrap();
                extra_column_values.retain(|s| !s.is_empty());
                extra_column_values.push(format!("path: {path_info}"));

                if !extra_column_values.is_empty() {
                    let joined = extra_column_values.join(", ");
                    r = row![format!("{indent_str}/* {joined} */\n{main_str}\n")];
                } else {
                    r = row![main_str];
                }
            }
        }
        table.add_row(r);
    }

    let children = imp.children();
    for (i, child) in children.iter().enumerate() {
        let mut child_path = path.to_vec();
        if children.len() > 1 {
            child_path.push(i);
        }
        pprint_inner(
            table,
            child,
            name_env,
            costs_table,
            depth + 1,
            style,
            &child_path,
        );
    }
}

fn fill_costs_table_entry<'a, Tgt: Target>(
    table: &mut HashMap<ByThinAddress<&'a ImplNode<Tgt>>, Cost>,
    imp: &'a ImplNode<Tgt>,
) -> Cost {
    let imp_address = ByThinAddress(imp);
    if let Some(c) = table.get(&imp_address) {
        c.clone()
    } else {
        let child_costs = imp
            .children()
            .iter()
            .map(|k| fill_costs_table_entry(table, k))
            .collect::<Vec<_>>();
        let c = Cost::from_node_and_child_costs(imp, &child_costs);
        drop(child_costs);
        table.insert(imp_address, c.clone());
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::subspecs::SpecApp;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::spec::{LogicalSpec, Spec};
    use crate::target::{
        Avx2Target,
        CpuMemoryLevel::{GL, L1},
    };

    #[test]
    fn test_can_pprint_a_specapp_with_no_aux() {
        let logical_spec: LogicalSpec<Avx2Target> =
            lspec!(Move([4], (u8, GL, row_major), (u8, L1, row_major)));
        let spec_app: ImplNode<Avx2Target> =
            SpecApp::new_with_default_params(Spec(logical_spec, Avx2Target::max_mem())).into();
        pprint(&spec_app, ImplPrintStyle::Full);
        pprint(&spec_app, ImplPrintStyle::Compact);
    }
}
