use crate::table::DatabaseIOStore;
use crate::{common::Problem, table::Database, target::Target};
use prettytable::{self, format, row, Cell};

pub fn pprint<Tgt: Target, S: DatabaseIOStore<Tgt>>(db: &Database<Tgt, S>, root: &Problem<Tgt>) {
    let mut table = prettytable::Table::new();
    let mut titles = row!["", "Spec", "Cost"];
    for level in Tgt::levels() {
        titles.add_cell(Cell::new(&level.to_string()));
    }
    table.set_titles(titles);
    pformat_visit(&mut table, db, root, 0);

    let format = format::FormatBuilder::new()
        .separator(
            format::LinePosition::Title,
            format::LineSeparator::new('-', ' ', ' ', ' '),
        )
        .column_separator(' ')
        .build();
    table.set_format(format);

    table.printstd()
}

fn pformat_visit<Tgt: Target, S: DatabaseIOStore<Tgt>>(
    table: &mut prettytable::Table,
    db: &Database<Tgt, S>,
    root: &Problem<Tgt>,
    depth: usize,
) {
    let node = db.get(root).unwrap();
    assert!(!node.is_empty(), "Problem not in database: {:?}", root);
    assert_eq!(node.len(), 1);
    let mut r = row![
        format!("{}{}", " ".repeat(depth), node[0].0),
        format!("{}", root.0),
        format!("{:?}", node[0].1.main)
    ];
    for level_peak in node[0].1.peaks.iter() {
        r.add_cell(Cell::new(&format!("{: >4}", level_peak)));
    }
    table.add_row(r);

    let child_memory_limits = root.1.transition(&root.0, &node[0].0).unwrap();
    for (subspec, mlims) in node[0]
        .0
        .child_specs(&root.0)
        .iter()
        .zip(child_memory_limits)
    {
        pformat_visit(table, db, &Problem(subspec.clone(), mlims), depth + 1)
    }
}
