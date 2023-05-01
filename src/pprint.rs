use crate::{common::Problem, table::Database, target::Target};
use prettytable::{
    self,
    format,
    row
};

pub fn pprint<Tgt: Target>(db: &Database<Tgt>, root: &Problem<Tgt>) {
    let mut table = prettytable::Table::new();
    table.set_titles(row!["Impl", "Spec", "Cost"]);
    pformat_visit(&mut table, db, root, 0);

    let format = format::FormatBuilder::new()
        .column_separator('|')
        .borders('|')
        .separators(
            &[format::LinePosition::Top, format::LinePosition::Bottom],
            format::LineSeparator::new('-', '+', '+', '+'),
        )
        .padding(1, 1)
        .build();
    table.set_format(format);

    table.printstd()
}

fn pformat_visit<Tgt: Target>(
    table: &mut prettytable::Table,
    db: &Database<Tgt>,
    root: &Problem<Tgt>,
    depth: usize,
) {
    let node = db.get(root).unwrap();
    assert!(!node.is_empty(), "Problem not in database: {:?}", root);
    assert_eq!(node.len(), 1);
    table.add_row(row![
        format!("{}{}", " ".repeat(depth), node[0].0),
        format!("{}", root.0),
        format!("{:?}", node[0].1)
    ]);

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
