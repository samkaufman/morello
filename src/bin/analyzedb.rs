use clap::Parser;
use itertools::Itertools;
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use std::path;

use morello::common::Dtype;
use morello::datadeps::SpecKey;
use morello::db::{deblockify_points, ActionCostVec, DashmapDiskDatabase, DbBlock, GetPreference};
use morello::grid::general::BiMap;
use morello::spec::Spec;
use morello::target::X86Target;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,
    #[arg(long)]
    group: bool,
    #[arg(long)]
    no_empty: bool,
    #[arg(long)]
    no_unsat: bool,
    #[arg(long)]
    hide_pts: bool,
}

fn block_stats(block: &DbBlock) -> String {
    match block {
        DbBlock::ActionOnly(b) => format!("runs_actiononly={}", b.0.runs_len()),
        DbBlock::Structured(b) => format!("runs_structured={}", b.0.runs_len()),
        DbBlock::Rle(rle_block) => {
            format!(
                "filled_runs={}, main_costs_runs={} peaks_runs={} depthsactions_runs={} peaks=[{}]",
                rle_block.filled.runs_len(),
                rle_block.main_costs.runs_len(),
                rle_block.peaks.runs_len(),
                rle_block.depths_actions.runs_len(),
                rle_block
                    .peaks
                    .data
                    .runs()
                    .map(|r| format!("{r:?}"))
                    .join(", "),
            )
        }
    }
}

fn main() {
    let args = Args::parse();
    let db = DashmapDiskDatabase::new(args.db.as_deref(), true, 1);
    let bimap = db.spec_bimap();
    let p = if args.group { "      " } else { "" };

    // First, scan the blocks to sort them.
    let keys_sorted = db.blocks.iter().map(|b| b.key().clone()).sorted_by_key(
        |((spec_key, factored_pt), block_key)| {
            let kind_int: u8 = match spec_key {
                SpecKey::Zero {
                    dtype: Dtype::Uint8,
                } => 0,
                SpecKey::Zero {
                    dtype: Dtype::Uint32,
                } => 1,
                SpecKey::Move {
                    dtype: Dtype::Uint8,
                } => 2,
                SpecKey::Move {
                    dtype: Dtype::Uint32,
                } => 3,
                SpecKey::Matmul {
                    dtype: Dtype::Uint8,
                } => 4,
                SpecKey::Matmul {
                    dtype: Dtype::Uint32,
                } => 5,
                SpecKey::Conv {
                    dtype: Dtype::Uint8,
                } => 6,
                SpecKey::Conv {
                    dtype: Dtype::Uint32,
                } => 7,
            };
            (kind_int, factored_pt.clone(), block_key.clone())
        },
    );

    for db_key in keys_sorted {
        let entry = db.blocks.get(&db_key).unwrap();
        let (table_key, block_key) = entry.key();
        let block = entry.value();
        let table_key_description = format!("{:?}", table_key);
        let block_key_description = format!("{:?}", block_key);
        if args.group {
            println!("{}", table_key_description);
            println!("  {} ({})", block_key_description, block_stats(&block));
        }
        let mut last_value = None;
        for inner_pt in block
            .shape()
            .iter()
            .map(|&d| 0..u8::try_from(d).unwrap())
            .multi_cartesian_product()
        {
            let value_description =
                match block.get_with_preference::<X86Target>(&db, todo!(), &inner_pt) {
                    GetPreference::Hit(ActionCostVec(v)) => {
                        if args.no_unsat && v.is_empty() {
                            continue;
                        }
                        format!("{:?}", v)
                    }
                    GetPreference::Miss(_) => {
                        if args.no_empty {
                            continue;
                        }
                        "empty".to_string()
                    }
                };

            let (table_key_part, value_part) = if args.group {
                (String::from(""), String::from(""))
            } else {
                (
                    format!("{table_key_description}\t"),
                    format!("{value_description}\t"),
                )
            };

            if args.group && Some(&value_description) != last_value.as_ref() {
                println!("    {}", value_description);
                last_value = Some(value_description);
            }

            if !args.hide_pts {
                let global_pt = deblockify_points(block_key, &inner_pt);
                let spec: Spec<X86Target> =
                    BiMap::apply_inverse(&bimap, &(table_key.clone(), global_pt));
                println!(
                    "{p}{spec}\t{value_part}{table_key_part}{block_key_description}\t{inner_pt:?}",
                )
            }
        }
    }
}
