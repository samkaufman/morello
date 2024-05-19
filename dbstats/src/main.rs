#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use anyhow::Result;
use clap::Parser;
use morello::db::FilesDatabase;
use std::path;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const K: u8 = 1;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    db: path::PathBuf, // TODO: Make the only arg.
    #[arg(
        short,
        long,
        default_value = "false",
        help = "Continue after an error reading a superblock."
    )]
    keep_going: bool,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let db = FilesDatabase::new(Some(&args.db), true, K);
    db.analyze(args.keep_going);
}
