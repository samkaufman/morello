use clap::Parser;
use morello::{db::FilesDatabase, target::X86Target};
use std::{fs, path};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const K: u8 = 1;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "false",
        help = "Continue after an error reading a superblock."
    )]
    keep_going: bool,
    #[arg(long, default_value = "128", help = "Cache size in database pages.")]
    cache_size: usize,
    #[arg(long, default_value = "1")]
    sample: usize,
    db: path::PathBuf,
    out: path::PathBuf,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let db = FilesDatabase::new(Some(&args.db), true, K, args.cache_size, 1);
    fs::create_dir_all(&args.out).unwrap();
    db.analyze::<X86Target>(&args.out, args.sample, args.keep_going);
}
