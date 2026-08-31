mod timing_log;

use clap::Parser;
use morello::db::{FilesDatabase, TileScale};
use std::num::NonZeroUsize;
use std::{fs, path};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const K: u8 = 1;
const SUMMARY_CSV_FILE: &str = "summary.csv";

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "false",
        help = "Continue after an error reading a page."
    )]
    keep_going: bool,
    #[arg(long, default_value = "128", help = "Cache size in database pages.")]
    cache_size: usize,
    #[arg(long, default_value = "1", help = "Read only one in this many pages.")]
    sample: NonZeroUsize,
    db: path::PathBuf,
    out: path::PathBuf,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    // Read the timing log first, so a problems are reported before the expensive walk below.
    let logged_duration = timing_log::read_total_logged_duration(&args.db)
        .unwrap_or_else(|err| panic!("Failed to read timing log: {err}"));

    let db = FilesDatabase::open(Some(&args.db), TileScale::PowerOfTwo, K, args.cache_size, 1)
        .expect("Failed to open database");
    fs::create_dir_all(&args.out).unwrap();
    let analysis = db.analyze(&args.out, args.sample.get(), args.keep_going);
    // Scale up to account for the whole database if sampling.
    let spec_count = analysis.sampled_spec_count * u128::try_from(args.sample.get()).unwrap();

    let secs = logged_duration.map(|d| d.as_secs_f64());
    let rate = secs.filter(|s| *s > 0.0).map(|s| spec_count as f64 / s);
    write_summary(&args.out, spec_count, args.sample, secs, rate)
        .unwrap_or_else(|err| panic!("Failed to write {SUMMARY_CSV_FILE}: {err}"));

    println!("Specs computed: {spec_count}");
    if args.sample.get() > 1 {
        println!("  (estimated from a 1/{} page sample)", args.sample);
    }
    match secs {
        Some(secs) => println!("Total precompute time: {secs:.3}s"),
        None => println!(
            "No {} found in database directory; cannot compute Specs per second.",
            timing_log::TIMING_CSV_FILE
        ),
    }
    if let Some(rate) = rate {
        println!("Specs per second: {rate:.3}");
    }
}

/// Write a one-row summary of this run beside the per-page CSVs `analyze` writes.
///
/// The `precompute_secs` and `specs_per_second` columns are empty if the database has no timing
/// log.
fn write_summary(
    out_dir: &path::Path,
    spec_count: u128,
    sample: NonZeroUsize,
    secs: Option<f64>,
    rate: Option<f64>,
) -> csv::Result<()> {
    let mut writer = csv::Writer::from_path(out_dir.join(SUMMARY_CSV_FILE))?;
    writer.write_record([
        "spec_count",
        "sample",
        "precompute_secs",
        "specs_per_second",
    ])?;
    writer.write_record([
        spec_count.to_string(),
        sample.to_string(),
        secs.map(|s| s.to_string()).unwrap_or_default(),
        rate.map(|r| r.to_string()).unwrap_or_default(),
    ])?;
    writer.flush()?;
    Ok(())
}
