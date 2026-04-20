use clap::Parser;
use csv::Writer;
use morello::db::read_rectangles;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser)]
struct Args {
    #[arg(help = "Path to the database directory")]
    db: PathBuf,
    #[arg(help = "Path to the output directory for CSV files")]
    out: PathBuf,
    #[arg(
        long,
        default_value = "false",
        help = "Continue after errors reading pages"
    )]
    keep_going: bool,
}

#[derive(Default)]
struct CombinedStats(HashMap<String, TableGroupStats>);

/// Multiple combined [TableStats].
#[derive(Default)]
struct TableGroupStats(Vec<DiagonalAggregatedStats>);

#[derive(Default)]
struct TableStats(Vec<DiagonalStats>);

#[derive(Clone)]
struct DiagonalAggregatedStats {
    compression_ratio_min: f32,
    compression_ratio_max: f32,
    compression_ratio_sum: f64,
    compression_ratio_count: usize,
    compression_ratios: Vec<f32>, // sample ratios, sorted lazily for median
    sorted: bool,
}

#[derive(Default, Clone)]
struct DiagonalStats {
    rectangle_count: usize,
    filled_volume: usize,
}

impl CombinedStats {
    fn merge(&mut self, other: CombinedStats) {
        for (k, v) in other.0 {
            let table_group_stats = self.0.entry(k).or_default();
            table_group_stats.merge_group(v);
        }
    }
}

impl TableGroupStats {
    fn merge_group(&mut self, other: TableGroupStats) {
        // Merge the shared prefix
        for (self_stats, other_stats) in self.0.iter_mut().zip(&other.0) {
            self_stats.merge_agg(other_stats);
        }
        // Push new DiagonalAggregatedStats for any extension
        self.0.extend(other.0.into_iter().skip(self.0.len()));
    }

    fn merge(&mut self, other: &TableStats) {
        // Merge the shared prefix
        for (self_stats, other_stats) in self.0.iter_mut().zip(&other.0) {
            self_stats.merge(other_stats);
        }
        // Push new DiagonalAggregatedStats for any extension
        self.0
            .extend(other.0.iter().skip(self.0.len()).map(Into::into));
    }
}

impl Default for DiagonalAggregatedStats {
    fn default() -> Self {
        DiagonalAggregatedStats {
            compression_ratio_min: f32::INFINITY,
            compression_ratio_max: f32::NEG_INFINITY,
            compression_ratio_sum: 0.0,
            compression_ratio_count: 0,
            compression_ratios: Vec::new(),
            sorted: true,
        }
    }
}

impl DiagonalAggregatedStats {
    fn merge_agg(&mut self, aggregated: &DiagonalAggregatedStats) {
        if aggregated.compression_ratio_count == 0 {
            return;
        }
        if self.compression_ratio_count == 0 {
            *self = aggregated.clone();
            return;
        }

        self.compression_ratio_min = self
            .compression_ratio_min
            .min(aggregated.compression_ratio_min);
        self.compression_ratio_max = self
            .compression_ratio_max
            .max(aggregated.compression_ratio_max);
        self.compression_ratio_sum += aggregated.compression_ratio_sum;
        self.compression_ratio_count += aggregated.compression_ratio_count;
        self.compression_ratios
            .extend_from_slice(&aggregated.compression_ratios);
        self.sorted = false;
    }

    fn merge(&mut self, unaggregated: &DiagonalStats) {
        if let Some(ratio) = unaggregated.compression_ratio() {
            self.compression_ratio_min = self.compression_ratio_min.min(ratio);
            self.compression_ratio_max = self.compression_ratio_max.max(ratio);
            self.compression_ratio_sum += f64::from(ratio);
            self.compression_ratio_count += 1;
            self.compression_ratios.push(ratio);
            self.sorted = false;
        }
    }

    fn compression_ratio_mean(&self) -> Option<f32> {
        if self.compression_ratio_count == 0 {
            return None;
        }
        Some((self.compression_ratio_sum / self.compression_ratio_count as f64) as f32)
    }

    fn compression_ratio_median(&mut self) -> Option<f32> {
        if self.compression_ratio_count == 0 {
            return None;
        }
        self.ensure_sorted();
        let mid = self.compression_ratio_count / 2;
        let median = if self.compression_ratio_count % 2 == 1 {
            self.compression_ratios[mid]
        } else {
            (self.compression_ratios[mid - 1] + self.compression_ratios[mid]) / 2.0
        };
        Some(median)
    }

    fn ensure_sorted(&mut self) {
        if self.sorted {
            return;
        }
        self.compression_ratios
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        self.sorted = true;
    }
}

impl<'a> From<&'a DiagonalStats> for DiagonalAggregatedStats {
    fn from(value: &'a DiagonalStats) -> Self {
        let mut stats = DiagonalAggregatedStats::default();
        stats.merge(value);
        stats
    }
}

impl DiagonalStats {
    fn compression_ratio(&self) -> Option<f32> {
        if self.rectangle_count == 0 {
            return None;
        }
        Some(self.filled_volume as f32 / self.rectangle_count as f32)
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    fs::create_dir_all(&args.out).expect("Failed to create output directory");

    let leaf_dirs = collect_leaf_dirs(&args.db);
    log::info!("Found {} leaf directories", leaf_dirs.len());

    let final_result: CombinedStats = leaf_dirs
        .into_par_iter()
        .fold(
            CombinedStats::default,
            |mut combined_stats, (relative, absolute)| {
                let aggregate_key = path_root_dir(&relative).to_owned();
                let processed_table = process_table(&absolute);
                combined_stats
                    .0
                    .entry(aggregate_key)
                    .or_default()
                    .merge(&processed_table);
                combined_stats
            },
        )
        .reduce_with(|mut a, b| {
            a.merge(b);
            a
        })
        .unwrap();

    // Write CSV files for each table group
    for (table_group_name, mut table_group_stats) in final_result.0 {
        let csv_path = args.out.join(format!("{}.csv", table_group_name));
        let mut wtr = Writer::from_path(&csv_path)
            .unwrap_or_else(|e| panic!("Failed to create CSV file {:?}: {}", csv_path, e));

        // Write header
        wtr.write_record([
            "diagonal",
            "sample_count",
            "min_compression_ratio",
            "max_compression_ratio",
            "mean_compression_ratio",
            "median_compression_ratio",
        ])
        .expect("Failed to write CSV header");

        // Write data rows
        for (diagonal_idx, diag_stats) in table_group_stats.0.iter_mut().enumerate() {
            if diag_stats.compression_ratio_count == 0 {
                continue; // Skip diagonals with no samples
            }

            let cr_min = diag_stats.compression_ratio_min;
            let cr_max = diag_stats.compression_ratio_max;
            let cr_mean = diag_stats.compression_ratio_mean().unwrap_or(f32::NAN);
            let cr_median = diag_stats.compression_ratio_median().unwrap_or(f32::NAN);

            wtr.write_record([
                diagonal_idx.to_string(),
                diag_stats.compression_ratio_count.to_string(),
                format!("{:.6}", cr_min),
                format!("{:.6}", cr_max),
                format!("{:.6}", cr_mean),
                format!("{:.6}", cr_median),
            ])
            .expect("Failed to write CSV row");
        }

        wtr.flush().expect("Failed to flush CSV writer");
        log::info!("Wrote CSV file: {:?}", csv_path);
    }
}

fn path_root_dir(path: &Path) -> &str {
    for component in path.components() {
        match component {
            std::path::Component::RootDir => {}
            _ => {
                return component.as_os_str().to_str().unwrap();
            }
        }
    }
    panic!("no path components");
}

fn process_table(leaf_dir_path: &Path) -> TableStats {
    debug_assert!(is_leaf_dir(leaf_dir_path));

    let maxes = compute_maxes_for_table(leaf_dir_path);
    let steps = usize::try_from(maxes.iter().copied().max().unwrap()).unwrap() + 1;

    let mut diagonals = vec![DiagonalStats::default(); steps];
    for entry in fs::read_dir(leaf_dir_path).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let ft = entry.file_type().expect("Failed to get file type");
        assert!(ft.is_file());

        let file = File::open(entry.path()).expect("Failed to open file");
        for (bottom, top) in read_rectangles(file).expect("Failed to read page") {
            debug_assert_eq!(bottom.len(), maxes.len());
            debug_assert!(bottom.iter().all(|&b| b >= 0));
            debug_assert!(top.iter().all(|&t| t >= 0));

            let first_diagonal = bottom
                .iter()
                .map(|&b| usize::try_from(b).unwrap())
                .max()
                .unwrap();
            for (i, diag) in diagonals.iter_mut().enumerate().skip(first_diagonal) {
                diag.rectangle_count += 1;
                diag.filled_volume += {
                    let intersection_top = top
                        .iter()
                        .map(|&t| usize::try_from(t).unwrap().min(i))
                        .collect::<Vec<_>>();
                    debug_assert_eq!(bottom.len(), intersection_top.len());
                    bottom
                        .iter()
                        .zip(intersection_top)
                        .map(|(b, it)| 1 + it - usize::try_from(*b).unwrap())
                        .product::<usize>()
                };
            }
        }
    }
    TableStats(diagonals)
}

fn compute_maxes_for_table(leaf_dir_path: &Path) -> Vec<u64> {
    let mut agg: Option<Vec<u64>> = None;

    for entry in fs::read_dir(leaf_dir_path).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let ft = entry.file_type().expect("Failed to get file type");
        assert!(ft.is_file());
        let path = entry.path();
        let maxes = compute_maxes_from_file(&path);

        match &mut agg {
            None => {
                agg = Some(maxes);
            }
            Some(acc) => {
                assert_ne!(
                    acc.len(),
                    maxes.len(),
                    "Dimension count mismatch: expected {}, got {} for file {path:?}",
                    acc.len(),
                    maxes.len()
                );
                for (a, m) in acc.iter_mut().zip(maxes.into_iter()) {
                    *a = (*a).max(m);
                }
            }
        }
    }

    agg.unwrap_or_default()
}

/// Convenience helper: open a table file, read rectangles, and compute inclusive bounds.
fn compute_maxes_from_file(file_path: &Path) -> Vec<u64> {
    let file = File::open(file_path).expect("Failed to open file");
    let rectangles = read_rectangles(file).expect("Failed to read page");
    if rectangles.is_empty() {
        return Vec::new();
    }

    assert!(!rectangles.is_empty(), "empty rectangles input");
    let dim_count = rectangles[0].0.len();
    let mut maxes = vec![0u64; dim_count];
    for (_, top) in rectangles {
        for i in 0..dim_count {
            maxes[i] = maxes[i].max(top[i].try_into().unwrap());
        }
    }
    maxes
}

/// Gather all leaf directories under `root`, returning `(relative, absolute)` pairs.
fn collect_leaf_dirs(root: &Path) -> Vec<(PathBuf, PathBuf)> {
    let mut leaves = Vec::new();
    for dent in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if dent.file_type().is_dir() {
            let p = dent.path();
            if is_leaf_dir(p) {
                let absolute = p.to_path_buf();
                let relative = match p.strip_prefix(root) {
                    Ok(relative) if relative.as_os_str().is_empty() => PathBuf::from("."),
                    Ok(relative) => relative.to_path_buf(),
                    Err(_) => absolute.clone(),
                };
                leaves.push((relative, absolute));
            }
        }
    }
    leaves
}

/// A "leaf directory" = contains ≥1 file and contains no subdirectories.
fn is_leaf_dir(dir: &Path) -> bool {
    let mut has_file = false;
    for entry in fs::read_dir(dir).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let ft = entry.file_type().expect("Failed to get file type");
        if ft.is_dir() {
            return false; // has a subdir → not a leaf
        }
        if ft.is_file() {
            has_file = true;
        }
    }
    has_file
}
