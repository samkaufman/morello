//! Reader for the precompute run-time log stored alongside a database.

use serde::Deserialize;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

pub const TIMING_CSV_FILE: &str = "timing.csv";

#[derive(Debug, Deserialize)]
struct TimingRecord {
    duration_secs: f64,
}

/// Sum the durations of all runs recorded in `db_path`'s timing log, or `None` if no log exists.
pub fn read_total_logged_duration(db_path: &Path) -> csv::Result<Option<Duration>> {
    let file = match fs::File::open(db_path.join(TIMING_CSV_FILE)) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut total = Duration::ZERO;
    for record in csv::Reader::from_reader(file).deserialize::<TimingRecord>() {
        total += Duration::from_secs_f64(record?.duration_secs);
    }
    Ok(Some(total))
}
