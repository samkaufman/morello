//! Log precompute run times alongside the database.
//!
//! Each row records one completed precompute run: wall-clock start and end times as milliseconds
//! since the Unix epoch, plus a duration measured separately with a monotonic timer.
//!
//! The window covers synthesis and the final flush to disk, but not opening the database, so the
//! times stay comparable across runs that resume a large existing database and runs that start from
//! nothing.

use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TIMING_CSV_FILE: &str = "timing.csv";

#[derive(Debug, Serialize)]
struct TimingRecord {
    start_unix_ms: u64,
    end_unix_ms: u64,
    duration_secs: f64,
}

/// Append one record to the timing log in `db_path`, creating the file (with a header) if it
/// doesn't exist.
pub fn append_run(
    db_path: &Path,
    start: SystemTime,
    end: SystemTime,
    duration: Duration,
) -> csv::Result<()> {
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(db_path.join(TIMING_CSV_FILE))?;
    let needs_header = file.metadata()?.len() == 0;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(needs_header)
        .from_writer(file);
    writer.serialize(TimingRecord {
        start_unix_ms: unix_ms(start),
        end_unix_ms: unix_ms(end),
        duration_secs: duration.as_secs_f64(),
    })?;
    writer.flush()?;
    Ok(())
}

fn unix_ms(t: SystemTime) -> u64 {
    u64::try_from(
        t.duration_since(UNIX_EPOCH)
            .expect("timestamp should be after Unix epoch")
            .as_millis(),
    )
    .expect("timestamp should fit in a u64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_writes_a_header_and_one_row_per_run() {
        let dir = tempfile::tempdir().unwrap();
        let start = UNIX_EPOCH + Duration::from_millis(1_700_000_000_000);
        let end = start + Duration::from_millis(1_500);
        append_run(dir.path(), start, end, Duration::from_secs_f64(1.5)).unwrap();
        append_run(dir.path(), start, end, Duration::from_secs_f64(2.25)).unwrap();

        let contents = fs::read_to_string(dir.path().join(TIMING_CSV_FILE)).unwrap();
        assert_eq!(
            contents,
            "start_unix_ms,end_unix_ms,duration_secs\n\
             1700000000000,1700000001500,1.5\n\
             1700000000000,1700000001500,2.25\n"
        );
    }
}
