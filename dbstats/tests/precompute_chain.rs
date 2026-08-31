//! End-to-end check of the precompute -> dbstats chain.
//!
//! This test runs both binaries and asserts dbstats derives a Specs-per-second figure from what
//! precompute wrote.

use std::process::Command;

#[test]
fn dbstats_reports_a_rate_from_precomputes_timing_log() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let out = tmp.path().join("out");

    // Run twice so dbstats has more than one row to total up.
    let cargo_cmd = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    for _ in 0..2 {
        run(Command::new(&cargo_cmd)
            .args(["run", "-q", "-p", "precompute", "--"])
            .args(["--through", "move", "--db"])
            .arg(&db)
            .arg("1"));
    }

    let log = std::fs::read_to_string(db.join("timing.csv")).unwrap();
    let rows: Vec<&str> = log.lines().collect();
    assert_eq!(
        rows.len(),
        3,
        "expected a header and one row per run, got {log:?}"
    );
    let logged_secs: f64 = rows[1..]
        .iter()
        .map(|r| r.rsplit(',').next().unwrap().parse::<f64>().unwrap())
        .sum();

    let stats = run(Command::new(env!("CARGO_BIN_EXE_dbstats"))
        .arg(&db)
        .arg(&out));
    let specs: f64 = field(&stats, "Specs computed: ").parse().unwrap();
    let rate: f64 = field(&stats, "Specs per second: ").parse().unwrap();

    assert!(
        (rate - specs / logged_secs).abs() / rate < 1e-6,
        "{rate} is not {specs} / {logged_secs}"
    );
}

/// Run `cmd` to completion, returning its stdout and panicking on failure.
fn run(cmd: &mut Command) -> String {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} exited with {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Returns the remainder of the line in `stdout` beginning with `prefix`.
fn field<'a>(stdout: &'a str, prefix: &str) -> &'a str {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(prefix))
        .unwrap_or_else(|| panic!("no {prefix:?} line in {stdout:?}"))
}
