#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "duckdb",
#   "pandas",
#   "joblib",
# ]
# ///
"""Plot rectangle counts from dbratios output CSV files."""

import sys
import pathlib
import pandas as pd
import re
import duckdb
from joblib import Memory

DIM_COL_PATTERN = re.compile(r"dim(\d+)_top")

cache_dir = pathlib.Path(__file__).parent / ".cache" / "morello" / "plot_ratios"
memory = Memory(cache_dir, verbose=1)


def cache_is_valid(metadata):
    current_mtime = 0.0
    for f in metadata["input_args"]["directory"].glob("**/*.csv.gz"):
        current_mtime = max(current_mtime, f.stat().st_mtime)
    # Cache is valid only if no files are newer than when the cache was created
    return current_mtime <= metadata["time"]


# TODO: We should normalize relative paths for memoizing somehow.
@memory.cache(cache_validation_callback=cache_is_valid)
def read_all_csvs(directory: pathlib.Path) -> pd.DataFrame:
    """Read all gzipped CSVs and compute compression ratio statistics across files."""

    con = duckdb.connect()

    # Create a temporary view (not materialized) over all input CSVs
    glob_pattern = str(directory / "**" / "*.csv.gz")
    con.execute(f"""
        CREATE TEMP VIEW csv_data AS
        SELECT * FROM read_csv(
            '{glob_pattern}',
            compression='gzip',
            union_by_name=true,
            filename=true
        )
    """)

    # Get the schema to identify dimension columns
    schema_df = con.execute("SELECT * FROM csv_data LIMIT 0").df()
    dim_cols = [col for col in schema_df.columns if DIM_COL_PATTERN.match(col)]

    query = f"""
    SELECT
        regexp_replace(filename, '^[^/]+/([^/]+)/.*', '\\1') as top_level_dir,
        max_dim,
        COUNT(*) as count,
        AVG(rectangles) as mean_rectangles,
        AVG(covered) as mean_covered,
        AVG(compression_ratio) as mean_compression_ratio,
        MIN(compression_ratio) as min_compression_ratio,
        MAX(compression_ratio) as max_compression_ratio
    FROM (
        SELECT 
            t1.*,
            GREATEST({", ".join([f"COALESCE({col}, 0)" for col in dim_cols])}) as max_dim,
            CAST(t1.covered AS DOUBLE) / NULLIF(t1.rectangles, 0) as compression_ratio,
            {", ".join([f"t2.max_{col}" for col in dim_cols])}
        FROM csv_data t1
        JOIN (
            SELECT filename,
                   {", ".join([f"MAX({col}) as max_{col}" for col in dim_cols])}
            FROM csv_data
            GROUP BY filename
        ) t2 ON t1.filename = t2.filename
        WHERE {" AND ".join([f"({col} = max_{col} OR {col} = max_dim)" for col in dim_cols])}
    ) filtered
    GROUP BY regexp_replace(filename, '^[^/]+/([^/]+)/.*', '\\1'), max_dim
    ORDER BY top_level_dir, max_dim
    """

    # Write query plan to file for inspection
    con.execute("PRAGMA explain_output = 'all'")
    explain_df = con.execute(f"EXPLAIN {query}").df()
    explain_path = pathlib.Path(__file__).parent / "query_plan.txt"
    with open(explain_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("QUERY PLAN:\n")
        f.write("=" * 80 + "\n\n")

        # Write each column's plan with proper newline handling
        for col in explain_df.columns:
            f.write(f"{col.upper()}:\n")
            f.write("-" * 80 + "\n")
            plan_text = explain_df[col].iloc[0]
            if plan_text:
                # Replace literal \n with actual newlines and \t with tabs
                formatted = plan_text.replace("\\n", "\n").replace("\\t", "\t")
                f.write(formatted + "\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
    print(f"Query plan written to: {explain_path}")

    return con.execute(query).df()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    directory = pathlib.Path(sys.argv[1])
    df = read_all_csvs(directory)
    print(df)
