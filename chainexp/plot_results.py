#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "altair",
#   "pandas",
#   "vl-convert-python",
# ]
# ///
import argparse
import sys
from pathlib import Path
from typing import Callable

import altair as alt
import numpy as np
import pandas as pd

METRIC_LABELS = {
    "kernel_runtime": "Kernel Runtime (s)",
    "throughput": "Throughput (iter/s)",
    "gflops_per_sec": "GFLOPs/second",
}
REQUIRED_COLS = {
    "total_size",
    "total_k_outer",
    "total_k_inner",
    "split_size",
    "tile_size",
    "buffer_level",
    "cost",
    "kernel_runtime",
}
GROUP_COLS = ["total_size", "split_size", "tile_size", "buffer_level", "cost"]
ERROR_BAR_LOG_RATIO_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated measurements emitted by `chainexp` and "
            "render an Altair plot with error bars for the kernel runtime metric."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("chainexp_plots"),
        help="Destination directory for output SVG files (default: chainexp_plots).",
    )
    parser.add_argument(
        "--title",
        help="Optional title for the chart (defaults to one derived from the metric).",
    )
    return parser.parse_args()


def build_metric_chart(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    required_cols = set(GROUP_COLS + [metric])
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for metric chart: {sorted(missing)}")

    metric_label = METRIC_LABELS[metric]
    y_axis = alt.Axis(format=".1e")

    aggregated = (
        alt.Chart(df)
        .transform_aggregate(
            median_value=f"median({metric})",
            min_value=f"min({metric})",
            max_value=f"max({metric})",
            groupby=GROUP_COLS,
        )
        .transform_calculate(
            median_value="datum.median_value > 0 ? datum.median_value : 1e-12",
            min_value="datum.min_value > 0 ? datum.min_value : 1e-12",
            max_value="datum.max_value > 0 ? datum.max_value : 1e-12",
        )
        .transform_calculate(
            lower="datum.min_value < datum.median_value ? datum.min_value : datum.median_value",
            upper="datum.max_value > datum.median_value ? datum.max_value : datum.median_value",
            lower_offset="datum.lower - datum.median_value",
            upper_offset="datum.upper - datum.median_value",
            log_ratio="datum.min_value > 0 && datum.max_value > 0 ? abs(log(datum.max_value / datum.min_value)) : 0",
        )
        .encode(
            x=alt.X(
                "cost:Q",
                title="Cost",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format=".1e"),
            ),
            y=alt.Y(
                "median_value:Q",
                title=f"Median {metric_label}",
                scale=alt.Scale(type="log"),
                axis=y_axis,
            ),
            color=alt.Color("total_size:O", title="Total Size"),
            shape=alt.Shape("tile_size:O", title="Tile Size"),
            detail=[
                "total_size:O",
                "split_size:O",
                "tile_size:O",
                "buffer_level:N",
            ],
        )
    )

    points = aggregated.mark_point(size=80, filled=True)

    errorbars = (
        aggregated.transform_filter(
            f"datum.log_ratio >= {ERROR_BAR_LOG_RATIO_THRESHOLD}"
        )
        .mark_errorbar(rule={"size": 2}, ticks={"size": 12, "thickness": 2})
        .encode(
            yError="upper_offset:Q",
            yError2="lower_offset:Q",
        )
    )

    return (errorbars + points).properties(title=title)


def count_errorbar_points(df: pd.DataFrame, metric: str) -> tuple[int, int]:
    required_cols = set(GROUP_COLS + [metric])
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for error bar count: {sorted(missing)}")

    grouped = (
        df.groupby(GROUP_COLS, dropna=False)[metric]
        .agg(minimum="min", maximum="max")
        .reset_index()
    )

    positives = (grouped["minimum"] > 0) & (grouped["maximum"] > 0)
    ratios = np.ones(len(grouped))
    ratios[positives] = (
        grouped.loc[positives, "maximum"] / grouped.loc[positives, "minimum"]
    )
    log_ratios = np.zeros(len(grouped))
    valid = positives & (ratios != 1)
    log_ratios[valid] = np.abs(np.log(ratios[valid]))

    count = int((log_ratios >= ERROR_BAR_LOG_RATIO_THRESHOLD).sum())
    return count, len(grouped)


def summarize_metric_errors(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required_cols = set(GROUP_COLS + [metric])
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for metric summary: {sorted(missing)}")

    grouped = (
        df.groupby(GROUP_COLS, dropna=False)[metric]
        .agg(median="median", minimum="min", maximum="max")
        .reset_index()
    )

    lower_dev = grouped["median"] - grouped["minimum"]
    upper_dev = grouped["maximum"] - grouped["median"]
    grouped["max_deviation"] = np.maximum(lower_dev, upper_dev)

    median_nonzero = grouped["median"].where(grouped["median"] != 0, np.nan)
    grouped["relative_error"] = grouped["max_deviation"] / median_nonzero
    grouped["relative_error_pct"] = grouped["relative_error"] * 100

    grouped.sort_values(
        by=["relative_error", "max_deviation"], ascending=[False, False], inplace=True
    )
    return grouped


def print_metric_errors(summary_df: pd.DataFrame, metric: str) -> None:
    if summary_df.empty:
        print(
            f"No observations available to summarize {metric} deviations.",
            file=sys.stderr,
        )
        return

    display_cols = GROUP_COLS + [
        "median",
        "minimum",
        "maximum",
        "max_deviation",
        "relative_error_pct",
    ]

    formatters: dict[str, Callable[[float], str]] = {
        "median": lambda x: f"{x:.3e}",
        "minimum": lambda x: f"{x:.3e}",
        "maximum": lambda x: f"{x:.3e}",
        "max_deviation": lambda x: f"{x:.3e}",
        "relative_error_pct": lambda x: "nan" if pd.isna(x) else f"{x:.2f}%",
    }

    total_rows = len(summary_df)
    display_df = summary_df[display_cols].head(20)

    print(f"Maximum observed {metric} deviations by configuration:")
    print(display_df.to_string(index=False, formatters=formatters))
    if total_rows > len(display_df):
        print(
            f"... ({total_rows - len(display_df)} additional configurations not shown)"
        )


def build_bump_chart(df: pd.DataFrame) -> alt.LayerChart:
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for bump chart: {sorted(missing)}")

    # Create facet label in pandas before passing to Altair
    df = df.copy()
    df["facet_label"] = (
        df["total_size"].astype(str)
        + ","
        + df["total_k_outer"].astype(str)
        + ","
        + df["total_k_inner"].astype(str)
    )

    base = (
        alt.Chart(df)
        .transform_filter("isValid(datum.cost) && isValid(datum.kernel_runtime)")
        .transform_aggregate(
            cost_median="median(cost)",
            runtime_median="median(kernel_runtime)",
            groupby=[
                "facet_label",
                "total_size",
                "total_k_outer",
                "total_k_inner",
                "split_size",
                "tile_size",
                "buffer_level",
            ],
        )
        .transform_fold(
            ["cost_median", "runtime_median"],
            as_=["metric", "median_value"],
        )
        .transform_calculate(
            rank_type="datum.metric == 'cost_median' ? 'Cost' : 'Median Runtime'",
        )
        .transform_window(
            window=[{"op": "dense_rank", "as": "rank"}],
            sort=[alt.SortField("median_value", order="ascending")],
            groupby=["facet_label", "rank_type"],
        )
        .encode(
            x=alt.X(
                "rank:Q",
                title="Rank",
                scale=alt.Scale(reverse=True, nice=False, zero=False),
                axis=alt.Axis(format=".0f", tickMinStep=1),
            ),
            y=alt.Y(
                "rank_type:N",
                title="",
                sort=["Cost", "Median Runtime"],
            ),
            color=alt.Color("buffer_level:N", title="Buffer Level"),
            shape=alt.Shape("tile_size:O", title="Tile Size"),
            detail=["split_size:O", "tile_size:O", "buffer_level:N"],
        )
    )

    lines = base.mark_line(size=2)
    points = base.mark_point(size=80, filled=True)

    chart = (lines + points).facet(
        row=alt.Row(
            "facet_label:O",
            title="(total_size, total_k_outer, total_k_inner)",
            sort="ascending",
        )
    )
    return chart.properties(
        title="Cost vs. Median Runtime Rank", spacing=16
    ).resolve_scale(
        x="shared",
        y="shared",
    )


def build_cumulative_best_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Build a chart showing cumulative best runtime by cost rank.
    For each facet (problem size), configurations are ranked by cost.
    At each rank, we show the best runtime seen so far (cumulative minimum),
    divided by the overall best runtime in that facet, converted to iterations/sec.
    """
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for cumulative best chart: {sorted(missing)}")

    # Prepare data in pandas for clearer computation
    df_prepared = df.copy()
    df_prepared["facet_label"] = (
        df_prepared["total_size"].astype(str)
        + ","
        + df_prepared["total_k_outer"].astype(str)
        + ","
        + df_prepared["total_k_inner"].astype(str)
    )

    # Aggregate to get median cost and runtime per configuration
    grouped = (
        df_prepared.groupby(
            [
                "facet_label",
                "total_size",
                "total_k_outer",
                "total_k_inner",
                "split_size",
                "tile_size",
                "buffer_level",
            ],
            dropna=False,
        )
        .agg({"cost": "median", "kernel_runtime": "median"})
        .reset_index()
        .rename(columns={"cost": "median_cost", "kernel_runtime": "median_runtime"})
    )

    # Sort by facet and cost to establish rank order
    grouped = grouped.sort_values(["facet_label", "median_cost"]).reset_index(drop=True)

    # Compute cumulative minimum runtime and overall best for each facet
    grouped["cumulative_best_runtime"] = grouped.groupby("facet_label")[
        "median_runtime"
    ].cummin()
    grouped["overall_best_runtime"] = grouped.groupby("facet_label")[
        "median_runtime"
    ].transform("min")

    # Compute throughput metric: best_at_rank / best_overall (as iterations/sec)
    grouped["best_throughput_at_rank"] = 1.0 / grouped["cumulative_best_runtime"]
    grouped["best_overall_throughput"] = 1.0 / grouped["overall_best_runtime"]
    grouped["throughput_ratio"] = (
        grouped["best_throughput_at_rank"] / grouped["best_overall_throughput"]
    )

    # Assign cost rank within each facet
    grouped["cost_rank"] = grouped.groupby("facet_label").cumcount() + 1

    chart = (
        alt.Chart(grouped)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "cost_rank:Q",
                title="Rank (by Cost)",
                axis=alt.Axis(format=".0f", tickMinStep=1),
            ),
            y=alt.Y(
                "throughput_ratio:Q",
                title="Best Throughput at Rank / Best Overall Throughput",
                scale=alt.Scale(domain=[0, 1]),
            ),
        )
        .facet(
            row=alt.Row(
                "facet_label:O",
                title="(total_size, total_k_outer, total_k_inner)",
                sort="ascending",
            )
        )
        .properties(title="Cumulative Best Runtime by Cost Rank", spacing=16)
        .resolve_scale(x="independent", y="shared")
    )

    return chart


def build_histogram_scatter_chart(df: pd.DataFrame) -> alt.VConcatChart:
    """
    Build a 2D histogram scatter plot showing the relationship between
    cost rank (x-axis) and normalized throughput (y-axis), concatenated
    with a cumulative percentage chart showing how many groups have found
    their best runtime by each cost rank.

    For each configuration:
    - x = cost rank within its problem size group
    - y = (median iters/sec) / (best median iters/sec for that group)

    Points are binned into bins: [1.0, 0.9), [0.9, 0.8), etc.
    Circle size represents the count of data points in that bin.
    """
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns for histogram scatter chart: {sorted(missing)}"
        )

    # Prepare data in pandas
    df_prepared = df.copy()
    df_prepared["facet_label"] = (
        df_prepared["total_size"].astype(str)
        + ","
        + df_prepared["total_k_outer"].astype(str)
        + ","
        + df_prepared["total_k_inner"].astype(str)
    )

    # Aggregate to get median cost and runtime per configuration
    grouped = (
        df_prepared.groupby(
            [
                "facet_label",
                "total_size",
                "total_k_outer",
                "total_k_inner",
                "split_size",
                "tile_size",
                "buffer_level",
            ],
            dropna=False,
        )
        .agg({"cost": "median", "kernel_runtime": "median"})
        .reset_index()
        .rename(columns={"cost": "median_cost", "kernel_runtime": "median_runtime"})
    )

    # Sort by facet and cost to establish rank order
    grouped = grouped.sort_values(["facet_label", "median_cost"]).reset_index(drop=True)

    # Compute normalized throughput
    grouped["throughput"] = 1.0 / grouped["median_runtime"]
    grouped["best_throughput"] = grouped.groupby("facet_label")["throughput"].transform(
        "max"
    )
    grouped["normalized_throughput"] = (
        grouped["throughput"] / grouped["best_throughput"]
    )

    # Assign cost rank within each facet
    grouped["cost_rank"] = grouped.groupby("facet_label").cumcount() + 1

    # Convert normalized throughput to percentage
    grouped["throughput_pct"] = grouped["normalized_throughput"] * 100

    # Compute aspect ratio for grouped data
    grouped["aspect_ratio_raw"] = grouped["total_size"] / grouped["total_k_outer"]

    # Categorize aspect ratios into five groups
    def categorize_aspect_ratio(ratio):
        if ratio <= 0.25:
            return "≤0.25"
        elif ratio == 0.5:
            return "0.5"
        elif ratio == 1.0:
            return "1.0"
        elif ratio == 2.0:
            return "2.0"
        elif ratio >= 4.0:
            return "≥4.0"
        else:
            # For any values that don't exactly match, assign to nearest category
            if ratio < 0.5:
                return "≤0.25"
            elif ratio < 1.0:
                return "0.5"
            elif ratio < 2.0:
                return "1.0"
            elif ratio < 4.0:
                return "2.0"
            else:
                return "≥4.0"

    grouped["aspect_ratio_category"] = grouped["aspect_ratio_raw"].apply(
        categorize_aspect_ratio
    )

    # Create the histogram scatter chart
    histogram_chart = (
        alt.Chart(grouped)
        .mark_circle(color="#404040")
        .encode(
            x=alt.X(
                "cost_rank:Q",
                title="",
                bin=alt.Bin(step=1),
            ),
            y=alt.Y(
                "throughput_pct:Q",
                title="% of Peak Throughput",
                bin=alt.Bin(step=15),
                axis=alt.Axis(values=[100, 85, 70, 55, 40, 25, 10]),
            ),
            size=alt.Size("count():Q", title="Count"),
        )
        .properties(
            width=600,
            height=200,
        )
    )

    # For the cumulative chart, find at what rank each group reaches within 95% of best
    # (normalized_throughput >= 0.95)
    THRESHOLD = 0.95

    # For each group, find the first rank where normalized_throughput >= THRESHOLD
    within_threshold_rank = (
        grouped[grouped["normalized_throughput"] >= THRESHOLD]
        .groupby("facet_label")
        .agg(
            {
                "cost_rank": "min",
                "total_size": "first",
                "total_k_outer": "first",
            }
        )
        .reset_index()
        .rename(columns={"cost_rank": "rank_at_threshold"})
    )

    # Compute aspect ratio for each group
    within_threshold_rank["aspect_ratio_raw"] = (
        within_threshold_rank["total_size"] / within_threshold_rank["total_k_outer"]
    )

    # Categorize aspect ratios into five groups
    def categorize_aspect_ratio(ratio):
        if ratio <= 0.25:
            return "≤0.25"
        elif ratio == 0.5:
            return "0.5"
        elif ratio == 1.0:
            return "1.0"
        elif ratio == 2.0:
            return "2.0"
        elif ratio >= 4.0:
            return "≥4.0"
        else:
            # For any values that don't exactly match, assign to nearest category
            if ratio < 0.5:
                return "≤0.25"
            elif ratio < 1.0:
                return "0.5"
            elif ratio < 2.0:
                return "1.0"
            elif ratio < 4.0:
                return "2.0"
            else:
                return "≥4.0"

    within_threshold_rank["aspect_ratio_category"] = within_threshold_rank[
        "aspect_ratio_raw"
    ].apply(categorize_aspect_ratio)

    # Create a range of all possible ranks
    max_rank = int(grouped["cost_rank"].max())

    # For each aspect ratio category and rank, count cumulative groups that have reached threshold
    # Order categories from largest to smallest (reversed)
    cumulative_data = []
    for aspect_ratio_cat in ["≥4.0", "2.0", "1.0", "0.5", "≤0.25"]:
        aspect_groups = within_threshold_rank[
            within_threshold_rank["aspect_ratio_category"] == aspect_ratio_cat
        ]
        total_groups_for_aspect = len(aspect_groups)
        if total_groups_for_aspect == 0:
            continue
        aspect_ratio_label = f"{aspect_ratio_cat} (n={total_groups_for_aspect})"

        for rank in range(1, max_rank + 1):
            groups_found = (aspect_groups["rank_at_threshold"] <= rank).sum()
            cumulative_data.append(
                {
                    "cost_rank": rank,
                    "aspect_ratio_label": aspect_ratio_label,
                    "cumulative_pct": (groups_found / total_groups_for_aspect) * 100,
                }
            )

    cumulative_df = pd.DataFrame(cumulative_data)

    # Create the cumulative percentage chart with lines for each aspect ratio
    cumulative_chart = (
        alt.Chart(cumulative_df)
        .mark_line(
            point=False,
            interpolate="step-after",
        )
        .encode(
            x=alt.X(
                "cost_rank:Q",
                title="",
                scale=alt.Scale(domain=[1, max_rank]),
            ),
            y=alt.Y(
                "cumulative_pct:Q",
                title="% within 95% Peak",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "aspect_ratio_label:N",
                title="Aspect Ratio",
                sort=None,  # Preserve the order from the data
            ),
        )
        .properties(
            width=600,
            height=100,
        )
    )

    # Create count data for stacked area chart
    # Count configurations at each cost rank by aspect ratio category
    count_data = []
    for aspect_ratio_cat in ["≥4.0", "2.0", "1.0", "0.5", "≤0.25"]:
        aspect_grouped = grouped[grouped["aspect_ratio_category"] == aspect_ratio_cat]
        counts_by_rank = (
            aspect_grouped.groupby("cost_rank").size().reset_index(name="count")
        )

        # Get the count for this category
        cat_count = len(aspect_grouped["facet_label"].unique())
        aspect_ratio_label = f"{aspect_ratio_cat} (n={cat_count})"

        for _, row in counts_by_rank.iterrows():
            count_data.append(
                {
                    "cost_rank": row["cost_rank"],
                    "aspect_ratio_label": aspect_ratio_label,
                    "count": row["count"],
                }
            )

    count_df = pd.DataFrame(count_data)

    # Define the explicit order for aspect ratios (reversed)
    aspect_order = ["≥4.0", "2.0", "1.0", "0.5", "≤0.25"]

    # Create a mapping to get sort order
    count_df["sort_order"] = count_df["aspect_ratio_label"].apply(
        lambda x: next(
            (i for i, cat in enumerate(aspect_order) if x.startswith(cat)), 999
        )
    )

    # Create stacked area chart showing counts by aspect ratio
    count_chart = (
        alt.Chart(count_df)
        .mark_area()
        .encode(
            x=alt.X(
                "cost_rank:Q",
                title="Cost Rank",
                scale=alt.Scale(domain=[1, max_rank]),
            ),
            y=alt.Y(
                "count:Q",
                title="Number of Impls.",
                stack="zero",
            ),
            color=alt.Color(
                "aspect_ratio_label:N",
                title="Aspect Ratio",
                sort=None,
            ),
            order=alt.Order("sort_order:Q", sort="descending"),
        )
        .properties(
            width=600,
            height=80,
        )
    )

    return alt.vconcat(histogram_chart, cumulative_chart, count_chart).resolve_scale(
        x="shared"
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(sys.stdin)
    if df.empty:
        print("Input CSV is empty; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    alt.data_transformers.disable_max_rows()

    metric_title = METRIC_LABELS["kernel_runtime"]
    primary_title = args.title or f"{metric_title} by Tile and Split Size"
    primary_chart = build_metric_chart(df, "kernel_runtime", primary_title)

    bump_chart = build_bump_chart(df)
    cumulative_chart = build_cumulative_best_chart(df)
    histogram_scatter_chart = build_histogram_scatter_chart(df)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_output = output_dir / "runtime_vs_cost_scatter.svg"
    bump_output = output_dir / "runtime_vs_cost_bump.svg"
    cumulative_output = output_dir / "cumulative_best_by_cost_rank.svg"
    histogram_scatter_output = output_dir / "cost_rank_composite.svg"

    print("Saving runtime vs cost scatter...", file=sys.stderr)
    primary_chart.save(str(metric_output))
    print("Saving bump chart...", file=sys.stderr)
    bump_chart.save(str(bump_output))
    print("Saving faceted cumulative chart...", file=sys.stderr)
    cumulative_chart.save(str(cumulative_output))
    print("Saving histogram scatter chart...", file=sys.stderr)
    histogram_scatter_chart.save(str(histogram_scatter_output))

    errorbar_count, total_points = count_errorbar_points(df, "kernel_runtime")
    summary = pd.DataFrame(
        [
            {
                "metric": "kernel_runtime",
                "log_ratio_threshold": ERROR_BAR_LOG_RATIO_THRESHOLD,
                "errorbar_points": errorbar_count,
                "total_points": total_points,
            }
        ]
    )
    summary_output = output_dir / "chainexp_errors.csv"
    summary.to_csv(summary_output, index=False)


if __name__ == "__main__":
    main()
