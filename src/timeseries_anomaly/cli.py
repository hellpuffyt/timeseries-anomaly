"""Command-line interface for timeseries-anomaly."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from timeseries_anomaly.detect import Anomaly, DetectionResult, detect_anomalies
from timeseries_anomaly.io import load_file
from timeseries_anomaly.series import build_series
from timeseries_anomaly.sparkline import sparkline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timeseries-anomaly",
        description=(
            "Detect anomalies in metric time series using seasonal decomposition and robust thresholds."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="detect anomalies in a CSV/JSON series file")
    detect_p.add_argument("input", help="path to a .csv or .json series file")
    detect_p.add_argument(
        "--period",
        type=int,
        default=None,
        help="seasonal period in samples (default: 168 for hourly data spanning >=2 weeks, else 24, else 0)",
    )
    detect_p.add_argument("--method", choices=["robust", "hampel"], default="robust")
    detect_p.add_argument(
        "--threshold", type=float, default=4.0, help="modified z-score anomaly threshold"
    )
    detect_p.add_argument("--hampel-window", type=int, default=7)
    detect_p.add_argument(
        "--min-consecutive", type=int, default=1, help="suppress spike runs shorter than this"
    )
    detect_p.add_argument("--shift-window", type=int, default=24)
    detect_p.add_argument("--shift-threshold", type=float, default=5.0)
    detect_p.add_argument(
        "--max-gap-samples", type=int, default=3, help="longest gap to linearly interpolate"
    )
    detect_p.add_argument(
        "--freq-seconds", type=float, default=None, help="override inferred sample spacing"
    )
    detect_p.add_argument(
        "--json", dest="json_out", default=None, help="write detailed JSON report to this path"
    )
    detect_p.add_argument(
        "--sparkline", action="store_true", help="print an ASCII sparkline with anomalies marked"
    )
    detect_p.add_argument("--sparkline-width", type=int, default=120)
    detect_p.add_argument("--quiet", action="store_true", help="suppress the human-readable summary")

    gen_p = sub.add_parser("generate", help="generate a synthetic seasonal series (for demos/tests)")
    gen_p.add_argument("--output", required=True, help="output CSV path")
    gen_p.add_argument(
        "--days", type=int, default=70, help="more weeks means a more reliable seasonal estimate"
    )
    gen_p.add_argument("--seed", type=int, default=0)
    gen_p.add_argument("--with-anomalies", action="store_true", help="plant a spike and a level shift")
    gen_p.add_argument("--noise", type=float, default=1.0, help="stddev of injected gaussian noise")
    gen_p.add_argument(
        "--start",
        default="2026-01-05T00:00:00",
        help="ISO timestamp for the first sample (a Monday by default)",
    )

    return parser


def _infer_default_period(n_points: int, freq_seconds: float) -> int:
    hourly = abs(freq_seconds - 3600.0) < 1e-6
    if hourly and n_points >= 168 * 2:
        return 168
    if hourly and n_points >= 24 * 2:
        return 24
    # Fall back to a quarter of the series length (min 2) so decomposition
    # still runs on short or non-hourly series, just without a real
    # seasonal claim.
    return max(2, min(24, n_points // 4)) if n_points >= 8 else 2


def _print_summary(result: DetectionResult) -> None:
    print(f"series: {result.key}")
    print(f"points: {result.n_points}  period: {result.period}  method: {result.method}")
    print(f"gaps: {result.n_short_gaps} interpolated, {result.n_long_gaps} flagged as long gaps")
    if result.long_gap_ranges:
        for s, e in result.long_gap_ranges:
            print(f"  long gap: {s.isoformat()} -> {e.isoformat()}")
    print(f"anomalies found: {len(result.anomalies)}")
    for a in result.anomalies:
        print(
            f"  [{a.severity:<8}] {a.type:<12} {a.start_time.isoformat()} -> {a.end_time.isoformat()} "
            f"(n={a.length}, |z|={a.max_abs_zscore:.2f}) {a.description}"
        )


def _print_sparkline(values: np.ndarray, anomalies: list[Anomaly], width: int) -> None:
    mask = np.zeros(values.size, dtype=bool)
    for a in anomalies:
        mask[a.start_index : a.end_index + 1] = True

    n = values.size
    chunk = min(width, n) if width > 0 else n
    if chunk <= 0:
        return
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        line = sparkline(values[start:end])
        markers = "".join("^" if bool(m) else " " for m in mask[start:end])
        print(line)
        print(markers)


def cmd_detect(args: argparse.Namespace) -> int:
    data = load_file(args.input)
    exit_code = 0
    for key, (timestamps, values) in data.items():
        series = build_series(
            timestamps,
            values,
            key=key,
            freq_seconds=args.freq_seconds,
            max_gap_samples=args.max_gap_samples,
        )
        period = (
            args.period
            if args.period is not None
            else _infer_default_period(len(series), series.freq_seconds)
        )
        result = detect_anomalies(
            series,
            period=period,
            method=args.method,
            threshold=args.threshold,
            hampel_window=args.hampel_window,
            min_consecutive=args.min_consecutive,
            shift_window=args.shift_window,
            shift_threshold=args.shift_threshold,
        )
        if not args.quiet:
            _print_summary(result)
            if args.sparkline:
                _print_sparkline(series.values, result.anomalies, args.sparkline_width)
            print()
        if result.anomalies:
            exit_code = 3

        if args.json_out:
            out_path = Path(args.json_out)
            if len(data) > 1:
                out_path = out_path.with_name(f"{out_path.stem}.{key}{out_path.suffix}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            if not args.quiet:
                print(f"wrote {out_path}")

    return exit_code


def cmd_generate(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    n = args.days * 24
    start = datetime.fromisoformat(args.start)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    hour_of_day = np.array([t.hour for t in timestamps], dtype=np.float64)
    day_of_week = np.array([t.weekday() for t in timestamps], dtype=np.float64)  # 0=Mon .. 6=Sun
    is_weekend = day_of_week >= 5

    daily = 10.0 + 6.0 * np.sin((hour_of_day - 8.0) / 24.0 * 2 * np.pi)
    weekend_scale = np.where(is_weekend, 0.5, 1.0)
    baseline = 20.0 + daily * weekend_scale
    noise = rng.normal(0.0, args.noise, size=n)
    values = baseline + noise

    if args.with_anomalies:
        spike_idx = n // 3
        values[spike_idx] += 40.0
        shift_start = (2 * n) // 3
        values[shift_start:] += 15.0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])
        for t, v in zip(timestamps, values, strict=True):
            writer.writerow([t.isoformat(), f"{v:.4f}"])

    print(f"wrote {n} points to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        return cmd_detect(args)
    if args.command == "generate":
        return cmd_generate(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
