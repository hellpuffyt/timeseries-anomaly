"""End-to-end example: build a synthetic series with a planted spike and
level shift, run detection, and print the results.

Run with:  python examples/generate_demo.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from timeseries_anomaly.detect import detect_anomalies
from timeseries_anomaly.series import build_series
from timeseries_anomaly.sparkline import sparkline


def main() -> None:
    n_days = 70
    n = n_days * 24
    start = datetime(2026, 1, 5)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    hour_of_day = np.array([t.hour for t in timestamps], dtype=np.float64)
    day_of_week = np.array([t.weekday() for t in timestamps], dtype=np.float64)
    is_weekend = day_of_week >= 5

    daily = 10.0 + 6.0 * np.sin((hour_of_day - 8.0) / 24.0 * 2 * np.pi)
    weekend_scale = np.where(is_weekend, 0.5, 1.0)
    baseline = 20.0 + daily * weekend_scale

    rng = np.random.default_rng(42)
    values = baseline + rng.normal(0.0, 1.0, size=n)

    # Plant a spike and a sustained level shift.
    values[n // 3] += 40.0
    values[(2 * n) // 3 :] += 15.0

    series = build_series(timestamps, values)
    result = detect_anomalies(series, period=168, threshold=4.0)

    print(f"{result.n_points} points, period={result.period}")
    print(f"{len(result.anomalies)} anomalies found:\n")
    for a in result.anomalies:
        print(f"  [{a.severity}] {a.type} at {a.start_time} (|z|={a.max_abs_zscore:.1f}): {a.description}")

    print("\nsparkline (first two weeks):")
    print(sparkline(series.values[: 24 * 14]))


if __name__ == "__main__":
    main()
