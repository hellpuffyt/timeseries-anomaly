"""Shared fixtures/helpers for building synthetic series with known ground truth."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

BASE_START = datetime(2026, 1, 5, 0, 0, 0)  # a Monday


def hourly_timestamps(n: int, start: datetime = BASE_START) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(n)]


def weekly_seasonal_values(
    n: int,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
    start: datetime = BASE_START,
) -> FloatArray:
    """A deterministic daily+weekly seasonal pattern, optionally with gaussian noise."""
    timestamps = hourly_timestamps(n, start)
    hour_of_day = np.array([t.hour for t in timestamps], dtype=np.float64)
    day_of_week = np.array([t.weekday() for t in timestamps], dtype=np.float64)
    is_weekend = day_of_week >= 5
    daily = 10.0 + 6.0 * np.sin((hour_of_day - 8.0) / 24.0 * 2 * np.pi)
    weekend_scale = np.where(is_weekend, 0.5, 1.0)
    baseline = 20.0 + daily * weekend_scale
    if noise_std > 0 and rng is not None:
        baseline = baseline + rng.normal(0.0, noise_std, size=n)
    return baseline.astype(np.float64)


def daily_seasonal_values(
    n: int, rng: np.random.Generator | None = None, noise_std: float = 0.0
) -> FloatArray:
    """A pure 24-hour sinusoidal pattern (no weekly component)."""
    hours = np.arange(n, dtype=np.float64) % 24
    values = 50.0 + 10.0 * np.sin(hours / 24.0 * 2 * np.pi)
    if noise_std > 0 and rng is not None:
        values = values + rng.normal(0.0, noise_std, size=n)
    return values.astype(np.float64)
