"""STL-style additive decomposition built from numpy primitives only.

The classical STL algorithm from Cleveland et al. uses loess smoothing.
We use a simpler, deliberately robust pipeline that is easy to reason
about and to test:

1. **Trend**: a centred moving *median* over one seasonal period. A
   median (rather than a mean) does not get dragged by the handful of
   extreme points a moving average would smear across its whole window.
2. **Seasonal**: after removing the trend, average the residual by
   position-within-period (hour-of-day for period=24, hour-of-week for
   period=168 on hourly data) using the median, then centre the pattern
   so it sums to ~0 across a period. This is the piece that lets a
   Saturday 3am reading be judged against other Saturday 3am readings
   instead of against the whole week.
3. **Remainder**: whatever is left, ``value - trend - seasonal``.

The three components sum back to the original series (within floating
point tolerance) everywhere the input was not NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class DecompositionResult:
    trend: FloatArray
    seasonal: FloatArray
    remainder: FloatArray
    period: int
    seasonal_pattern: FloatArray  # length == period


def _centered_moving_median(values: FloatArray, valid: BoolArray, window: int) -> FloatArray:
    """Centred moving median with an odd window, NaN where too few valid points."""
    n = values.size
    half = window // 2
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = values[lo:hi]
        seg_valid = valid[lo:hi]
        good = segment[seg_valid]
        # Require at least half the window to have real data.
        if good.size >= max(3, (hi - lo) // 2):
            out[i] = float(np.median(good))
    return out


def _seasonal_pattern_from(
    residual: FloatArray, residual_valid: BoolArray, n: int, period: int
) -> FloatArray:
    """Median residual per phase-within-period, centred to sum to ~0."""
    pattern = np.zeros(period, dtype=np.float64)
    if n < period * 2:
        return pattern
    for phase in range(period):
        idx = np.arange(phase, n, period)
        seg = residual[idx]
        seg_valid = residual_valid[idx]
        good = seg[seg_valid]
        if good.size > 0:
            pattern[phase] = float(np.median(good))
    finite_pattern = pattern[np.isfinite(pattern)]
    if finite_pattern.size:
        pattern -= float(np.mean(finite_pattern))
    return pattern


def decompose(
    values: FloatArray,
    valid: BoolArray | None = None,
    period: int = 168,
) -> DecompositionResult:
    """Decompose ``values`` into trend + seasonal + remainder.

    Parameters
    ----------
    values:
        The series, aligned to a regular grid. NaNs are tolerated and
        excluded from every statistic (pass ``valid`` to also exclude
        long-gap placeholders that were never given a real value).
    period:
        Number of samples per seasonal cycle (e.g. 24 for a daily cycle
        on hourly data, 168 for a weekly cycle on hourly data). Must be
        >= 2. If the series is shorter than two periods, seasonality
        cannot be reliably estimated and the seasonal component is
        returned as all-zero.

    Implementation note: a plain single-pass "moving-median trend, then
    seasonal from the detrended residual" is biased near the series
    edges, where the trend window is truncated and no longer contains a
    balanced mix of every phase (e.g. a truncated window over a weekly
    period might catch mostly weekdays). That imbalance leaks weekly
    amplitude into the edge trend and shows up as spurious remainder
    spikes right at the start/end of the series. To avoid it, seasonality
    is estimated in two passes: a rough seasonal pattern is fit to the
    raw values first, the series is deseasonalized *before* the moving
    median is taken (so a truncated trend window is no longer biased by
    the seasonal shape), and the seasonal pattern is then refined from
    the trend-corrected residual.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    n = values.size
    if valid is None:
        valid = ~np.isnan(values)

    window = period if period % 2 == 1 else period + 1

    # Pass 1: rough seasonal estimate straight from the raw values.
    rough_seasonal_pattern = _seasonal_pattern_from(values, valid, n, period)
    phases = np.arange(n) % period
    rough_seasonal = rough_seasonal_pattern[phases]

    # Pass 2: trend from the deseasonalized series, so edge truncation of
    # the moving-median window no longer carries a seasonal bias.
    deseasonalized = values - rough_seasonal
    trend = _centered_moving_median(deseasonalized, valid, window)

    # Pass 3: refine the seasonal pattern from the trend-corrected residual.
    detrended = values - trend
    detrended_valid = valid & ~np.isnan(trend)
    seasonal_pattern = _seasonal_pattern_from(detrended, detrended_valid, n, period)
    seasonal = seasonal_pattern[phases]

    remainder = values - trend - seasonal
    # Where trend is NaN (edges / sparse data), fall back to a value that
    # keeps trend+seasonal+remainder == values exactly, with remainder 0,
    # rather than propagating NaN through the whole pipeline.
    nan_trend = np.isnan(trend)
    if np.any(nan_trend):
        trend = trend.copy()
        trend[nan_trend] = values[nan_trend] - seasonal[nan_trend]
        remainder = values - trend - seasonal

    return DecompositionResult(
        trend=trend, seasonal=seasonal, remainder=remainder, period=period, seasonal_pattern=seasonal_pattern
    )
