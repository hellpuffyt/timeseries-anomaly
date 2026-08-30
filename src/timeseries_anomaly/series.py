"""Regular-grid time series representation and gap handling.

Real metric series are rarely perfectly regular: a scraper hiccups, an
agent restarts, a network blip drops a handful of samples. Decomposition
and thresholding both assume a fixed-frequency grid, so the first job is
to place the raw (timestamp, value) pairs onto one and be explicit about
what was missing rather than silently coercing a gap into a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class Series:
    """A single metric series resampled onto a regular time grid.

    Attributes
    ----------
    timestamps:
        Regular grid of timestamps, one per sample.
    values:
        Values aligned to ``timestamps``. Entries that are still missing
        after gap interpolation are ``NaN``.
    interpolated:
        True at indices whose value was filled in from a short gap.
    long_gap:
        True at indices that fall inside a gap too long to interpolate;
        these are excluded from decomposition and anomaly scoring.
    freq_seconds:
        Inferred (or supplied) spacing between grid points, in seconds.
    key:
        Optional series identifier, for multi-series input files.
    """

    timestamps: list[datetime]
    values: FloatArray
    interpolated: BoolArray
    long_gap: BoolArray
    freq_seconds: float
    key: str = "default"
    n_missing_original: int = 0
    n_short_gaps: int = 0
    n_long_gaps: int = 0
    long_gap_ranges: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def valid_mask(self) -> BoolArray:
        """Points usable for decomposition/detection: not NaN, not a long gap."""
        return ~np.isnan(self.values) & ~self.long_gap


def infer_freq_seconds(timestamps: list[datetime]) -> float:
    """Infer the sampling frequency as the median gap between timestamps."""
    if len(timestamps) < 2:
        raise ValueError("need at least 2 timestamps to infer a frequency")
    diffs = np.array(
        [(b - a).total_seconds() for a, b in zip(timestamps[:-1], timestamps[1:], strict=True)],
        dtype=np.float64,
    )
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("timestamps do not advance; cannot infer a frequency")
    return float(np.median(diffs))


def build_series(
    raw_timestamps: list[datetime],
    raw_values: FloatArray,
    key: str = "default",
    freq_seconds: float | None = None,
    max_gap_samples: int = 3,
) -> Series:
    """Sort, deduplicate, resample onto a regular grid, and handle gaps.

    Parameters
    ----------
    raw_timestamps, raw_values:
        The raw, possibly-irregular input.
    freq_seconds:
        Grid spacing to resample to. Inferred from the median spacing of
        ``raw_timestamps`` when not given.
    max_gap_samples:
        Runs of missing grid points up to this length are linearly
        interpolated. Longer runs are left as NaN and flagged via
        ``long_gap`` rather than being interpolated or treated as zero.
    """
    if len(raw_timestamps) != len(raw_values):
        raise ValueError("timestamps and values must have the same length")
    if len(raw_timestamps) == 0:
        raise ValueError("cannot build a series from zero points")

    order = np.argsort([t.timestamp() for t in raw_timestamps])
    ts_sorted = [raw_timestamps[i] for i in order]
    vals_sorted = raw_values[order]

    # Deduplicate exact-timestamp repeats by averaging.
    dedup_ts: list[datetime] = []
    dedup_vals: list[float] = []
    for t, v in zip(ts_sorted, vals_sorted, strict=True):
        if dedup_ts and dedup_ts[-1] == t:
            dedup_vals[-1] = (dedup_vals[-1] + v) / 2.0
        else:
            dedup_ts.append(t)
            dedup_vals.append(float(v))

    if freq_seconds is None:
        freq_seconds = infer_freq_seconds(dedup_ts) if len(dedup_ts) >= 2 else 3600.0

    if freq_seconds <= 0:
        raise ValueError("freq_seconds must be positive")

    start = dedup_ts[0]
    end = dedup_ts[-1]
    n_steps = int(round((end - start).total_seconds() / freq_seconds)) + 1
    grid = [start + timedelta(seconds=freq_seconds * i) for i in range(n_steps)]

    grid_values = np.full(n_steps, np.nan, dtype=np.float64)
    tol = freq_seconds * 0.4
    src_i = 0
    for gi, gt in enumerate(grid):
        while src_i < len(dedup_ts) and (dedup_ts[src_i] - gt).total_seconds() < -tol:
            src_i += 1
        if src_i < len(dedup_ts) and abs((dedup_ts[src_i] - gt).total_seconds()) <= tol:
            grid_values[gi] = dedup_vals[src_i]

    n_missing_original = int(np.isnan(grid_values).sum())

    interpolated = np.zeros(n_steps, dtype=bool)
    long_gap = np.zeros(n_steps, dtype=bool)
    long_gap_ranges: list[tuple[int, int]] = []
    n_short_gaps = 0
    n_long_gaps = 0

    is_nan = np.isnan(grid_values)
    i = 0
    while i < n_steps:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n_steps and is_nan[j]:
            j += 1
        gap_len = j - i
        has_left = i > 0
        has_right = j < n_steps
        if gap_len <= max_gap_samples and has_left and has_right:
            left_v = grid_values[i - 1]
            right_v = grid_values[j]
            for k in range(i, j):
                frac = (k - i + 1) / (gap_len + 1)
                grid_values[k] = left_v + frac * (right_v - left_v)
                interpolated[k] = True
            n_short_gaps += 1
        else:
            long_gap[i:j] = True
            long_gap_ranges.append((i, j))
            n_long_gaps += 1
        i = j

    return Series(
        timestamps=grid,
        values=grid_values,
        interpolated=interpolated,
        long_gap=long_gap,
        freq_seconds=freq_seconds,
        key=key,
        n_missing_original=n_missing_original,
        n_short_gaps=n_short_gaps,
        n_long_gaps=n_long_gaps,
        long_gap_ranges=long_gap_ranges,
    )
