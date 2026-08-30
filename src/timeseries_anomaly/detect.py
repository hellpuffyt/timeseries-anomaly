"""Anomaly detection on top of the seasonal decomposition.

Two distinct kinds of event are told apart deliberately, because they
call for different responses on-call:

* a **spike**: one or a short run of points that deviate sharply and then
  the series returns to its expected shape.
* a **level shift**: a sustained step to a new baseline. Reporting this
  as "47 consecutive anomalies" buries the one thing an operator needs
  to know (the baseline moved) under noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt

from timeseries_anomaly.decompose import DecompositionResult, decompose
from timeseries_anomaly.robust import hampel_filter, median_mad, modified_zscore
from timeseries_anomaly.series import Series

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

AnomalyType = Literal["spike", "level_shift"]
Severity = Literal["low", "medium", "high", "critical"]

Method = Literal["robust", "hampel"]


def severity_for(abs_z: float, threshold: float) -> Severity:
    if abs_z >= threshold * 3:
        return "critical"
    if abs_z >= threshold * 2:
        return "high"
    if abs_z >= threshold * 1.5:
        return "medium"
    return "low"


@dataclass
class Anomaly:
    type: AnomalyType
    start_index: int
    end_index: int  # inclusive
    start_time: datetime
    end_time: datetime
    severity: Severity
    max_abs_zscore: float
    values: list[float]
    description: str

    @property
    def length(self) -> int:
        return self.end_index - self.start_index + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "length": self.length,
            "severity": self.severity,
            "max_abs_zscore": round(self.max_abs_zscore, 4),
            "values": [round(v, 6) for v in self.values],
            "description": self.description,
        }


@dataclass
class DetectionResult:
    key: str
    decomposition: DecompositionResult
    anomalies: list[Anomaly]
    n_points: int
    n_short_gaps: int
    n_long_gaps: int
    long_gap_ranges: list[tuple[datetime, datetime]]
    period: int
    method: Method
    threshold: float
    zscores: FloatArray = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "period": self.period,
            "method": self.method,
            "threshold": self.threshold,
            "n_points": self.n_points,
            "n_short_gaps": self.n_short_gaps,
            "n_long_gaps": self.n_long_gaps,
            "long_gap_ranges": [[s.isoformat(), e.isoformat()] for s, e in self.long_gap_ranges],
            "anomaly_count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


def _runs_of_true(mask: BoolArray) -> list[tuple[int, int]]:
    """Return (start, end-inclusive) index pairs for each run of True."""
    runs: list[tuple[int, int]] = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def detect_level_shifts(
    deseasonalized: FloatArray,
    valid: BoolArray,
    window: int = 24,
    threshold: float = 5.0,
) -> list[tuple[int, float]]:
    """Find sustained step changes in a deseasonalized series.

    For each candidate index, compares the robust centre of the
    ``window`` points before it to the ``window`` points after it. A
    step is reported when that before/after difference is large relative
    to the pooled local spread *and* the "after" window is itself
    internally stable (low spread) -- i.e. the series actually settled at
    a new level rather than just producing one more spike.

    Returns a list of (onset_index, z_score) for non-overlapping shifts,
    scanned left to right with a cool-down of ``window`` samples after
    each detected onset.
    """
    n = deseasonalized.size
    shifts: list[tuple[int, float]] = []
    if n < window * 2 + 1:
        return shifts

    # A magnitude-aware floor so that floating point noise on an
    # otherwise perfectly flat/clean series (differences on the order of
    # 1e-13) is never mistaken for a "hard step" just because the local
    # spread also happens to round to exactly zero.
    finite_all = deseasonalized[valid & np.isfinite(deseasonalized)]
    overall_scale = float(np.median(np.abs(finite_all))) if finite_all.size else 0.0
    min_abs_jump = 1e-9 * (overall_scale + 1.0)

    i = window
    while i < n - window:
        before = deseasonalized[i - window : i]
        after = deseasonalized[i : i + window]
        before_valid = valid[i - window : i]
        after_valid = valid[i : i + window]
        b_good = before[before_valid]
        a_good = after[after_valid]
        if b_good.size < window // 2 or a_good.size < window // 2:
            i += 1
            continue
        bm, b_mad = median_mad(b_good)
        am, a_mad = median_mad(a_good)
        pooled = (b_mad + a_mad) / 2.0
        diff = am - bm
        if abs(diff) <= min_abs_jump:
            i += 1
            continue
        # No meaningful spread on either side: a jump clearing the
        # absolute floor above is a hard step regardless of scale.
        z = diff / pooled if pooled > min_abs_jump else np.sign(diff) * 1e9
        # Require the "after" window to be relatively stable (a real new
        # level, not just noisy) -- its spread should not itself be huge
        # compared to the jump.
        stable_after = a_mad <= abs(diff) * 1.5 + min_abs_jump
        if abs(z) >= threshold and stable_after:
            shifts.append((i, float(z)))
            i += window  # cool down so one shift isn't reported many times
        else:
            i += 1
    return shifts


def detect_anomalies(
    series: Series,
    period: int = 168,
    method: Method = "robust",
    threshold: float = 4.0,
    hampel_window: int = 7,
    min_consecutive: int = 1,
    shift_window: int = 24,
    shift_threshold: float = 5.0,
) -> DetectionResult:
    """Run seasonal decomposition and flag anomalies against a robust baseline.

    Parameters
    ----------
    series:
        A gap-handled, regular-grid :class:`~timeseries_anomaly.series.Series`.
    period:
        Seasonal period in samples (e.g. 168 for weekly seasonality on
        hourly data). Comparing each point to others at the same phase of
        this period is what suppresses legitimate, recurring dips (like
        a quiet Saturday night) from being flagged.
    method:
        ``"robust"`` scores the remainder against one global median/MAD.
        ``"hampel"`` instead uses a rolling local median/MAD, which can
        catch anomalies in series whose noise level itself drifts.
    threshold:
        Modified z-score magnitude above which a point is anomalous.
    min_consecutive:
        Runs of point-anomalies shorter than this are suppressed as
        noise. Level shifts are exempt (they are a distinct signal).
    shift_window, shift_threshold:
        Passed to :func:`detect_level_shifts`.
    """
    if len(series) == 0:
        raise ValueError("empty series")

    valid = series.valid_mask
    decomposition = decompose(series.values, valid=valid, period=period)
    remainder = decomposition.remainder

    if method == "robust":
        center, scale = median_mad(remainder[valid]) if valid.any() else (0.0, 0.0)
        z = modified_zscore(remainder, center=center, scale=scale)
    elif method == "hampel":
        z, _ = hampel_filter(remainder, window=hampel_window, n_sigmas=threshold)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"unknown method: {method}")

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_mask = valid & (np.abs(z) > threshold)

    deseasonalized = series.values - decomposition.seasonal
    shifts = detect_level_shifts(deseasonalized, valid, window=shift_window, threshold=shift_threshold)

    anomalies: list[Anomaly] = []
    suppressed = np.zeros(len(series), dtype=bool)
    n = len(series)

    for onset, shift_z in shifts:
        band_start = max(0, onset - shift_window // 2)
        band_end = min(n, onset + shift_window)
        suppressed[band_start:band_end] = True
        after_vals = series.values[onset : min(n, onset + shift_window)]
        after_vals = after_vals[~np.isnan(after_vals)]
        before_vals = series.values[max(0, onset - shift_window) : onset]
        before_vals = before_vals[~np.isnan(before_vals)]
        before_level = float(np.median(before_vals)) if before_vals.size else float("nan")
        after_level = float(np.median(after_vals)) if after_vals.size else float("nan")
        anomalies.append(
            Anomaly(
                type="level_shift",
                start_index=onset,
                end_index=min(n - 1, onset + shift_window - 1),
                start_time=series.timestamps[onset],
                end_time=series.timestamps[min(n - 1, onset + shift_window - 1)],
                severity=severity_for(abs(shift_z), shift_threshold),
                max_abs_zscore=abs(shift_z),
                values=[round(v, 6) for v in series.values[onset : min(n, onset + shift_window)].tolist()],
                description=(
                    f"sustained level shift: baseline moved from ~{before_level:.4g} to ~{after_level:.4g}"
                ),
            )
        )

    remaining_mask = candidate_mask & ~suppressed
    for start, end in _runs_of_true(remaining_mask):
        if (end - start + 1) < min_consecutive:
            continue
        run_z = np.abs(z[start : end + 1])
        max_z = float(np.max(run_z)) if run_z.size else 0.0
        anomalies.append(
            Anomaly(
                type="spike",
                start_index=start,
                end_index=end,
                start_time=series.timestamps[start],
                end_time=series.timestamps[end],
                severity=severity_for(max_z, threshold),
                max_abs_zscore=max_z,
                values=[round(v, 6) for v in series.values[start : end + 1].tolist()],
                description=(
                    f"deviates {max_z:.2f}x the robust noise level from the expected seasonal baseline"
                ),
            )
        )

    anomalies.sort(key=lambda a: a.start_index)

    return DetectionResult(
        key=series.key,
        decomposition=decomposition,
        anomalies=anomalies,
        n_points=n,
        n_short_gaps=series.n_short_gaps,
        n_long_gaps=series.n_long_gaps,
        long_gap_ranges=[
            (series.timestamps[s], series.timestamps[e - 1]) for s, e in series.long_gap_ranges
        ],
        period=period,
        method=method,
        threshold=threshold,
        zscores=z,
    )
