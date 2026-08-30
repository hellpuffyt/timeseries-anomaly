from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import hourly_timestamps, weekly_seasonal_values
from timeseries_anomaly.detect import detect_anomalies, detect_level_shifts, severity_for
from timeseries_anomaly.series import Series, build_series


def _series(values: np.ndarray, key: str = "s", max_gap_samples: int = 3) -> Series:
    ts = hourly_timestamps(len(values))
    return build_series(ts, values, key=key, max_gap_samples=max_gap_samples)


def test_severity_buckets() -> None:
    assert severity_for(4.0, 4.0) == "low"
    assert severity_for(6.0, 4.0) == "medium"
    assert severity_for(8.0, 4.0) == "high"
    assert severity_for(12.0, 4.0) == "critical"


def test_clean_weekly_series_zero_anomalies_including_weekends() -> None:
    """Headline guarantee: a clean, exactly-seasonal series (weekends
    included) produces zero anomalies."""
    values = weekly_seasonal_values(24 * 21)  # noise_std=0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0)
    assert result.anomalies == []


def test_clean_weekly_series_with_mild_noise_mostly_zero_anomalies() -> None:
    rng = np.random.default_rng(42)
    values = weekly_seasonal_values(24 * 70, rng=rng, noise_std=1.0)
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.5)
    # Allow for rare tail noise but assert it stays negligible relative
    # to the series length -- no systematic weekend false-positive burst.
    assert len(result.anomalies) <= 3


def test_planted_spike_is_detected() -> None:
    rng = np.random.default_rng(7)
    values = weekly_seasonal_values(24 * 21, rng=rng, noise_std=0.5)
    spike_idx = 300
    values = values.copy()
    values[spike_idx] += 40.0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0)
    spike_anomalies = [a for a in result.anomalies if a.type == "spike"]
    assert any(a.start_index <= spike_idx <= a.end_index for a in spike_anomalies)


def test_planted_spike_severity_is_high_or_critical() -> None:
    values = weekly_seasonal_values(24 * 21)
    spike_idx = 300
    values = values.copy()
    values[spike_idx] += 100.0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0)
    hit = next(a for a in result.anomalies if a.start_index <= spike_idx <= a.end_index)
    assert hit.severity in ("high", "critical")


def test_planted_level_shift_classified_as_shift_not_spikes() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    shift_start = 400
    values[shift_start:] += 20.0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0, shift_window=24, shift_threshold=5.0)
    shift_anomalies = [a for a in result.anomalies if a.type == "level_shift"]
    assert len(shift_anomalies) >= 1
    hit = shift_anomalies[0]
    assert abs(hit.start_index - shift_start) <= 24
    # It must not ALSO be reported as a long run of individual spikes.
    spike_anomalies = [a for a in result.anomalies if a.type == "spike"]
    overlapping_spikes = [
        a for a in spike_anomalies if a.start_index < hit.end_index and a.end_index > hit.start_index
    ]
    assert overlapping_spikes == []


def test_level_shift_detector_ignores_transient_spike() -> None:
    """A one-sample spike that reverts must not be mistaken for a shift."""
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300] += 50.0  # reverts immediately after
    valid = np.ones(len(values), dtype=bool)
    deseasonalized = values  # already deterministic/no seasonal noise here
    shifts = detect_level_shifts(deseasonalized, valid, window=24, threshold=5.0)
    assert shifts == []


def test_level_shift_detector_finds_sustained_step() -> None:
    n = 24 * 10
    values = np.zeros(n)
    values[n // 2 :] += 10.0
    valid = np.ones(n, dtype=bool)
    shifts = detect_level_shifts(values, valid, window=24, threshold=3.0)
    assert len(shifts) == 1
    onset, z = shifts[0]
    assert abs(onset - n // 2) <= 24
    assert z > 3.0


def test_flat_constant_series_no_anomalies_no_crash() -> None:
    values = np.full(24 * 21, 5.0)
    series = _series(values)
    with np.errstate(all="raise"):
        result = detect_anomalies(series, period=168, threshold=4.0)
    assert result.anomalies == []


def test_min_consecutive_suppresses_single_sample_noise() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300] += 30.0  # isolated single-sample spike
    series = _series(values)
    result_default = detect_anomalies(series, period=168, threshold=4.0, min_consecutive=1)
    result_strict = detect_anomalies(series, period=168, threshold=4.0, min_consecutive=3)
    assert any(a.start_index <= 300 <= a.end_index for a in result_default.anomalies)
    assert not any(a.start_index <= 300 <= a.end_index and a.type == "spike" for a in result_strict.anomalies)


def test_min_consecutive_keeps_runs_long_enough() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300:304] += 30.0  # 4-sample burst
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0, min_consecutive=3)
    assert any(a.start_index <= 300 <= a.end_index for a in result.anomalies)


def test_long_gap_excluded_from_anomaly_scoring() -> None:
    values = weekly_seasonal_values(24 * 21)
    ts = hourly_timestamps(len(values))
    keep = [i for i in range(len(values)) if not (200 <= i < 210)]
    series = build_series([ts[i] for i in keep], values[keep], max_gap_samples=3)
    result = detect_anomalies(series, period=168, threshold=4.0)
    assert result.n_long_gaps == 1
    # No anomaly should be reported inside the long-gap range.
    for a in result.anomalies:
        assert not (a.start_index <= 205 <= a.end_index)


def test_short_gap_interpolated_series_still_detects_real_spike() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300] += 40.0
    ts = hourly_timestamps(len(values))
    keep = [i for i in range(len(values)) if i != 100]  # single-sample short gap
    series = build_series([ts[i] for i in keep], values[keep], max_gap_samples=3)
    assert series.n_short_gaps == 1
    result = detect_anomalies(series, period=168, threshold=4.0)
    assert any(a.start_index <= 300 <= a.end_index for a in result.anomalies)


def test_detect_anomalies_rejects_empty_series() -> None:
    # An empty Series cannot be built via build_series (raises earlier),
    # so directly assert detect_anomalies guards len==0 defensively.
    from timeseries_anomaly.series import Series

    empty = Series(timestamps=[], values=np.array([]), interpolated=np.array([], dtype=bool),
                    long_gap=np.array([], dtype=bool), freq_seconds=3600.0)
    with pytest.raises(ValueError):
        detect_anomalies(empty)


def test_hampel_method_runs_and_detects_spike() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300] += 40.0
    series = _series(values)
    result = detect_anomalies(series, period=168, method="hampel", threshold=4.0, hampel_window=12)
    assert any(a.start_index <= 300 <= a.end_index for a in result.anomalies)


def test_detection_result_to_dict_roundtrips_json_shape() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300] += 40.0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0)
    d = result.to_dict()
    assert d["key"] == "s"
    assert isinstance(d["anomalies"], list)
    assert d["anomaly_count"] == len(result.anomalies)
    for a in d["anomalies"]:
        assert set(a.keys()) >= {"type", "start_time", "end_time", "severity", "max_abs_zscore"}


def test_anomaly_length_property() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[300:305] += 30.0
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=4.0, min_consecutive=1)
    hit = next(a for a in result.anomalies if a.start_index <= 300 <= a.end_index)
    assert hit.length == hit.end_index - hit.start_index + 1


def test_weekend_dip_alone_is_not_flagged_even_at_low_threshold() -> None:
    """The specific false-positive this package exists to kill."""
    values = weekly_seasonal_values(24 * 42)  # 6 clean weeks, noise-free
    series = _series(values)
    result = detect_anomalies(series, period=168, threshold=3.0)
    weekend_hits = [
        a
        for a in result.anomalies
        if any((i // 24) % 7 in (5, 6) for i in range(a.start_index, a.end_index + 1))
    ]
    assert weekend_hits == []
