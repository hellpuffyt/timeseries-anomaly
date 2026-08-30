from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.conftest import hourly_timestamps
from timeseries_anomaly.series import build_series, infer_freq_seconds


def test_infer_freq_seconds_hourly() -> None:
    ts = hourly_timestamps(10)
    assert infer_freq_seconds(ts) == 3600.0


def test_infer_freq_seconds_requires_two_points() -> None:
    with pytest.raises(ValueError):
        infer_freq_seconds([datetime(2026, 1, 1)])


def test_build_series_no_gaps() -> None:
    ts = hourly_timestamps(24)
    vals = np.arange(24, dtype=np.float64)
    s = build_series(ts, vals)
    assert len(s) == 24
    assert s.n_short_gaps == 0
    assert s.n_long_gaps == 0
    np.testing.assert_allclose(s.values, vals)


def test_build_series_sorts_unsorted_input() -> None:
    ts = hourly_timestamps(5)
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    order = [3, 1, 4, 0, 2]
    s = build_series([ts[i] for i in order], vals[order])
    np.testing.assert_allclose(s.values, vals)
    assert s.timestamps == ts


def test_build_series_deduplicates_repeated_timestamps() -> None:
    ts = [hourly_timestamps(3)[0]] * 2 + hourly_timestamps(3)[1:]
    vals = np.array([10.0, 20.0, 1.0, 2.0])
    s = build_series(ts, vals)
    assert len(s) == 3
    assert s.values[0] == 15.0  # averaged duplicate


def test_build_series_short_gap_is_interpolated() -> None:
    ts = hourly_timestamps(10)
    vals = np.arange(10, dtype=np.float64)
    # drop indices 4 and 5 (a 2-sample gap)
    keep = [i for i in range(10) if i not in (4, 5)]
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert s.n_short_gaps == 1
    assert s.n_long_gaps == 0
    assert s.interpolated[4]
    assert s.interpolated[5]
    # linear interpolation between 3.0 and 6.0
    assert abs(s.values[4] - 4.0) < 1e-9
    assert abs(s.values[5] - 5.0) < 1e-9


def test_build_series_long_gap_is_flagged_not_interpolated() -> None:
    ts = hourly_timestamps(20)
    vals = np.arange(20, dtype=np.float64)
    keep = [i for i in range(20) if not (5 <= i <= 12)]  # 8-sample gap
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert s.n_long_gaps == 1
    assert s.n_short_gaps == 0
    assert s.long_gap[5:13].all()
    assert np.isnan(s.values[5:13]).all()
    assert not s.interpolated[5:13].any()


def test_build_series_long_gap_excluded_from_valid_mask() -> None:
    ts = hourly_timestamps(20)
    vals = np.arange(20, dtype=np.float64)
    keep = [i for i in range(20) if not (5 <= i <= 12)]
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert not s.valid_mask[5:13].any()
    assert s.valid_mask[:5].all()


def test_build_series_grid_domain_starts_and_ends_on_real_data() -> None:
    """The grid always starts/ends at an observed timestamp, so a missing
    leading/trailing sample just shrinks the grid rather than creating an
    unfillable edge gap."""
    ts = hourly_timestamps(10)
    vals = np.arange(10, dtype=np.float64)
    keep = [i for i in range(10) if i != 0]  # drop the very first sample
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert len(s) == 9
    assert s.timestamps[0] == ts[1]
    assert not s.long_gap.any()
    assert not s.interpolated.any()


def test_build_series_gap_exactly_at_max_gap_samples_is_interpolated() -> None:
    ts = hourly_timestamps(12)
    vals = np.arange(12, dtype=np.float64)
    keep = [i for i in range(12) if i not in (5, 6, 7)]  # exactly 3-sample gap
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert s.n_short_gaps == 1
    assert s.n_long_gaps == 0
    assert s.interpolated[5:8].all()


def test_build_series_gap_one_longer_than_max_is_flagged() -> None:
    ts = hourly_timestamps(13)
    vals = np.arange(13, dtype=np.float64)
    keep = [i for i in range(13) if i not in (5, 6, 7, 8)]  # 4-sample gap
    s = build_series([ts[i] for i in keep], vals[keep], max_gap_samples=3)
    assert s.n_long_gaps == 1
    assert s.n_short_gaps == 0
    assert s.long_gap[5:9].all()


def test_build_series_rejects_mismatched_lengths() -> None:
    ts = hourly_timestamps(5)
    vals = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError):
        build_series(ts, vals)


def test_build_series_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_series([], np.array([], dtype=np.float64))


def test_build_series_explicit_freq_seconds() -> None:
    ts = [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(4)]
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    s = build_series(ts, vals, freq_seconds=900.0)
    assert s.freq_seconds == 900.0
    assert len(s) == 4


def test_build_series_key_is_preserved() -> None:
    ts = hourly_timestamps(5)
    vals = np.arange(5, dtype=np.float64)
    s = build_series(ts, vals, key="cpu")
    assert s.key == "cpu"
