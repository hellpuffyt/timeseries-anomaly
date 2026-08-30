from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import daily_seasonal_values, weekly_seasonal_values
from timeseries_anomaly.decompose import decompose


def test_decompose_rejects_period_below_2() -> None:
    with pytest.raises(ValueError):
        decompose(np.arange(10, dtype=np.float64), period=1)


def test_decompose_reconstructs_original_within_tolerance() -> None:
    rng = np.random.default_rng(3)
    values = weekly_seasonal_values(24 * 21, rng=rng, noise_std=1.0)
    result = decompose(values, period=168)
    reconstructed = result.trend + result.seasonal + result.remainder
    np.testing.assert_allclose(reconstructed, values, atol=1e-9)


def test_decompose_reconstructs_with_nans_present() -> None:
    values = weekly_seasonal_values(24 * 21)
    values = values.copy()
    values[10:13] = np.nan
    valid = ~np.isnan(values)
    result = decompose(values, valid=valid, period=168)
    finite = valid
    # Reconstruction identity should hold everywhere values is finite;
    # where values is NaN, trend/seasonal/remainder are still finite by
    # construction and don't propagate NaN through the pipeline.
    assert not np.isnan(result.trend).any()
    assert not np.isnan(result.seasonal).any()
    assert not np.isnan(result.remainder[finite]).any()


def test_decompose_clean_weekly_series_has_near_zero_remainder() -> None:
    """No noise, exact weekly seasonality: remainder should vanish."""
    values = weekly_seasonal_values(24 * 21)  # noise_std=0
    result = decompose(values, period=168)
    assert np.max(np.abs(result.remainder)) < 1e-6


def test_decompose_seasonal_pattern_length_matches_period() -> None:
    values = weekly_seasonal_values(24 * 21)
    result = decompose(values, period=168)
    assert result.seasonal_pattern.shape == (168,)


def test_decompose_seasonal_pattern_is_centered() -> None:
    rng = np.random.default_rng(4)
    values = weekly_seasonal_values(24 * 21, rng=rng, noise_std=0.5)
    result = decompose(values, period=168)
    assert abs(float(np.mean(result.seasonal_pattern))) < 1e-6


def test_decompose_short_series_gives_zero_seasonal() -> None:
    """Fewer than two full periods: seasonal component must be all-zero."""
    values = weekly_seasonal_values(24 * 7)  # only 1 week, period=168 needs >=2
    result = decompose(values, period=168)
    assert np.all(result.seasonal == 0.0)


def test_decompose_seasonal_captures_daily_shape() -> None:
    rng = np.random.default_rng(5)
    values = daily_seasonal_values(24 * 10, rng=rng, noise_std=0.2)
    result = decompose(values, period=24)
    # Peak of the sinusoid is at hour 6 (sin argument = pi/2), trough at hour 18.
    peak_hour = int(np.argmax(result.seasonal_pattern))
    trough_hour = int(np.argmin(result.seasonal_pattern))
    assert peak_hour == 6
    assert trough_hour == 18


def test_decompose_flat_constant_series_no_divide_by_zero() -> None:
    values = np.full(24 * 21, 42.0)
    with np.errstate(all="raise"):
        result = decompose(values, period=168)
    assert np.all(result.trend == 42.0)
    assert np.all(result.seasonal == 0.0)
    assert np.all(result.remainder == 0.0)


def test_decompose_trend_is_robust_to_single_spike() -> None:
    """A single huge spike should barely move the median trend."""
    values = weekly_seasonal_values(24 * 21)
    spiked = values.copy()
    spiked[500] += 1000.0
    result = decompose(spiked, period=168)
    baseline_result = decompose(values, period=168)
    # Trend around the spike should be close to the unspiked trend.
    assert abs(result.trend[500] - baseline_result.trend[500]) < 5.0


def test_decompose_edges_do_not_produce_spurious_large_remainder() -> None:
    """Regression test: truncated trend windows at series edges must not
    leak seasonal amplitude into the remainder (see decompose.py docstring)."""
    values = weekly_seasonal_values(24 * 63)  # 9 weeks, noise-free
    result = decompose(values, period=168)
    edge = 84  # half the trend window
    assert np.max(np.abs(result.remainder[:edge])) < 1e-6
    assert np.max(np.abs(result.remainder[-edge:])) < 1e-6


def test_decompose_weekend_phases_not_systematically_offset() -> None:
    """Weekend hours should decompose to ~0 remainder just like weekdays,
    i.e. the seasonal component actually absorbs the weekend dip."""
    values = weekly_seasonal_values(24 * 21)
    result = decompose(values, period=168)
    # hours 120..168 of each week (Sat 00:00 .. Sun 24:00) are weekend
    weekend_idx = [i for i in range(len(values)) if (i // 24) % 7 in (5, 6)]
    assert np.max(np.abs(result.remainder[weekend_idx])) < 1e-6
