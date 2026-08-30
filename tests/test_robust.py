from __future__ import annotations

import numpy as np

from timeseries_anomaly.robust import hampel_filter, mad, median_mad, modified_zscore


def test_median_mad_simple() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med, scaled_mad = median_mad(values)
    assert med == 3.0
    # abs deviations: 2,1,0,1,2 -> median 1 -> scaled 1.4826
    assert abs(scaled_mad - 1.4826) < 1e-9


def test_median_mad_ignores_nan() -> None:
    values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0])
    med, scaled_mad = median_mad(values)
    assert med == 3.0
    assert abs(scaled_mad - 1.4826) < 1e-9


def test_median_mad_all_nan_returns_nan() -> None:
    values = np.array([np.nan, np.nan])
    med, scaled_mad = median_mad(values)
    assert np.isnan(med)
    assert np.isnan(scaled_mad)


def test_median_mad_constant_series_is_zero_not_error() -> None:
    values = np.full(10, 7.0)
    med, scaled_mad = median_mad(values)
    assert med == 7.0
    assert scaled_mad == 0.0


def test_mad_matches_median_mad_second_value() -> None:
    values = np.array([10.0, 12.0, 9.0, 11.0, 500.0])
    assert mad(values) == median_mad(values)[1]


def test_mad_not_poisoned_by_extreme_outlier() -> None:
    """The headline robustness claim: MAD stays sane where std blows up."""
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 1.0, size=200)
    poisoned = clean.copy()
    poisoned[50] = 10_000.0  # one absurd outlier

    clean_std = float(np.std(clean))
    poisoned_std = float(np.std(poisoned))
    # The single outlier massively inflates the standard deviation.
    assert poisoned_std > clean_std * 20

    clean_mad = mad(clean)
    poisoned_mad = mad(poisoned)
    # MAD barely moves.
    assert abs(poisoned_mad - clean_mad) < 0.5 * clean_mad


def test_mad_poisoned_outlier_still_detected_by_zscore() -> None:
    """Even though MAD isn't poisoned, the modified z-score still flags the outlier."""
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 1.0, size=200)
    values[50] = 10_000.0
    z = modified_zscore(values)
    assert abs(z[50]) > 100
    # And a normal point should not be flagged at a threshold of 4.
    normal_flags = np.abs(z) > 4
    normal_flags[50] = False
    assert normal_flags.sum() < 5  # only rare tail noise, not systematic


def test_modified_zscore_zero_scale_returns_zeros() -> None:
    values = np.full(20, 3.0)
    z = modified_zscore(values)
    assert np.all(z == 0.0)


def test_modified_zscore_explicit_center_scale() -> None:
    values = np.array([0.0, 2.0, 4.0])
    z = modified_zscore(values, center=2.0, scale=2.0)
    np.testing.assert_allclose(z, [-1.0, 0.0, 1.0])


def test_modified_zscore_matches_manual_computation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    med, scaled_mad = median_mad(values)
    z = modified_zscore(values)
    expected = (values - med) / scaled_mad
    np.testing.assert_allclose(z, expected)


def test_hampel_filter_flags_local_outlier() -> None:
    values = np.zeros(41)
    values[20] = 50.0
    z, flags = hampel_filter(values, window=5, n_sigmas=3.0)
    assert flags[20]
    assert not flags[10]
    assert not flags[30]


def test_hampel_filter_constant_series_no_flags_no_crash() -> None:
    values = np.full(30, 5.0)
    z, flags = hampel_filter(values, window=4, n_sigmas=3.0)
    assert not flags.any()
    assert np.all(z == 0.0)


def test_hampel_filter_ignores_nan_points() -> None:
    values = np.zeros(21)
    values[10] = np.nan
    z, flags = hampel_filter(values, window=5, n_sigmas=3.0)
    assert not flags[10]
    assert z[10] == 0.0


def test_hampel_filter_drifting_noise_level() -> None:
    """A local filter should catch an outlier even where global noise varies."""
    rng = np.random.default_rng(2)
    quiet = rng.normal(0.0, 0.1, size=50)
    noisy = rng.normal(0.0, 5.0, size=50)
    noisy[25] = 200.0  # a real outlier even against the noisy segment
    values = np.concatenate([quiet, noisy])
    _, flags = hampel_filter(values, window=10, n_sigmas=4.0)
    assert flags[75]
