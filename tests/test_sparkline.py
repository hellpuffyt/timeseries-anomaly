from __future__ import annotations

import numpy as np

from timeseries_anomaly.sparkline import annotated_sparkline, sparkline


def test_sparkline_length_matches_input() -> None:
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    line = sparkline(values)
    assert len(line) == len(values)


def test_sparkline_is_ascii_only() -> None:
    values = np.linspace(0, 100, 50)
    line = sparkline(values)
    assert all(ord(c) < 128 for c in line)


def test_sparkline_monotonic_increase_is_monotonic_ticks() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    line = sparkline(values)
    from timeseries_anomaly.sparkline import _TICKS

    levels = [_TICKS.index(c) for c in line]
    assert levels == sorted(levels)


def test_sparkline_constant_series_uses_mid_level() -> None:
    values = np.full(10, 5.0)
    line = sparkline(values)
    from timeseries_anomaly.sparkline import _TICKS

    mid = _TICKS[len(_TICKS) // 2]
    assert all(c == mid for c in line)


def test_sparkline_all_nan_is_blank() -> None:
    values = np.full(5, np.nan)
    line = sparkline(values)
    assert line == "     "


def test_sparkline_nan_renders_as_space_gap() -> None:
    values = np.array([1.0, np.nan, 3.0])
    line = sparkline(values)
    assert line[1] == " "


def test_sparkline_downsamples_when_width_smaller() -> None:
    values = np.arange(1000, dtype=np.float64)
    line = sparkline(values, width=50)
    assert len(line) == 50


def test_sparkline_no_downsample_when_width_larger_than_series() -> None:
    values = np.arange(10, dtype=np.float64)
    line = sparkline(values, width=50)
    assert len(line) == 10


def test_annotated_sparkline_marker_alignment() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([False, False, True, False, True])
    line, markers = annotated_sparkline(values, mask)
    assert len(line) == len(markers) == 5
    assert markers[2] == "^"
    assert markers[4] == "^"
    assert markers[0] == " "
