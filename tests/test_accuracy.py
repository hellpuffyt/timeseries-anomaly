"""Measured detection accuracy on synthetic series with planted ground truth.

Each series is 70 days of hourly data with the weekly/daily pattern from
``conftest.weekly_seasonal_values`` plus N(0, 1) noise, five planted spikes
of 8-15 sigma (random sign), and optionally one level step of 8-15 sigma
somewhere in the second half. Matching is event-level: a reported spike
matches a planted spike it covers (+-1 sample); a reported level shift
matches when its onset is within +-3 samples of the planted step. Every
other report is a false positive.

Run ``python -m tests.test_accuracy`` to print the numbers quoted in
docs/PRODUCTION_READINESS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.conftest import hourly_timestamps, weekly_seasonal_values
from timeseries_anomaly.detect import detect_anomalies, detect_level_shifts
from timeseries_anomaly.series import build_series

N = 70 * 24
SHIFT_TOLERANCE = 3


@dataclass
class Planted:
    values: np.ndarray
    spikes: list[int]
    shift_at: int | None
    shift_size: float


def planted_series(seed: int, n_spikes: int = 5, with_shift: bool = True) -> Planted:
    rng = np.random.default_rng(seed)
    values = weekly_seasonal_values(N, rng=rng, noise_std=1.0)
    shift_at = int(rng.integers(N // 2, N - 200)) if with_shift else None
    spikes: list[int] = []
    while len(spikes) < n_spikes:
        i = int(rng.integers(30, N - 30))
        if shift_at is not None and abs(i - shift_at) < 48:
            continue
        if any(abs(i - s) < 10 for s in spikes):
            continue
        spikes.append(i)
    for i in spikes:
        values[i] += rng.choice([-1.0, 1.0]) * rng.uniform(8, 15)
    shift_size = 0.0
    if shift_at is not None:
        shift_size = float(rng.choice([-1.0, 1.0]) * rng.uniform(8, 15))
        values[shift_at:] += shift_size
    return Planted(values, spikes, shift_at, shift_size)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0


def score(seeds: range, with_shift: bool, threshold: float = 4.0) -> Score:
    total = Score()
    timestamps = hourly_timestamps(N)
    for seed in seeds:
        planted = planted_series(seed, with_shift=with_shift)
        result = detect_anomalies(build_series(timestamps, planted.values), period=168, threshold=threshold)
        matched: set[tuple[str, int]] = set()
        for anomaly in result.anomalies:
            hit: tuple[str, int] | None = None
            if anomaly.type == "spike":
                for s in planted.spikes:
                    if anomaly.start_index - 1 <= s <= anomaly.end_index + 1:
                        hit = ("spike", s)
            elif (
                planted.shift_at is not None
                and abs(anomaly.start_index - planted.shift_at) <= SHIFT_TOLERANCE
            ):
                hit = ("shift", planted.shift_at)
            if hit is not None and hit not in matched:
                matched.add(hit)
                total.tp += 1
            else:
                total.fp += 1
        expected = len(planted.spikes) + (planted.shift_at is not None)
        total.fn += expected - len(matched)
    return total


def test_spike_recall_and_precision_floor() -> None:
    result = score(range(20), with_shift=False)
    assert result.recall == 1.0
    assert result.precision >= 0.95


def test_level_shift_series_precision_and_recall_floor() -> None:
    result = score(range(20), with_shift=True)
    assert result.recall >= 0.98
    assert result.precision >= 0.95


def test_level_shift_onset_is_the_step_not_half_a_window_early() -> None:
    # The detector used to report the onset as soon as just over half of
    # its "after" window had moved: 5-11 samples before the real step.
    timestamps = hourly_timestamps(N)
    errors = []
    for seed in range(20):
        planted = planted_series(seed, n_spikes=0)
        result = detect_anomalies(build_series(timestamps, planted.values), period=168)
        shifts = [a for a in result.anomalies if a.type == "level_shift"]
        assert shifts, f"seed {seed}: planted step not reported"
        assert planted.shift_at is not None
        errors.append(abs(shifts[0].start_index - planted.shift_at))
    assert sum(e <= SHIFT_TOLERANCE for e in errors) >= 19, errors


def test_noise_free_step_onset_is_exact() -> None:
    values = np.zeros(200)
    values[120:] = 10.0
    shifts = detect_level_shifts(values, np.ones(200, dtype=bool), window=24, threshold=3.0)
    assert [onset for onset, _ in shifts] == [120]


def test_level_shift_description_states_the_step_size() -> None:
    # The before/after levels were medians of raw values, which mix in the
    # daily cycle: a +15 step was described as roughly +5.5.
    # Measured error before the fix on seeds 0-11: up to 8.4; after: <= 0.26.
    timestamps = hourly_timestamps(N)
    for seed in range(12):
        planted = planted_series(seed, n_spikes=0)
        result = detect_anomalies(build_series(timestamps, planted.values), period=168)
        [shift] = [a for a in result.anomalies if a.type == "level_shift"]
        before, after = (float(part.split("~")[1]) for part in shift.description.split(" to "))
        assert abs((after - before) - planted.shift_size) < 0.5, seed


def test_no_false_spikes_hug_a_level_shift() -> None:
    # The moving-median trend lags a step, so the remainder on both sides
    # was biased: 35 false spikes within 60 samples of the step over 50
    # seeds. Steps are now modelled in the trend before scoring.
    timestamps = hourly_timestamps(N)
    near = 0
    for seed in range(20):
        planted = planted_series(seed)
        assert planted.shift_at is not None
        result = detect_anomalies(build_series(timestamps, planted.values), period=168)
        for a in result.anomalies:
            covers_planted = any(a.start_index - 1 <= s <= a.end_index + 1 for s in planted.spikes)
            if a.type == "spike" and not covers_planted and abs(a.start_index - planted.shift_at) <= 60:
                near += 1
    assert near <= 2


def test_decomposition_still_sums_to_the_values_with_a_shift() -> None:
    timestamps = hourly_timestamps(N)
    planted = planted_series(1)
    result = detect_anomalies(build_series(timestamps, planted.values), period=168)
    d = result.decomposition
    np.testing.assert_allclose(d.trend + d.seasonal + d.remainder, planted.values, atol=1e-9)


def test_clean_noisy_series_rarely_alarms() -> None:
    # The phase medians pulled the remainder towards zero, so its MAD
    # underestimated the noise: ~1.3 false spikes per clean 70-day series
    # at the default threshold. The scale now comes from leave-one-out
    # seasonal residuals (measured 0.2 per series over 30 seeds).
    timestamps = hourly_timestamps(N)
    alarms = 0
    for seed in range(20):
        values = weekly_seasonal_values(N, rng=np.random.default_rng(1000 + seed), noise_std=1.0)
        alarms += len(detect_anomalies(build_series(timestamps, values), period=168).anomalies)
    assert alarms <= 8


def test_bundled_weekly_example_raises_no_alarms() -> None:
    # examples/weekly_seasonal.csv is documented as demonstrating the
    # weekend false-positive guard; with three weeks of data (three points
    # per phase) it used to report 38 anomalies.
    from pathlib import Path

    from timeseries_anomaly.io import load_file
    from timeseries_anomaly.series import build_series as build

    path = Path(__file__).resolve().parent.parent / "examples" / "weekly_seasonal.csv"
    loaded = load_file(str(path))
    [(timestamps, values)] = [(ts, vs) for ts, vs in loaded.values()]
    result = detect_anomalies(build(timestamps, values), period=168)
    assert result.anomalies == []


def test_cli_generated_demo_shift_onset_is_exact() -> None:
    # `timeseries-anomaly generate --with-anomalies` plants the step at 2n/3.
    timestamps = hourly_timestamps(N)
    values = weekly_seasonal_values(N, rng=np.random.default_rng(42), noise_std=1.0)
    values[(2 * N) // 3 :] += 15.0
    result = detect_anomalies(build_series(timestamps, values), period=168)
    [shift] = [a for a in result.anomalies if a.type == "level_shift"]
    assert shift.start_index == (2 * N) // 3


if __name__ == "__main__":  # pragma: no cover - reporting helper
    for label, with_shift in (("spikes only", False), ("spikes + one level shift", True)):
        for threshold in (4.0, 5.0):
            s = score(range(50), with_shift=with_shift, threshold=threshold)
            print(
                f"{label:26s} threshold={threshold}: precision={s.precision:.3f} "
                f"recall={s.recall:.3f} (tp={s.tp} fp={s.fp} fn={s.fn}, 50 series)"
            )
