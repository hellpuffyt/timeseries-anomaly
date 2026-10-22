# Production readiness — timeseries-anomaly

Audit date: 2026-10-22. Tested commit: `244828b` (branch `audit/production-readiness`, based on `main` @ `8db0a40`).

## Scope

NumPy-only library + CLI: regular-grid resampling with gap handling, moving-median trend +
phase-median seasonal decomposition, robust (median/MAD) or Hampel scoring, level-shift detection
reported separately from spikes, CSV/JSON input, JSON report, ASCII sparkline, synthetic `generate`.

## Platforms and prerequisites

CPython ≥ 3.10, `numpy>=1.24` (tested with NumPy 2.5.4). Verified on macOS 27 (Apple M5, arm64),
CPython 3.12.14. Other OSes rely on the existing CI matrix.

## Clean-clone verification (commit `244828b`)

```bash
git clone -b audit/production-readiness <local clone> timeseries-anomaly && cd timeseries-anomaly
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install .     # README: pip install .
timeseries-anomaly generate --output demo.csv --days 70 --seed 42 --with-anomalies
timeseries-anomaly detect demo.csv --sparkline --json report.json   # exit 3; spike at the planted
#   index and level_shift at exactly the planted step (2026-02-20T16:00, ~28.8 -> ~43.3 for a +15 step)
timeseries-anomaly detect examples/weekly_seasonal.csv              # 0 anomalies (was 38)
python examples/generate_demo.py                                    # same 2 anomalies
uv pip install -e ".[dev]" && pytest && ruff check . && mypy        # 105 passed / clean / clean
python -m build --wheel                                             # wheel builds
```

## Commands and results

| Command | Baseline (`8db0a40`) | After (`244828b`) |
| --- | --- | --- |
| `pytest -q` | exit 0 — 95 passed, 0 failed, 0 skipped | exit 0 — 105 passed, 0 failed, 0 skipped |
| `ruff check .` | exit 0 | exit 0 |
| `mypy` (strict) | exit 0 | exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |
| `detect examples/weekly_seasonal.csv` | 38 anomalies (README: demonstrates the weekend guard) | 0 anomalies |

CI status — default branch: success (`8db0a40`, 2026-08-30); audit branch: pending until pushed.

## Measured precision / recall (from `tests/test_accuracy.py` only)

Generator: 70 days hourly, the weekly/daily pattern from `tests/conftest.py`, N(0,1) noise, five
planted spikes of 8–15σ (random sign), optionally one 8–15σ level step in the second half; 50 seeds.
Event-level matching; a level shift must be reported within ±3 samples of the planted step.
Command: `python -m tests.test_accuracy`.

| Scenario | threshold | Baseline P / R | After P / R |
| --- | --- | --- | --- |
| spikes only | 4.0 (default) | 0.801 / 1.000 | 0.988 / 1.000 |
| spikes only | 5.0 | — | 0.996 / 1.000 |
| spikes + level shift | 4.0 (default) | 0.644 / 0.833 | 0.993 / 1.000 |
| spikes + level shift | 5.0 | — | 1.000 / 1.000 |

(Baseline numbers measured with the same generator via an equivalent scratch script before any fix.)
False alarms on clean N(0,1) series (30 seeds, 1,680 points each): ~1.3 per series before, 0.2 after.
These numbers describe this synthetic generator only; no real production metrics were evaluated.

## Bugs found and fixed

| Severity | Bug | Regression test (`tests/test_accuracy.py`) |
| --- | --- | --- |
| High (core claim) | Noise scale = MAD of the remainder, which the phase-median fit shrinks (one point per phase *is* the median). Bundled 3-week example: 38 false anomalies; clean series ~1.3 false spikes/70 days. Now leave-one-out seasonal residuals. | `test_bundled_weekly_example_raises_no_alarms`, `test_clean_noisy_series_rarely_alarms`, precision floors |
| Medium | Level-shift onset reported 5–11 samples early (first split with a majority on the new level) | `test_level_shift_onset_is_the_step_not_half_a_window_early`, `test_noise_free_step_onset_is_exact`, `test_cli_generated_demo_shift_onset_is_exact` |
| Medium | Shift description used raw-value medians (daily cycle mixed in): step-size error up to 8.4 on a 8–15 step | `test_level_shift_description_states_the_step_size` |
| Medium | False spikes on both sides of a step (trend lags the step): 35 within ±60 samples over 50 seeds → 2 | `test_no_false_spikes_hug_a_level_shift`, `test_decomposition_still_sums_to_the_values_with_a_shift` |

## Security review

- Local files only, stdlib `csv`/`json`, no network, no `eval`/pickle.
- Secret scan of full history: no hits. `pip-audit`: no known vulnerabilities.
- Workflow permissions `contents: read`; actions tag-pinned (left as is).
- Package name `timeseries-anomaly` on PyPI belongs to an unrelated project; README already
  installed from source and now says so explicitly.

## Known limitations

- One seasonal period at a time; Hampel mode does not get the leave-one-out scale correction.
- Moving-median trend is O(n·period) in pure Python loops; no row cap (documented in README).
- Accuracy figures are synthetic; real-world performance is not verified.

## Acceptance checklist

| # | Item | Expected | Command / test | Result |
| --- | --- | --- | --- | --- |
| 1 | trend + seasonal + remainder == values | exact (≤1e-9) | existing decompose tests + `test_decomposition_still_sums...` | PASS |
| 2 | Weekend dips not flagged | 0 anomalies on clean weekly series / example | existing tests + example run | PASS |
| 3 | Planted spikes found | recall 1.0 | `test_spike_recall_and_precision_floor` | PASS |
| 4 | Level shift reported once, at the step | ±3 samples | onset tests | PASS |
| 5 | Few false alarms at default | precision ≥ 0.95 | precision floors | PASS (0.988/0.993) |
| 6 | Gaps interpolated / flagged, not zero | as documented | existing gap tests | PASS |
| 7 | CLI exit 3 on anomalies | 3 | clean clone | PASS |
| 8 | Clean clone | install, tests, examples, wheel | section above | PASS |

## Classification proposal

**Production-Ready for Its Documented Scope** (proposal), with real-metric accuracy unverified.
