# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Level-shift onsets were reported 5-11 samples before the real step (the
  first split where just over half the "after" window had moved). The onset is
  now localised within the median-jump plateau by the largest clipped-mean jump.
- The level-shift description took before/after medians of raw values, mixing
  in the daily cycle (a 15-unit step could read as 5.5); it now uses the
  deseasonalized series.
- The robust threshold divided by the MAD of the remainder, which the
  phase-median seasonal fit shrinks towards zero (one point per phase *is*
  the median). Clean noisy series got ~1.3 false spikes per 70 days, and the
  bundled `examples/weekly_seasonal.csv` (three weeks) reported 38 anomalies.
  The noise scale now comes from leave-one-out seasonal residuals.
- False spikes clustered on both sides of a level shift because the trend lags
  the step (35 within 60 samples over 50 seeds, now 2). Detected steps are
  modelled in the trend before spikes are scored.

### Added

- `tests/test_accuracy.py`: measured precision/recall on planted anomalies, and
  regression tests for the fixes above. README "Measured accuracy" section and a
  documented false-positive rate. `SECURITY.md`, `.editorconfig`,
  `docs/PRODUCTION_READINESS.md`.

## [1.0.0] - 2026-08-30

### Added

- Robust STL-style decomposition (centred moving-median trend,
  period-averaged seasonal component, remainder) implemented from numpy
  primitives.
- Median/MAD and Hampel-filter robust anomaly thresholds.
- Seasonal-aware (hour-of-week) baselines to suppress weekend/off-hours
  false positives.
- Level-shift detection, reported distinctly from spike anomalies.
- Explicit short-gap interpolation and long-gap flagging.
- `--min-consecutive` noise suppression for single-sample spikes.
- CSV/JSON loading, including multi-series files via a `key` field.
- `detect` and `generate` CLI subcommands, JSON report output, and an
  ASCII terminal sparkline view.
