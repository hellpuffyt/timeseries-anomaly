# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
