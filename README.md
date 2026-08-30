# timeseries-anomaly

Detect anomalies in metric time series using seasonal decomposition and
robust statistics, tuned specifically to stop suppressing legitimate
weekend and off-hours behaviour.

## What

A small, dependency-light (numpy only) library and CLI that:

1. Decomposes an hourly metric series into trend, seasonal, and remainder
   components.
2. Scores the remainder with robust (median/MAD or Hampel) thresholds
   instead of mean/standard-deviation ones.
3. Tells apart short spikes from sustained level shifts.
4. Handles missing samples explicitly instead of pretending a gap is a
   zero.

## Why

Threshold alerts fire every Saturday because traffic legitimately halves,
and every deploy window because latency legitimately spikes for a minute.
Teams respond by raising the threshold until the alert catches nothing
useful at all. The actual fix is to model the *expected* shape of the
metric -- its daily and weekly rhythm -- and alert on deviation from
*that*, using statistics that a handful of extreme points cannot poison.

This package implements that pipeline from first principles with numpy,
on purpose: no `statsmodels`, no `scipy`. That keeps the install light
and, more importantly, keeps every step of the algorithm inspectable and
unit-testable rather than hidden behind a library's `STL()` call.

## How the decomposition works

Given a value `y[t]` on a regular grid:

1. **Trend** -- a centred moving *median* over one seasonal period
   (window = period, rounded up to odd). A median does not get dragged
   by a handful of extreme points the way a moving average would; those
   points get smeared into the trend, silently absorbing part of the
   very anomaly you are trying to detect.
2. **Seasonal** -- after subtracting the trend, group the residual by
   its position within the period (hour-of-day for `period=24`,
   hour-of-week for `period=168` on hourly data) and take the *median*
   of each group. This produces one repeating seasonal pattern, centred
   so it doesn't leak a mean shift into the trend. This step is the
   headline feature: a Saturday 3am reading is compared against every
   other Saturday 3am in the series, not against a Tuesday-afternoon
   average.
3. **Remainder** -- `y[t] - trend[t] - seasonal[t]`. What's left is
   scored for anomalies.

### Robust thresholds

The remainder is scored with a **modified z-score**:

```
z[t] = (remainder[t] - median(remainder)) / (1.4826 * MAD(remainder))
```

`MAD` is the median absolute deviation; `1.4826` scales it so it behaves
like a standard deviation under a roughly-normal distribution. Median and
MAD barely move when a minority of points are extreme, which is exactly
the case a mean/stddev threshold gets wrong: one huge outlier inflates
the standard deviation enough to make itself (and every smaller anomaly)
invisible.

An alternative **Hampel filter** (`--method hampel`) computes the same
modified z-score in a rolling local window instead of once globally,
useful when the "normal" noise level itself drifts over the series.

### Level shifts vs. spikes

A sustained step change (a config change that permanently doubles error
rate, say) is a fundamentally different event from a transient spike,
and reporting it as "60 consecutive anomalies" buries the one fact an
operator needs (the baseline moved) under noise. `detect_level_shifts`
compares the robust centre of the window before a candidate point to the
window after it; when the jump is large relative to the local spread
*and* the new level is itself stable, it's reported once, as a single
`level_shift` event, and the individual points inside it are not also
reported as spikes.

### Missing data

Raw timestamps are resampled onto a regular grid at the inferred (or
supplied) sampling frequency. Gaps up to `--max-gap-samples` (default 3)
are linearly interpolated and marked `interpolated`. Longer gaps are
left as `NaN`, excluded from decomposition and thresholding, and
reported separately as `long_gap` ranges -- never silently treated as
zero, which would itself look like a giant anomaly (or mask a real one).

## Features

- Robust STL-style decomposition (trend / seasonal / remainder) with
  configurable period (daily = 24, weekly = 168, or custom).
- Median/MAD and Hampel-filter robust thresholding.
- Seasonal-aware baselines (hour-of-week comparison) to suppress
  weekend/off-hours false positives.
- Level-shift detection, reported distinctly from spikes.
- Explicit gap handling: short-gap interpolation, long-gap flagging.
- `--min-consecutive` to suppress single-sample noise.
- CSV and JSON input, with optional multiple series via a `key` column
  / field.
- Human-readable summary, machine-readable JSON report, and an ASCII
  terminal sparkline with anomalies marked.
- A `generate` subcommand for producing deterministic synthetic series
  (`--seed`) for demos and tests.

## Architecture

```
src/timeseries_anomaly/
  series.py      regular-grid resampling + gap interpolation/flagging
  robust.py      median/MAD, modified z-score, Hampel filter
  decompose.py   trend / seasonal / remainder decomposition
  detect.py      point-anomaly scoring, level-shift detection, severity
  io.py          CSV/JSON loading (single or multi-series)
  sparkline.py   ASCII sparkline rendering
  cli.py         `detect` and `generate` subcommands
```

## Installation

```bash
pip install .
# or, for development:
pip install -e ".[dev]"
```

Requires Python 3.10+. Runtime dependency: `numpy>=1.24`.

## Usage

```bash
# Generate a deterministic demo series (70 days hourly, weekly seasonality)
timeseries-anomaly generate --output demo.csv --days 70 --seed 42 --with-anomalies

# Detect anomalies
timeseries-anomaly detect demo.csv --sparkline --json report.json
```

Exit code is `0` when no anomalies are found, `3` when at least one is
reported (handy for alerting pipelines / CI smoke checks).

### CLI options (`detect`)

| Flag | Default | Meaning |
|---|---|---|
| `--period` | auto (168/24/adaptive) | samples per seasonal cycle |
| `--method` | `robust` | `robust` (global MAD) or `hampel` (rolling local MAD) |
| `--threshold` | `4.0` | modified z-score magnitude to flag |
| `--min-consecutive` | `1` | suppress spike runs shorter than this |
| `--shift-window` / `--shift-threshold` | `24` / `5.0` | level-shift detector window and sensitivity |
| `--max-gap-samples` | `3` | longest run of missing samples to interpolate |
| `--freq-seconds` | inferred | override the sampling frequency |
| `--json PATH` | - | write a detailed JSON report |
| `--sparkline` | off | print an ASCII sparkline with anomalies marked |

## Input formats

**CSV** (a `key` column is optional; omit it for a single series):

```csv
timestamp,value
2026-01-05T00:00:00,21.3
2026-01-05T01:00:00,19.8
```

```csv
timestamp,value,key
2026-01-05T00:00:00,21.3,cpu
2026-01-05T00:00:00,512.0,memory
```

**JSON**, either a flat list of records (optionally with `key`):

```json
[
  {"timestamp": "2026-01-05T00:00:00", "value": 21.3},
  {"timestamp": "2026-01-05T01:00:00", "value": 19.8}
]
```

or a mapping of key to its records:

```json
{
  "cpu": [{"timestamp": "2026-01-05T00:00:00", "value": 21.3}],
  "memory": [{"timestamp": "2026-01-05T00:00:00", "value": 512.0}]
}
```

Timestamps accept ISO-8601 (`Z` or `+00:00` offsets normalized to UTC) or
a unix epoch (seconds).

## Tuning

- **Too many alerts on noisy-but-normal series**: raise `--threshold`,
  or raise `--min-consecutive` to require a sustained run before firing.
- **Missing genuine short spikes**: lower `--threshold`, or check that
  `--period` matches your data's actual seasonality (a wrong period
  leaves real seasonal swings in the remainder, inflating MAD and
  hiding smaller anomalies).
- **Noise level drifts over time** (e.g. quiet at night, noisy at peak
  traffic): try `--method hampel`, which re-estimates the local spread
  in a rolling window instead of once globally.
- **A step change reported as a spike run instead of a level shift**:
  lower `--shift-threshold` or shorten `--shift-window` to make the
  detector more sensitive to smaller/faster steps.

## Examples

See `examples/`:

- `examples/generate_demo.py` -- builds and analyzes a synthetic series
  with a planted spike and level shift end to end.
- `examples/weekly_seasonal.csv` -- a small hand-built series
  demonstrating the weekend false-positive guard.

## Testing

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

The test suite is built entirely on synthetic series where the ground
truth is known: planted spikes, planted level shifts, clean seasonal
series that must produce zero anomalies (including at weekends -- the
headline guarantee), poisoned-outlier scenarios that would fool a
mean/stddev threshold but not MAD, short gaps that must be interpolated,
long gaps that must be flagged and excluded, a flat constant series that
must not divide by zero, and round-trip reconstruction checks
(`trend + seasonal + remainder == value`).

## Limitations

- The decomposition assumes one dominant seasonal period at a time; a
  series with strong daily *and* weekly seasonality will only have the
  period you select modeled explicitly (the other leaks into trend/
  remainder). Run the tool twice with different periods if you need
  both views.
- The period-averaged seasonal component needs at least two full cycles
  of data to be estimated; shorter series get an all-zero seasonal
  component (equivalent to plain robust thresholding on the trend
  residual).
- Level-shift detection uses a fixed-width before/after window and will
  miss shifts smaller than `--shift-window` samples from the start or
  end of the series, or shifts that ramp in gradually rather than
  stepping.
- This is univariate, single-metric analysis; it does not correlate
  anomalies across multiple series or metrics.

## Security

This tool only reads local CSV/JSON files supplied on the command line
and performs local numeric computation; it makes no network calls. When
loading JSON, standard library `json.load` is used (no arbitrary code
execution). If you feed it untrusted files, the usual caution around
resource limits for very large inputs still applies -- there is no
built-in row-count cap.

## License

MIT. See [LICENSE](LICENSE).
