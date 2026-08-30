"""Reading (timestamp, value[, key]) series from CSV or JSON files."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_KEY = "default"


def parse_timestamp(raw: str | int | float) -> datetime:
    """Parse an ISO-8601 string, or a unix timestamp (seconds), into a naive UTC datetime."""
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).replace(tzinfo=None)
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fall back to treating it as a numeric unix timestamp string.
        return datetime.fromtimestamp(float(text), tz=timezone.utc).replace(tzinfo=None)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def load_csv(path: str | Path) -> dict[str, tuple[list[datetime], FloatArray]]:
    """Load a CSV with columns ``timestamp``, ``value`` and an optional ``key``."""
    series: dict[str, list[tuple[datetime, float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV file")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "timestamp" not in fields or "value" not in fields:
            raise ValueError(f"{path}: CSV must have 'timestamp' and 'value' columns")
        ts_col = fields["timestamp"]
        val_col = fields["value"]
        key_col = fields.get("key")
        for row_num, row in enumerate(reader, start=2):
            raw_ts = row.get(ts_col)
            raw_val = row.get(val_col)
            if raw_ts is None or raw_ts == "" or raw_val is None or raw_val == "":
                raise ValueError(f"{path}: row {row_num} missing timestamp or value")
            key = (row.get(key_col) or DEFAULT_KEY) if key_col else DEFAULT_KEY
            ts = parse_timestamp(raw_ts)
            val = float(raw_val)
            series.setdefault(key, []).append((ts, val))
    return _finalize(series)


def load_json(path: str | Path) -> dict[str, tuple[list[datetime], FloatArray]]:
    """Load JSON records: a flat list, or a mapping of key -> list of records.

    Each record is an object with ``timestamp`` and ``value`` (and
    optionally ``key``, when using the flat-list form).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    series: dict[str, list[tuple[datetime, float]]] = {}

    if isinstance(data, list):
        for rec in data:
            key = str(rec.get("key", DEFAULT_KEY))
            ts = parse_timestamp(rec["timestamp"])
            val = float(rec["value"])
            series.setdefault(key, []).append((ts, val))
    elif isinstance(data, dict):
        for key, records in data.items():
            for rec in records:
                ts = parse_timestamp(rec["timestamp"])
                val = float(rec["value"])
                series.setdefault(str(key), []).append((ts, val))
    else:
        raise ValueError(f"{path}: unrecognized JSON structure")

    return _finalize(series)


def _finalize(
    series: dict[str, list[tuple[datetime, float]]]
) -> dict[str, tuple[list[datetime], FloatArray]]:
    if not series:
        raise ValueError("no data rows found")
    out: dict[str, tuple[list[datetime], FloatArray]] = {}
    for key, points in series.items():
        points.sort(key=lambda p: p[0])
        timestamps = [p[0] for p in points]
        values = np.array([p[1] for p in points], dtype=np.float64)
        out[key] = (timestamps, values)
    return out


def load_file(path: str | Path) -> dict[str, tuple[list[datetime], FloatArray]]:
    """Load a series file, dispatching on extension (.csv or .json)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_csv(p)
    if suffix == ".json":
        return load_json(p)
    raise ValueError(f"unsupported file extension: {suffix} (expected .csv or .json)")
