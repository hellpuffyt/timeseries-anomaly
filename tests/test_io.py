from __future__ import annotations

import json
from pathlib import Path

import pytest

from timeseries_anomaly.io import load_csv, load_file, load_json, parse_timestamp


def test_parse_timestamp_iso() -> None:
    dt = parse_timestamp("2026-01-05T00:00:00")
    assert dt.year == 2026 and dt.month == 1 and dt.day == 5


def test_parse_timestamp_iso_with_z_suffix() -> None:
    dt = parse_timestamp("2026-01-05T00:00:00Z")
    assert dt.year == 2026 and dt.hour == 0


def test_parse_timestamp_unix_seconds_numeric() -> None:
    dt = parse_timestamp(1767571200)
    assert dt.year == 2026


def test_parse_timestamp_unix_seconds_string() -> None:
    dt = parse_timestamp("1767571200")
    assert dt.year == 2026


def test_load_csv_single_series(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("timestamp,value\n2026-01-05T00:00:00,1.0\n2026-01-05T01:00:00,2.0\n")
    data = load_csv(p)
    assert set(data.keys()) == {"default"}
    ts, vals = data["default"]
    assert len(ts) == 2
    assert list(vals) == [1.0, 2.0]


def test_load_csv_multi_series_by_key(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text(
        "timestamp,value,key\n"
        "2026-01-05T00:00:00,1.0,cpu\n"
        "2026-01-05T00:00:00,500.0,memory\n"
        "2026-01-05T01:00:00,2.0,cpu\n"
    )
    data = load_csv(p)
    assert set(data.keys()) == {"cpu", "memory"}
    assert list(data["cpu"][1]) == [1.0, 2.0]
    assert list(data["memory"][1]) == [500.0]


def test_load_csv_missing_required_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("time,val\n1,2\n")
    with pytest.raises(ValueError):
        load_csv(p)


def test_load_csv_missing_cell_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,value\n2026-01-05T00:00:00,\n")
    with pytest.raises(ValueError):
        load_csv(p)


def test_load_csv_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError):
        load_csv(p)


def test_load_json_flat_list(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text(
        json.dumps(
            [
                {"timestamp": "2026-01-05T00:00:00", "value": 1.0},
                {"timestamp": "2026-01-05T01:00:00", "value": 2.0},
            ]
        )
    )
    data = load_json(p)
    ts, vals = data["default"]
    assert len(ts) == 2
    assert list(vals) == [1.0, 2.0]


def test_load_json_flat_list_with_keys(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text(
        json.dumps(
            [
                {"timestamp": "2026-01-05T00:00:00", "value": 1.0, "key": "cpu"},
                {"timestamp": "2026-01-05T00:00:00", "value": 500.0, "key": "memory"},
            ]
        )
    )
    data = load_json(p)
    assert set(data.keys()) == {"cpu", "memory"}


def test_load_json_mapping_form(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text(
        json.dumps(
            {
                "cpu": [{"timestamp": "2026-01-05T00:00:00", "value": 1.0}],
                "memory": [{"timestamp": "2026-01-05T00:00:00", "value": 500.0}],
            }
        )
    )
    data = load_json(p)
    assert set(data.keys()) == {"cpu", "memory"}


def test_load_json_bad_structure_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps("just a string"))
    with pytest.raises(ValueError):
        load_json(p)


def test_load_file_dispatches_by_extension(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("timestamp,value\n2026-01-05T00:00:00,1.0\n")
    json_path = tmp_path / "d.json"
    json_path.write_text(json.dumps([{"timestamp": "2026-01-05T00:00:00", "value": 1.0}]))

    assert "default" in load_file(csv_path)
    assert "default" in load_file(json_path)


def test_load_file_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        load_file(p)


def test_load_csv_sorted_by_timestamp_even_if_unordered(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text(
        "timestamp,value\n"
        "2026-01-05T02:00:00,3.0\n"
        "2026-01-05T00:00:00,1.0\n"
        "2026-01-05T01:00:00,2.0\n"
    )
    ts, vals = load_csv(p)["default"]
    assert list(vals) == [1.0, 2.0, 3.0]
