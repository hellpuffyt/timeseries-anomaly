from __future__ import annotations

import json
from pathlib import Path

import pytest

from timeseries_anomaly.cli import main


def test_generate_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    code = main(["generate", "--output", str(out), "--days", "7", "--seed", "1"])
    assert code == 0
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "timestamp,value"
    assert len(lines) == 7 * 24 + 1


def test_generate_is_deterministic_with_seed(tmp_path: Path) -> None:
    out1 = tmp_path / "a.csv"
    out2 = tmp_path / "b.csv"
    main(["generate", "--output", str(out1), "--days", "7", "--seed", "5"])
    main(["generate", "--output", str(out2), "--days", "7", "--seed", "5"])
    assert out1.read_text() == out2.read_text()


def test_generate_different_seeds_differ(tmp_path: Path) -> None:
    out1 = tmp_path / "a.csv"
    out2 = tmp_path / "b.csv"
    main(["generate", "--output", str(out1), "--days", "7", "--seed", "1"])
    main(["generate", "--output", str(out2), "--days", "7", "--seed", "2"])
    assert out1.read_text() != out2.read_text()


def test_detect_exit_code_zero_when_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "clean.csv"
    main(["generate", "--output", str(out), "--days", "70", "--seed", "3", "--noise", "0"])
    code = main(["detect", str(out), "--period", "168", "--quiet"])
    assert code == 0


def test_detect_exit_code_nonzero_with_anomalies(tmp_path: Path) -> None:
    out = tmp_path / "anom.csv"
    main(
        ["generate", "--output", str(out), "--days", "70", "--seed", "3", "--with-anomalies", "--noise", "0"]
    )
    code = main(["detect", str(out), "--period", "168", "--quiet"])
    assert code == 3


def test_detect_writes_json_report(tmp_path: Path) -> None:
    out = tmp_path / "anom.csv"
    report = tmp_path / "report.json"
    main(
        ["generate", "--output", str(out), "--days", "70", "--seed", "3", "--with-anomalies", "--noise", "0"]
    )
    main(["detect", str(out), "--period", "168", "--quiet", "--json", str(report)])
    data = json.loads(report.read_text())
    assert data["anomaly_count"] >= 1
    assert "anomalies" in data


def test_detect_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "demo.csv"
    main(["generate", "--output", str(out), "--days", "70", "--seed", "3", "--noise", "0"])
    capsys.readouterr()
    main(["detect", str(out), "--period", "168"])
    captured = capsys.readouterr()
    assert "points:" in captured.out
    assert "anomalies found:" in captured.out


def test_detect_sparkline_flag_prints_ascii_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "demo.csv"
    main(["generate", "--output", str(out), "--days", "10", "--seed", "3"])
    capsys.readouterr()
    main(["detect", str(out), "--period", "24", "--sparkline"])
    captured = capsys.readouterr()
    assert all(ord(c) < 128 for c in captured.out)


def test_main_no_command_prints_help_and_returns_nonzero() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_detect_on_multi_series_json(tmp_path: Path) -> None:
    data = {
        "cpu": [
            {"timestamp": f"2026-01-05T{h:02d}:00:00", "value": 10.0 + (h % 5)} for h in range(24)
        ]
        * 3,
        "memory": [
            {"timestamp": f"2026-01-05T{h:02d}:00:00", "value": 500.0} for h in range(24)
        ]
        * 3,
    }
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(data))
    code = main(["detect", str(p), "--quiet"])
    assert code in (0, 3)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("missing.csv", None),
        ("bad_timestamp.csv", b"timestamp,value\nnot-a-date,1\n"),
        ("bad_columns.csv", b"a,b\n1,2\n"),
        ("bad.json", b"{bad"),
        ("not_utf8.csv", b"\xff\xfe"),
        ("series.txt", b"timestamp,value\n"),
    ],
)
def test_detect_invalid_input_is_a_one_line_error_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, content: bytes | None
) -> None:
    # Before: each of these ended in an uncaught Python traceback.
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    code = main(["detect", str(path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert len(captured.err.strip().splitlines()) == 1
    assert "Traceback" not in captured.err


def test_detect_help_describes_the_real_period_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    # Before: the help said the period fell back to 0; the code uses min(24, n/4), at least 2.
    with pytest.raises(SystemExit):
        main(["detect", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "else 0" not in help_text
    assert "min(24, n/4)" in help_text
