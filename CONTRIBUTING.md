# Contributing

Contributions are welcome. This project intentionally keeps its runtime
dependency footprint to `numpy` only -- please don't add `scipy` or
`statsmodels` as a dependency; the point of the project is a readable,
testable, from-scratch implementation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest
ruff check .
mypy
python -m build --wheel   # confirm packaging still works
```

All three must pass. New behaviour should come with tests that assert
the actual expected numbers/flags on a synthetic series where you know
the ground truth, not just "it doesn't crash".

## Style

- Type hints everywhere; `mypy --strict` is a hard gate.
- Keep functions small and testable in isolation -- prefer adding a new
  pure function in `robust.py`/`decompose.py`/`detect.py` over growing
  an existing one.
- Docstrings should explain *why* a design choice was made (e.g. why
  median over mean) when it isn't obvious, not just restate the
  signature.

## Reporting bugs

Please include a minimal CSV/JSON reproduction and the exact CLI flags
used -- this is a numerical library, and "it flagged the wrong thing" is
much easier to fix with the actual data than a description of it.
