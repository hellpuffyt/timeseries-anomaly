# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/timeseries-anomaly/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

timeseries-anomaly reads local CSV/JSON files and performs numeric computation; it makes no network calls and uses `json`/`csv` from the standard library only. Relevant reports include: input that executes code, writes outside the requested `--output`/`--json` path, or causes unbounded resource use beyond what the input size implies.
