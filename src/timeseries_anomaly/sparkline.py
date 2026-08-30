"""ASCII sparklines: a quick terminal-native view of a series and its anomalies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_TICKS = " .-:=+*#%@"


def sparkline(values: FloatArray, width: int | None = None) -> str:
    """Render ``values`` as a one-line unicode sparkline.

    NaNs are rendered as a space (gap). If ``width`` is given and shorter
    than the series, the series is downsampled by simple block-averaging.
    """
    v = values.astype(np.float64)
    if width is not None and width > 0 and v.size > width:
        v = _downsample(v, width)

    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return " " * v.size
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo

    chars = []
    for x in v:
        if not np.isfinite(x):
            chars.append(" ")
            continue
        if span == 0:
            level = len(_TICKS) // 2
        else:
            level = int(round((x - lo) / span * (len(_TICKS) - 1)))
            level = max(0, min(len(_TICKS) - 1, level))
        chars.append(_TICKS[level])
    return "".join(chars)


def _downsample(values: FloatArray, width: int) -> FloatArray:
    n = values.size
    bounds = np.linspace(0, n, width + 1).astype(int)
    out = np.full(width, np.nan, dtype=np.float64)
    for i in range(width):
        lo, hi = bounds[i], max(bounds[i] + 1, bounds[i + 1])
        segment = values[lo:hi]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            out[i] = float(np.mean(finite))
    return out


def annotated_sparkline(
    values: FloatArray, anomaly_mask: FloatArray, width: int | None = None
) -> tuple[str, str]:
    """Return (sparkline, marker line) where marker line has '^' under anomalies.

    Downsampling is not applied here so that markers stay aligned 1:1
    with the underlying samples; callers with very long series should
    chunk the output (see the CLI) instead of shrinking it.
    """
    line = sparkline(values, width=None)
    markers = "".join("^" if bool(m) else " " for m in anomaly_mask)
    return line, markers
