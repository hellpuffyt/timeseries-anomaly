"""Robust statistics: median, MAD, modified z-scores, and a Hampel filter.

These are the building blocks that keep the rest of the package free of a
scipy/statsmodels dependency, and free of the classic failure mode of
mean/standard-deviation thresholds: a single huge outlier inflates the
standard deviation enough to hide itself (and every smaller anomaly along
with it). Median and MAD barely move when a minority of points are extreme.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Scale factor that makes MAD a consistent estimator of the standard
# deviation for normally distributed data: 1 / Phi^-1(3/4).
MAD_SCALE = 1.4826

FloatArray = npt.NDArray[np.float64]


def median_mad(values: FloatArray) -> tuple[float, float]:
    """Return (median, scaled MAD) of ``values``, ignoring NaNs.

    The returned MAD is already multiplied by :data:`MAD_SCALE`, so it is
    directly comparable to a standard deviation for roughly-normal data.
    """
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(finite))
    deviations = np.abs(finite - med)
    mad_raw = float(np.median(deviations))
    return med, mad_raw * MAD_SCALE


def mad(values: FloatArray) -> float:
    """Scaled median absolute deviation of ``values`` (NaNs ignored)."""
    return median_mad(values)[1]


def modified_zscore(
    values: FloatArray, center: float | None = None, scale: float | None = None
) -> FloatArray:
    """Iglewicz-Hoaglin style modified z-score using median and scaled MAD.

    If ``center``/``scale`` are not supplied they are computed from
    ``values`` itself. When the scale is zero (the bulk of the data is
    identical, e.g. a flat series or a noise-free remainder) there is no
    dispersion to divide by. A point that also equals ``center`` (within
    a tolerance tied to the data's own magnitude, to absorb floating
    point noise rather than exact-equality comparisons) gets z=0, same
    as always. But a point that genuinely differs from an otherwise
    constant baseline is a hard, unambiguous outlier -- reporting z=0
    for it (as a naive "no dispersion, so nothing can be an outlier"
    rule would) is wrong, so it gets a large sentinel z-score instead.
    """
    if center is None or scale is None:
        med, scaled_mad = median_mad(values)
        center = med if center is None else center
        scale = scaled_mad if scale is None else scale

    finite = values[np.isfinite(values)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 0.0
    atol = 1e-9 * (magnitude + 1.0)
    # Anything below the tolerance is treated as "no real dispersion",
    # not just an exact 0.0 -- a scale that rounds to, say, 1e-17 while
    # the data itself is tiny floating point noise is functionally zero,
    # and dividing by it would blow negligible deviations up into
    # enormous, meaningless z-scores.
    if scale is None or np.isnan(scale) or scale <= atol:
        deviation = values - center
        return np.where(np.abs(deviation) <= atol, 0.0, np.sign(deviation) * 1.0e6)
    out = (values - center) / scale
    return out


def hampel_filter(
    values: FloatArray, window: int = 7, n_sigmas: float = 3.0
) -> tuple[FloatArray, FloatArray]:
    """Rolling-window Hampel identifier.

    For every index, computes the median and scaled MAD of a centred
    window of ``2 * window + 1`` samples (clipped at the series edges) and
    flags the point as an outlier when it deviates from that local median
    by more than ``n_sigmas`` scaled MADs. This is a local alternative to
    the global robust threshold, useful when the "normal" spread itself
    drifts over the series.

    Returns ``(z_scores, is_outlier)``.
    """
    n = values.size
    z = np.zeros(n, dtype=np.float64)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = values[lo:hi]
        med, scaled_mad = median_mad(local)
        if np.isnan(values[i]) or np.isnan(med):
            continue
        atol = 1e-9 * (abs(med) + 1.0)
        if scaled_mad <= atol:
            # No meaningful local spread: the window is (locally) constant,
            # so any deviation from it is a hard outlier by definition, not
            # a divide-by-zero to paper over with a zero z-score. Compare
            # with a tolerance tied to the local magnitude to absorb
            # floating point noise rather than exact bitwise equality.
            deviation = values[i] - med
            if abs(deviation) <= atol:
                z[i] = 0.0
                flags[i] = False
            else:
                z[i] = np.sign(deviation) * 1e6
                flags[i] = True
            continue
        zi = (values[i] - med) / scaled_mad
        z[i] = zi
        flags[i] = bool(abs(zi) > n_sigmas)
    return z, flags
