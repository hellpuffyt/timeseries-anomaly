"""timeseries-anomaly: seasonal-aware anomaly detection for metric time series.

Detects anomalies by modelling expected daily/weekly seasonality with a
robust, statsmodels-free decomposition and flagging deviations using
median-absolute-deviation based thresholds that a handful of extreme
points cannot poison.
"""

from timeseries_anomaly.decompose import DecompositionResult, decompose
from timeseries_anomaly.detect import Anomaly, DetectionResult, detect_anomalies
from timeseries_anomaly.robust import hampel_filter, mad, median_mad, modified_zscore
from timeseries_anomaly.series import Series

__all__ = [
    "Anomaly",
    "DecompositionResult",
    "DetectionResult",
    "Series",
    "decompose",
    "detect_anomalies",
    "hampel_filter",
    "mad",
    "median_mad",
    "modified_zscore",
]

__version__ = "1.0.0"
