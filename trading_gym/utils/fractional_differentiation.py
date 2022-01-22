"""
Fractional differentiation.
See https://medium.com/swlh/fractionally-differentiated-features-9c1947ed2b55
for reference.
"""

import numpy as np
from typing import Union
from statsmodels.tsa.stattools import adfuller
from numba import njit, prange
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

NUM_CORES = multiprocessing.cpu_count()


@njit(cache=True, nogil=True)
def get_weights(d: Union[float, int], series_length: int, thres: float) -> np.ndarray:

    """
    Weights for fixed-width window fractional differentiation.

    Parameters
    ----------
    d: float or int
        Differentiation coefficient. Must be greater than 0 to
        guarantee convergence.
    series_length: int
        Lenght of the series to differentiate.
    thres: float
        Minimum value allowed for a weight. All the weights below thres will be
        discarded. It limits the length of the memory.

    Returns
    -------
    w: np.ndarray
        Weights.
    """

    w = [1.0]
    k = 1

    while k < series_length:
        w_ = (-w[-1] * (d - k + 1)) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1

    w = np.array(w[::-1], dtype=np.float64).reshape(-1, 1)

    return w


@njit(cache=True, nogil=True, parallel=True)
def fixed_width_window_fracdiff(
    series: np.ndarray, d: Union[float, int], thres: float = 1e-3
) -> np.ndarray:
    """
    Differentiates a series using the Fixed-Width Window Fractional
    Differentiation method. The threshold determines which data points
    will be discarded (weights equals to 0). A greater threshold discard more
    data points.

    Parameters
    ----------
    series: np.ndarray
        Series to differentiate.
    d: float or int
        Differentiation coefficient. Must be greater than 0 to
        guarantee convergence.
    tresh: float
        Minimum value allowed for a weight. All the weights below thres will be
        discarded. It limits the length of the memory.

    Returns
    -------
    diff_series: np.ndarray
        Differentiated series.
    """
    w = get_weights(d, series.shape[0], thres)

    width = len(w) - 1

    diff_series = np.zeros(shape=series.shape, dtype=np.float64)

    diff_series[:width] = np.nan

    for i in prange(width, series.shape[0]):

        diff_series[i] = np.dot(w.T, series[i - width : i + 1])[0]

    return diff_series


def _diff_and_statistics(
    series: np.ndarray,
    d: float,
    thres: float,
) -> list:

    diff_series = fixed_width_window_fracdiff(series, d, thres)

    adf = adfuller(
        diff_series[~np.isnan(diff_series)], maxlag=1, regression="c", autolag=None
    )

    corr = np.corrcoef(
        series[~np.isnan(diff_series)],
        diff_series[~np.isnan(diff_series)],
    )[0, 1]

    return [d] + list(adf[:2]) + [adf[3]] + [adf[4]["5%"]] + [corr]


def diff_series(
    series: np.ndarray,
    n_values: int = 1000,
    weights_thres: float = 1e-3,
) -> np.ndarray:
    """
    Find the optimal differentiation coefficient given an ADF test threshold
    and return the differentiated series.

    Parameters
    ----------
    series: np.ndarray
        Series to differentiate
    adf_threshold: float
        Threshold for the ADF test.
    n_values: int
        Amount of values of d between 0 and 2 to consider.
    weights_thres: float
        Minimum value allowed for a weight. All the weights below thres will be
        discarded. It limits the length of the memory.

    Returns
    -------
    np.ndarray
        Fractionally differentiated series.
    """
    ds = np.linspace(0.0, 2.0, n_values)

    stats = Parallel(n_jobs=NUM_CORES)(
        delayed(_diff_and_statistics)(series, d, weights_thres) for d in ds
    )

    stats = pd.DataFrame(
        stats, columns=["d", "adf_stat", "p_val", "n_obs", "95_conf", "corr"]
    )

    mask_1 = stats.loc[:, "adf_stat"] <= stats.loc[:, "95_conf"]
    mask_2 = stats.loc[:, "p_val"] <= 0.05
    d = stats.loc[mask_1 & mask_2].sort_values("corr", ascending=False).iloc[0]["d"]

    diff_series = fixed_width_window_fracdiff(
        series,
        d,
    )

    return diff_series
