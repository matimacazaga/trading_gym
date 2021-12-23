from typing import Tuple, Union
import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
import scipy.stats as sp_stats

eps = np.finfo(float).eps


def _check_returns(returns: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:

    if returns.ndim > 2:
        raise ValueError("Returns tensor is not compatible.")


def kurtosis(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute the kurtosis of the returns.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
    Strategy returns (percentage).

    Returns
    -------
    pd.Series | pd.DataFrame
        Kurtosis.
    """

    _check_returns(returns)

    if isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            returns = pd.Series(returns)
        else:
            returns = pd.DataFrame(returns)

    return returns.kurt(axis=0)


def skewness(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute the skewness of the returns.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
    Strategy returns (percentage).

    Returns
    -------
    pd.Series | pd.DataFrame
        Skewness.
    """

    _check_returns(returns)

    if isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            returns = pd.Series(returns)
        else:
            returns = pd.DataFrame(returns)

    return returns.skew(axis=0)


def pnl(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Computes the PnL.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    pnl: np.ndarray | pd.Series | pd.DataFrame
        PnL of the strategy.
    """

    _check_returns(returns)

    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)

    return out


def cumulative_returns(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Computes the cumulative returns.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    cumulative_returns: pd.Series
        Cumulative returns.
    """

    out = pnl(returns)
    out = np.subtract(out, 1.0)

    return out


def sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Computes the Sharpe Ratio.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    float | np.ndarray | pd.Series
        Sharpe Ratio (not annualized).
    """

    _check_returns(returns)

    return np.mean(returns, axis=0) / np.std(returns, ddof=1, axis=0)


def probabilistic_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame], benchmark_sr: float
) -> float:
    """
    Computes the probabilistic Sharpe Ratio. [Pag. 203, Advances in Financial
    Machine Learning, M. Lopez de Prado]

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    float | pd.Series | pd.DataFrame
        Probabilistic Sharpe Ratio.
    """

    sr = sharpe_ratio(returns)

    if isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            returns = pd.Series(returns)
        else:
            returns = pd.DataFrame(returns)

    numerator = (sr - benchmark_sr) * np.sqrt(len(returns) - 1.0)

    denominator = np.sqrt(
        1.0 - skewness(returns) * sr + 0.25 * (kurtosis(returns) - 1.0) * sr ** 2
    )

    x = numerator / denominator

    if isinstance(x, pd.Series):

        return sp_stats.norm.cdf(x)

    else:

        return x.apply(sp_stats.norm.cdf)


def deflated_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series], srs: Union[list, np.ndarray, pd.Series]
) -> float:
    """
    Computes the deflated Sharpe ratio. [Pag. 204, Advances in Financial
    Machine Learning, M. Lopez de Prado]

    Parameters
    ----------
    returns: np.ndarray | pd.Series
        Returns of the strategy.
    srs: list | np.array | pd.Series
        Set of Sharpe ratio estimations.

    Returns
    -------
    float
        Probabilistic Sharpe ratio using an estimation of SR* as benchmark.
    """
    a = np.sqrt(np.var(srs, ddof=1))
    b = (1.0 - np.euler_gamma) * sp_stats.norm.ppf(1.0 - (1.0 / len(srs)))
    c = np.euler_gamma * sp_stats.norm.ppf(1.0 - (1.0 / len(srs)) * np.exp(-1))
    sr = a * (b + c)

    return probabilistic_sharpe_ratio(returns, sr)


def compute_drawdowns_and_time_under_water(
    series: pd.Series, dollars: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes the drawdowns series and the associated time under water.
    [Pag. 201, Advances in Financial Machine Learning, M. Lopez de Prado]

    Parameters
    ----------
    series: pd.Series
        Returns series or performance series in dollars.
    dollars: bool
        Indicates whether the series is in dollars (performance) or not.

    Returns
    -------
    dd: pd.DataFrame
        Drawdowns.
    tuw: pd.Series
        Time under water.
    """

    df0 = series.to_frame("pnl")
    df0.loc[:, "hwm"] = series.expanding().max()
    df1 = df0.groupby("hwm").min().reset_index()
    df1.columns = ["hwm", "min"]
    df1.index = df0.loc[:, "hwm"].drop_duplicates(keep="first").index
    df1 = df1.loc[df1.loc[:, "hwm"] > df1.loc[:, "min"]]
    if dollars:
        dd = df1.loc[:, "hwm"] - df1.loc[:, "min"]
    else:
        dd = 1.0 - df1.loc[:, "min"] / df1.loc[:, "hwm"]

    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, "Y")).values
    tuw = pd.Series(tuw, index=df1.index[:-1])

    return dd, tuw


def get_hhi(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Computes the return concentration. [Pag. 200, Advances in Financial
    Machine Learning, M. Lopez de Prado]

    It can be computed for negative returns, positive returns, or by fequency (e.g., monthly returns).

    Parameters
    ----------
    returns: pd.Series | np.ndarray
        Strategy returns (percentage).

    Returns
    -------
    hhi: float
        Returns concentration.
    """

    if returns.shape[0] <= 2:
        return np.nan

    wght = returns / returns.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - len(returns) ** (-1)) / (1.0 - len(returns) ** (-1))

    return hhi


def hit_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Computes the hit ratio (number of positive trades from the total amount of
    trades performed).

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.

    Returns
    -------
    float | np.ndarray | pd.Series
        Hit ratio
    """

    _check_returns(returns)

    return np.sum(returns > 0.0, axis=0) / len(returns)


def avg_win_to_avg_loss(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the ratio between the average win and the average loss.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    float | np.ndarray | pd.Series
        Ratio between the average win and average loss.
    """

    _check_returns(returns)

    avg_win = returns[returns > 0.0].mean(axis=0)
    avg_loss = returns[returns < 0.0].mean(axis=0)

    return np.abs((avg_win + eps) / (avg_loss + eps))


def avg_profit_per_trade(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the average profit per trade.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    float | np.ndarray | pd.Series
        Average profit per trade
    """
    n = len(returns)
    mask_win = returns > 0.0
    mask_loss = returns < 0.0
    prob_win = np.sum(mask_win, axis=0) / n
    prob_loss = np.sum(mask_loss, axis=0) / n
    avg_win = returns[mask_win].mean(axis=0)
    avg_loss = returns[mask_loss].mean(axis=0)

    return (prob_win * avg_win) - (prob_loss * avg_loss)


def drawdown(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Computes the drawdown.

    Parameters
    ----------
    returns: np.ndarray | pd.Series
        Strategy returns (percentage).

    Returns
    -------
    drawdown: pd.Series
        Strategy drawdown.
    """

    cum_returns = cumulative_returns(returns)
    if isinstance(cum_returns, np.ndarray):
        if cum_returns.ndim == 1:
            cum_returns = pd.Series(cum_returns)
        else:
            cum_returns = pd.DataFrame(cum_returns)

    expanding_max = cum_returns.expanding(1).max()
    drawdown = expanding_max - cum_returns

    return drawdown


def max_drawdown(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes the maximum drawdown.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).
    Returns
    -------
    pd.Series | pd.DataFrame
        Maximum drawdown over time.
    """
    _drawdown = drawdown(returns)
    return _drawdown.expanding(1).max()


def average_drawdown_time(returns: Union[np.ndarray, pd.Series]) -> pd.Series:
    """
    Computes the average drawdown time.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns (percentage).

    Returns
    -------
    datetime.timedelta
    """
    _drawdown = drawdown(returns)
    return _drawdown[_drawdown == 0.0].index.to_series().diff().mean()


def mean_return(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the mean return.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.

    Returns
    -------
    float | np.ndarray | pd.Series
        Mean return.
    """
    return returns.mean(axis=0)


def std_dev(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the standard deviation of the returns.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.

    Returns
    -------
    float | np.ndarray | pd.Series
        Standard deviation of the returns.
    """
    return returns.std(ddof=1, axis=0)


def tail_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[float, np.ndarray]:
    """
    Computes the ratio between the 95-percentil and the 5-percentile (absolute
    value) of the returns.

    For instance, a tail ratio of 0.25 means that the losses are four times as
    bad as the profits.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.

    Returns
    -------
    float | np.ndarray
        Tail ratio.
    """

    return np.abs(np.percentile(returns, 95, axis=0)) / np.abs(
        np.percentile(returns, 5, axis=0)
    )


def value_at_risk(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame], cutoff: float = 0.05
) -> Union[float, np.ndarray]:
    """
    Computes the VaR.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.
    cutoff: float
        Float representing the cutoff percentage for the inferior returns
        percentile.

    Returns
    -------
    float | np.ndarray
        The VaR value.
    """

    return np.percentile(returns, 100.0 * cutoff, axis=0)


def conditional_value_at_risk(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame], cutoff: float = 0.05
) -> Union[float, np.ndarray]:
    """
    Computes the CVaR.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame
        Strategy returns.
    cutoff: float
        Float representing the cutoff percentage for the inferior returns
        percentile.

    Returns
    -------
    float | np.ndarray
        The CVaR value.
    """

    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(
        np.partition(returns, cutoff_index, axis=0)[: cutoff_index + 1], axis=0
    )


# TODO: Define kurtosis and skewness. Finish copying the other metrics.
