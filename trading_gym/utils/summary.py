from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltl
from ..utils import metrics


def stats(returns: Union[np.ndarray, pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Generates a statistical report of the strategy.

    Parameters
    ----------
    returns: np.ndarray | pd.Series | pd.DataFrame

    Returns
    -------
    table: pd.Series
        Strategy report.
    """

    report = {
        "Mean Return": metrics.mean_return(returns),
        "Cumulative Returns": metrics.cumulative_returns(returns),
        "PnL": metrics.pnl(returns),
        "Volatility": metrics.std_dev(returns),
        "Sharpe Ratio": metrics.sharpe_ratio(returns),
        "Max Drawdown": metrics.max_drawdown(returns).iloc[-1],
        "Average Drawdown Time": metrics.average_drawdown_time(returns).days,
        "Skewness": metrics.skewness(returns),
        "Kurtosis": metrics.kurtosis(returns),
        "Tail Ratio": metrics.tail_ratio(returns),
        "VaR": metrics.value_at_risk(returns),
        "CVaR": metrics.conditional_value_at_risk(returns),
        "Hit Ratio": metrics.hit_ratio(returns),
        "Average Win to Average Loss": metrics.avg_win_to_avg_loss(returns),
        "Average Profitability per Trade": metrics.avg_profit_per_trade(returns),
    }

    table = pd.Series(report, name=(returns.name or "Strategy"), dtype=object)

    return table
