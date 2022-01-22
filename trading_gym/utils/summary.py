from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ..utils import metrics
from ..agents.base import Agent
from datetime import datetime


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
        "Semi-Volatility": metrics.semi_std_dev(returns),
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


def get_agents_metrics(
    agents: List[Agent],
    agents_names: List[str],
    start_date: datetime,
    close_prices: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame]:

    agents_returns = {}
    for agent_name, agent in zip(agents_names, agents):
        returns = agent["agent"].rewards.loc[start_date:].sum(axis=1)
        returns.iloc[0] = 0
        agents_returns[agent_name] = returns

    if close_prices is not None:
        for symbol in close_prices.columns:
            returns = close_prices.loc[:, symbol].pct_change().loc[start_date:]
            returns.iloc[0] = 0.0
            agents_returns[symbol] = returns

    agents_stats = {}
    for agent_name in agents_returns:
        agents_stats[agent_name] = stats(agents_returns[agent_name])

    stats_df = {}
    for agent_name, sts in agents_stats.items():
        stats_df[agent_name] = {
            k: v for k, v in sts.items() if k not in ["Cumulative Returns", "PnL"]
        }
        stats_df[agent_name]["PnL"] = sts["PnL"].iloc[-1]
        stats_df[agent_name]["Sharpe Ratio"] *= np.sqrt(365.0)

    stats_df = pd.DataFrame(stats_df)

    cumulative_returns = (
        agents_stats[agents_names[0]]["Cumulative Returns"]
        .to_frame()
        .reset_index()
        .rename({"index": "date", 0: agents_names[0]}, axis=1)
    )

    for agent_name in agents_names[1:] + close_prices.columns.tolist():
        cumulative_returns.loc[:, agent_name] = agents_stats[agent_name][
            "Cumulative Returns"
        ].values

    cumulative_returns = (
        cumulative_returns.set_index("date")
        .stack()
        .reset_index()
        .rename({"level_1": "Agent", 0: "cumulative_return"}, axis=1)
    )

    return stats_df, cumulative_returns
