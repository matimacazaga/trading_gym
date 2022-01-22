import streamlit as st
import pickle
from .summary import stats
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime


DATA_PATH = "./assets_data_testing.pickle"

PERFORMANCE_COLUMNS = [
    "Mean Return",
    "Sharpe Ratio (Ann.)",
    "Tail Ratio",
    "Hit Ratio",
    "Average Win to Average Loss",
    "Average Profitability per Trade",
    "Cumulative Return",
]

RISK_COLUMNS = [
    "Volatility",
    "Semi-Volatility",
    "Max Drawdown",
    "Average Drawdown Time",
    "VaR",
    "CVaR",
]
MAPPER_PERFORMANCE_COLUMNS = {
    "Cumulative Return": "{:.2%}",
    "Mean Return": "{:.2%}",
    "Tail Ratio": "{:.2f}",
    "Hit Ratio": "{:.2%}",
    "Average Win to Average Loss": "{:.2f}",
    "Average Profitability per Trade": "{:.2f}",
    "Sharpe Ratio (Ann.)": "{:.2f}",
}

MAPPER_RISK_COLUMNS = {
    "Volatility": "{:.2%}",
    "Semi-Volatility": "{:.2%}",
    "Max Drawdown": "{:.2f}",
    "Average Drawdown Time": "{:.0f}",
    "VaR": "{:.2%}",
    "CVaR": "{:.2%}",
}


def get_strategy_info(file, start_date):

    statistics = {}

    rolling_volatilities = {}

    returns = file["agent"].rewards.loc[start_date:].sum(axis=1)

    returns.iloc[0] = 0.0

    weights = (
        file["agent"]
        .actions.loc[start_date:]
        .stack()
        .reset_index()
        .rename({"level_0": "date", "level_1": "symbol", 0: "weight"}, axis=1)
    )

    weights = weights.loc[weights.loc[:, "weight"] != 0.0]

    days_in_portfolio = (
        weights.groupby("symbol").size().to_frame().rename({0: "days"}, axis=1)
    )

    weights.drop_duplicates(subset=["symbol", "weight"], inplace=True)

    rolling_volatilities["ML Strategy"] = returns.rolling(7).std(ddof=1).dropna()

    statistics = {}

    statistics["ML Strategy"] = stats(returns)

    assets_data = pickle.load(open(DATA_PATH, "rb"))

    btc = assets_data["close"].loc[:, "BTCUSDT"]

    btc = btc.pct_change().loc[start_date:]

    btc.iloc[0] = 0.0

    eth = assets_data["close"].loc[:, "ETHUSDT"]

    eth = eth.pct_change().loc[start_date:]

    eth.iloc[0] = 0.0

    rolling_volatilities["BTCUSDT"] = btc.rolling(7).std(ddof=1).dropna()

    rolling_volatilities["ETHUSDT"] = eth.rolling(7).std(ddof=1).dropna()

    rolling_volatilities = pd.DataFrame(rolling_volatilities)

    rolling_volatilities = (
        rolling_volatilities.stack()
        .reset_index()
        .rename(
            {
                "level_0": "date",
                "level_1": "Strategy",
                0: "Volatility",
            },
            axis=1,
        )
    )

    statistics["BTCUSDT"] = stats(btc)

    statistics["ETHUSDT"] = stats(eth)

    stats_df = {}

    for name in statistics:

        stats_df[name] = {
            k: v
            for k, v in statistics[name].items()
            if k not in ["PnL", "Cumulative Returns"]
        }

        stats_df[name]["Sharpe Ratio"] *= np.sqrt(365.0)

        stats_df[name]["Cumulative Return"] = statistics[name][
            "Cumulative Returns"
        ].iloc[-1]

    stats_df = pd.DataFrame(stats_df)

    stats_df.rename(
        {"Sharpe Ratio": "Sharpe Ratio (Ann.)"},
        axis=0,
        inplace=True,
    )

    cumulative_returns = {}

    cumulative_returns["ML Strategy"] = statistics["ML Strategy"]["Cumulative Returns"]
    cumulative_returns["BTCUSDT"] = statistics["BTCUSDT"]["Cumulative Returns"]
    cumulative_returns["ETHUSDT"] = statistics["ETHUSDT"]["Cumulative Returns"]

    cumulative_returns = (
        pd.DataFrame(cumulative_returns)
        .stack()
        .reset_index()
        .rename(
            {"level_0": "date", "level_1": "Strategy", 0: "Cumulative Return"}, axis=1
        )
    )

    return (
        stats_df,
        cumulative_returns,
        rolling_volatilities,
        weights,
        days_in_portfolio,
    )


def run_app(file_path: str):

    file = pickle.load(open(file_path, "rb"))

    st.write("# Strategy Report")

    st.write("**NOTE:** all the results are computed considering a $1\%$ fee.")

    start_date = st.date_input(
        "Select starting date:",
        value=datetime(2021, 1, 1),
        min_value=datetime(2021, 1, 1),
        max_value=datetime(2022, 1, 16),
    )

    (
        stats_df,
        cumulative_returns,
        rolling_volatilities,
        weights,
        days_in_portfolio,
    ) = get_strategy_info(file, start_date)

    st.write("## Performance metrics")

    st.dataframe(
        stats_df.loc[PERFORMANCE_COLUMNS, :].apply(
            lambda s: s.map(MAPPER_PERFORMANCE_COLUMNS.get(s.name, "{:.2f}").format),
            axis=1,
        )
    )

    c = (
        alt.Chart(cumulative_returns)
        .mark_line()
        .encode(
            x="date:T",
            y="Cumulative Return:Q",
            color="Strategy:N",
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("Cumulative Return:Q", format=".2f"),
            ],
        )
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    st.write("## Risk metrics")

    st.dataframe(
        stats_df.loc[RISK_COLUMNS, :].apply(
            lambda s: s.map(MAPPER_RISK_COLUMNS.get(s.name, "{:.2f}").format),
            axis=1,
        )
    )

    c = (
        alt.Chart(rolling_volatilities)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "Volatility:Q",
                title="Volatility (7D rolling window)",
                axis=alt.Axis(format=".0%"),
            ),
            color="Strategy",
        )
    )

    st.altair_chart(c, use_container_width=True)

    st.write("## Weights")

    unique_coins = weights.loc[:, "symbol"].nunique()

    st.write(f"Number of unique coins selected by the model: **{unique_coins}**")

    highest_avg_weight = (
        weights.groupby("symbol")
        .agg({"weight": "mean"})
        .sort_values("weight", ascending=False)
    )

    st.write("### Top 10 coins with highets average weight")

    c = (
        alt.Chart(highest_avg_weight.iloc[:10].reset_index())
        .mark_bar()
        .encode(
            x=alt.X("symbol:N", title="Symbol"),
            y=alt.Y("weight:Q", title="Avg. Weight", axis=alt.Axis(format=".2%")),
            tooltip=[alt.Tooltip("weight:Q", format=".2%")],
        )
    )

    st.altair_chart(c, use_container_width=True)

    top_10_days_in_portfolio = days_in_portfolio.sort_values("days", ascending=False)

    c = (
        alt.Chart(top_10_days_in_portfolio.iloc[:10].reset_index())
        .mark_bar()
        .encode(
            x=alt.X("symbol:N", title="Symbol"),
            y=alt.Y("days:Q", title="Days in Portfolio"),
            tooltip=[alt.Tooltip("days:Q")],
        )
    )

    st.altair_chart(c, use_container_width=True)
