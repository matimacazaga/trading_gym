from datetime import datetime
from typing import Any, Dict, List
from trading_gym.utils.summary import stats
from trading_gym.agents.base import Agent
from trading_gym.envs.trading import TradingEnv
import pickle
from tqdm import tqdm
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.cla import CLAAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_training.pickle", "rb"))
    windows = [260, 60, 30, 15]
    objectives = ["max_sharpe", "min_volatility"]
    screeners = [
        Screener("volume", 10, 15),
        Screener("volume", 5, 15),
        Screener("volume", 10, 5),
        Screener("volume", 5, 5),
    ]
    params_grid = []
    for w in windows:
        for objective in objectives:
            for screener in screeners:
                params_grid.append({"window": w, "J": objective, "screener": screener})

    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)
    start_eval_date = datetime(2020, 1, 1)
    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        agent_class=CLAAgent,
        agent_params_grid=params_grid,
        file_suffix="volume",
    )
