from datetime import datetime
from typing import Any, Dict, List, Optional
from trading_gym.utils.summary import stats
from trading_gym.agents.base import Agent
from trading_gym.envs.trading import TradingEnv
import pickle
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from trading_gym.agents.combined import CombinedAgent
from trading_gym.agents.hrp_riskfoliolib import HRPAgent


def simulation(
    params: Dict[str, Dict[str, Any]],
    i: int,
    assets_data: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    cash: bool = False,
    fee: float = 0,
    file_suffix: Optional[str] = None,
):

    env = TradingEnv(assets_data=assets_data, cash=cash, start=start, end=end, fee=fee)

    agent_combined = CombinedAgent(
        action_space=env.action_space, agent_name="hrp", **params["combined"]
    )

    agent_returns = HRPAgent(action_space=env.action_space, **params["hrp_returns"])

    agent_volatility = HRPAgent(
        action_space=env.action_space, **params["hrp_volatility"]
    )

    agent_returns._id = "hrp_returns"

    agent_volatility._id = "hrp_volatility"

    env.register(agent_combined)
    env.register(agent_returns)
    env.register(agent_volatility)

    ob = env.reset()

    reward = {agent_returns.name: 0.0, agent_volatility.name: 0.0}

    done = False

    for _ in range(env._max_episode_steps):
        if done:
            break

        agent_returns.observe(
            observation=ob,
            action=None,
            reward=reward[agent_returns.name],
            done=done,
            next_reward=None,
        )

        agent_volatility.observe(
            observation=ob,
            action=None,
            reward=reward[agent_volatility.name],
            done=done,
            next_reward=None,
        )

        agent_combined.observe(
            reward[agent_returns.name], reward[agent_volatility.name]
        )

        action_returns = agent_returns.act(ob)
        action_volatility = agent_volatility.act(ob)
        action_combined = agent_combined.act(ob, action_returns, action_volatility)

        ob, reward, done, _ = env.step(
            {
                agent_returns.name: action_returns,
                agent_volatility.name: action_volatility,
                agent_combined.name: action_combined,
            }
        )

    results = {"agent": env.agents[agent_combined.name], "params": params}

    pickle.dump(
        results,
        open(
            f"./simulation_results/{agent_combined.name}/results_{agent_combined.name}_params_{i}{'_' + file_suffix if file_suffix else ''}.pickle",
            "wb",
        ),
    )


def simulate_agent(
    assets_data: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    agents_params_grid: List[Dict[str, Dict[str, Any]]],
    cash: bool = False,
    fee: float = 0,
    file_suffix: Optional[str] = None,
    parallel: bool = True,
):

    if not parallel:
        for i, params in tqdm(enumerate(agents_params_grid)):

            simulation(params, i, assets_data, cash, start, end, fee, file_suffix)
    else:
        _ = Parallel(n_jobs=min(24, len(agents_params_grid)))(
            delayed(simulation)(
                params, i, assets_data, cash, start, end, fee, file_suffix
            )
            for i, params in tqdm(enumerate(agents_params_grid))
        )

    print("Done!")


if __name__ == "__main__":
    from trading_gym.utils.screener import Screener

    # params_grid = []

    # n_obs_volatility = [7, 10, 15, 20]
    # n_obs_returns = [7, 10, 15, 20]
    # window_combined = [3, 5, 7, 10]
    # for n_obs_volat in n_obs_volatility:
    #     for n_obs_ret in n_obs_returns:
    #         for window in window_combined:
    #             params_grid.append(
    #                 {
    #                     "hrp_returns": {
    #                         "window": 180,
    #                         "screener": [
    #                             Screener("volume", 200, 15),
    #                             Screener("returns", 10, n_obs_ret),
    #                         ],
    #                         "model": "HRP",
    #                         "rebalance_each_n_obs": 7,
    #                         "codependence": "pearson",
    #                         "covariance": "hist",
    #                         "objective": "Sharpe",
    #                         "risk_measure": "MDD",
    #                         "leaf_order": True,
    #                         "w_min": 0.05,
    #                         "w_max": 0.35,
    #                     },
    #                     "hrp_volatility": {
    #                         "window": 180,
    #                         "screener": [
    #                             Screener("volume", 200, 15),
    #                             Screener("volatility", 10, n_obs_volat),
    #                         ],
    #                         "model": "HRP",
    #                         "rebalance_each_n_obs": 7,
    #                         "codependence": "pearson",
    #                         "covariance": "hist",
    #                         "objective": "Sharpe",
    #                         "risk_measure": "MDD",
    #                         "leaf_order": True,
    #                         "w_min": 0.05,
    #                         "w_max": 0.35,
    #                     },
    #                     "combined": {"window": window, "rebalance_each_n_obs": 7},
    #                 },
    #             )

    params = [
        {
            "hrp_returns": {
                "window": 180,
                "screener": [
                    Screener("volume", 200, 15),
                    Screener("returns", 10, 15),
                ],
                "model": "HRP",
                "rebalance_each_n_obs": 7,
                "codependence": "pearson",
                "covariance": "hist",
                "objective": "Sharpe",
                "risk_measure": "MDD",
                "leaf_order": True,
                "w_min": 0.05,
                "w_max": 0.35,
            },
            "hrp_volatility": {
                "window": 180,
                "screener": [
                    Screener("volume", 200, 15),
                    Screener("volatility", 10, 15),
                ],
                "model": "HRP",
                "rebalance_each_n_obs": 7,
                "codependence": "pearson",
                "covariance": "hist",
                "objective": "Sharpe",
                "risk_measure": "MDD",
                "leaf_order": True,
                "w_min": 0.05,
                "w_max": 0.35,
            },
            "combined": {"window": 7, "rebalance_each_n_obs": 7},
        },
    ]

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))

    simulate_agent(
        assets_data,
        datetime(2019, 5, 1),
        datetime(2022, 1, 17),
        params,
        fee=0.01,
        file_suffix="testing",
        parallel=False,
    )
