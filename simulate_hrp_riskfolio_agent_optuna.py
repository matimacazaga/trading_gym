from datetime import datetime
from typing import Dict, Optional
from trading_gym.utils.summary import stats
from trading_gym.envs.trading import TradingEnv
import pickle
from tqdm import tqdm
import pandas as pd
import optuna
from trading_gym.agents.hrp_riskfoliolib import HRPAgent
from trading_gym.utils.screener import Screener
from trading_gym.utils.summary import stats


def objective(
    trial,
    assets_data: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    start_eval_date: datetime,
    cash: bool = True,
    fee: float = 0,
):

    window = trial.suggest_int("window", 30, 180)

    n_assets = trial.suggest_int("n_assets", 5, 20)

    n_obs = trial.suggest_int("n_obs", 5, 30)

    codependence = trial.suggest_categorical(
        "codependence",
        [
            "pearson",
            "spearman",
            "abs_pearson",
            "abs_spearman",
            "distance",
            "mutual_info",
            "tail",
        ],
    )

    risk_measure = trial.suggest_categorical(
        "risk_measure",
        [
            "MV",
            "MSV",
            "FLPM",
            "SLPM",
            "VaR",
            "CVaR",
            "WR",
            "MDD",
        ],
    )

    rebalance_each_n_obs = trial.suggest_int("rebalance_each_n_obs", 1, 15)

    leaf_order = trial.suggest_categorical("leaf_order", [True, False])

    env = TradingEnv(assets_data=assets_data, cash=cash, start=start, end=end, fee=fee)

    agent = HRPAgent(
        action_space=env.action_space,
        window=window,
        screener=Screener(n_assets=n_assets, n_obs=n_obs),
        model="HRP",
        rebalance_each_n_obs=rebalance_each_n_obs,
        codependence=codependence,
        covariance="hist",
        objective="Sharpe",
        risk_measure=risk_measure,
        leaf_order=leaf_order,
    )

    try:

        env.register(agent)

        ob = env.reset()

        reward = 0

        done = False

        for _ in tqdm(range(env._max_episode_steps)):
            if done:
                break

            agent.observe(
                observation=ob,
                action=None,
                reward=reward,
                done=done,
                next_reward=None,
            )
            action = agent.act(ob)

            ob, reward, done, _ = env.step({agent.name: action})

        returns = env.agents[agent.name].rewards.loc[start_eval_date:].sum(axis=1)
        returns.iloc[0] = 0.0
        statistics = stats(returns)

        return statistics["Sharpe Ratio"]

    except Exception as e:

        print(e)

        return -999.0


if __name__ == "__main__":

    import pickle
    from datetime import datetime

    assets_data = pickle.load(open("./assets_data_training_v2.pickle", "rb"))
    start = datetime(2018, 5, 1)
    end = datetime(2021, 1, 1)
    start_eval_date = datetime(2019, 1, 1)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(
            trial,
            assets_data,
            start,
            end,
            start_eval_date,
            True,
            0.1,
        ),
        n_trials=500,
    )

    best_params = study.best_params

    print(best_params)

    pickle.dump(
        best_params,
        open(
            "./simulation_results/hrp_riskfolio/best_params_cash_01.pickle",
            "wb",
        ),
    )
