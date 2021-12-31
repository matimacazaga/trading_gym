from datetime import datetime
from typing import Any, Dict, List
from trading_gym.utils.summary import stats
from trading_gym.agents.base import Agent
from trading_gym.envs.trading import TradingEnv
import pickle
from tqdm import tqdm


def calibrate_agent(
    random_universes: List[List[str]],
    start: datetime,
    end: datetime,
    agent_class: Agent,
    agent_params_grid: List[Dict[str, Any]],
    start_eval_date: datetime,
):

    for i, params in enumerate(agent_params_grid, start=3):

        statistics = []

        for j, assets_data in enumerate(random_universes):

            print(
                f"Simulating params set {i}/{len(agent_params_grid)}, universe {j}/{len(random_universes)}"
            )

            env = TradingEnv(
                assets_data=assets_data,
                cash=True,
                start=start,
                end=end,
            )

            agent = agent_class(action_space=env.action_space, **params)

            env.register(agent)

            ob = env.reset()

            reward = 0

            done = False

            for _ in tqdm(range(env._max_episode_steps)):
                if done:
                    break
                agent.observe(ob, None, reward, done, None)
                action = agent.act(ob)
                ob, reward, done, _ = env.step({agent.name: action})

            statistics.append(
                stats(env.agents[agent.name].rewards.loc[start_eval_date:].sum(axis=1))
            )

        pickle.dump(
            statistics,
            open(
                f"./calibration_results/statistics_{agent.name}_params_{i}.pickle", "wb"
            ),
        )

    print("Done!")


if __name__ == "__main__":

    # from trading_gym.agents.dlpopt import DeepLPortfolioAgent
    from trading_gym.agents.genetic import GeneticAgent

    random_universes = pickle.load(open("./random_universes.pickle", "rb"))
    # windows = [180, 60, 30, 15]
    windows = [
        15,
    ]
    epochs = 200
    params_grid = []
    for w in windows:
        params_grid.append({"window": w, "pop_size": 500, "generations": 200})
    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)
    start_eval_date = datetime(2020, 1, 1)
    calibrate_agent(
        random_universes=random_universes[:10],
        start=start,
        end=end,
        agent_class=GeneticAgent,
        agent_params_grid=params_grid,
        start_eval_date=start_eval_date,
    )
