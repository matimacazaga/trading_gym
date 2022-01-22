from datetime import datetime
from typing import Any, Dict, List, Optional
from trading_gym.utils.summary import stats
from trading_gym.agents.base import Agent
from trading_gym.envs.trading import TradingEnv
import pickle
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed


def simulate_agent_random_universes(
    random_universes: List[List[str]],
    start: datetime,
    end: datetime,
    agent_class: Agent,
    agent_params_grid: List[Dict[str, Any]],
    start_eval_date: datetime,
):

    for i, params in enumerate(agent_params_grid):

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


def sim(
    assets_data,
    agent_class,
    cash,
    start,
    end,
    fee,
    tipp_agent,
    tipp_agent_params,
    i,
    params,
    file_suffix,
):

    env = TradingEnv(assets_data=assets_data, cash=cash, start=start, end=end, fee=fee)

    if tipp_agent is None:
        agent = agent_class(action_space=env.action_space, **params)
    else:
        agent = agent_class(
            action_space=env.action_space,
            agent=tipp_agent(action_space=env.action_space, **tipp_agent_params),
            **params,
        )

    env.register(agent)

    ob = env.reset()

    reward = 0

    done = False

    for _ in range(env._max_episode_steps):
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

    results = {"agent": env.agents[agent.name], "params": params}

    pickle.dump(
        results,
        open(
            f"./simulation_results/{agent.name}/results_{agent.name}_params_{i}{'_' + file_suffix if file_suffix else ''}.pickle",
            "wb",
        ),
    )


def simulate_agent(
    assets_data: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    agent_class: Agent,
    agent_params_grid: List[Dict[str, Any]],
    cash: bool = True,
    fee: float = 0,
    tipp_agent: Optional[Agent] = None,
    tipp_agent_params: Optional[Dict[str, Any]] = None,
    file_suffix: Optional[str] = None,
):

    for i, params in enumerate(agent_params_grid):
        # try:
        print(f"Simulating params set {i+1}/{len(agent_params_grid)}")

        env = TradingEnv(
            assets_data=assets_data, cash=cash, start=start, end=end, fee=fee
        )

        if tipp_agent is None:
            agent = agent_class(action_space=env.action_space, **params)
        else:
            agent = agent_class(
                action_space=env.action_space,
                agent=tipp_agent(action_space=env.action_space, **tipp_agent_params),
                **params,
            )

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

        results = {"agent": env.agents[agent.name], "params": params}

        pickle.dump(
            results,
            open(
                f"./simulation_results/{agent.name}/results_{agent.name}_params_{i}{'_' + file_suffix if file_suffix else ''}.pickle",
                "wb",
            ),
        )

        # except Exception as e:
        #     print(e)
        #     print(params)

    print("Done!")


def parallel_simulate_agent(
    assets_data: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    agent_class: Agent,
    agent_params_grid: List[Dict[str, Any]],
    cash: bool = True,
    fee: float = 0,
    tipp_agent: Optional[Agent] = None,
    tipp_agent_params: Optional[Dict[str, Any]] = None,
    file_suffix: Optional[str] = None,
):

    _ = Parallel(n_jobs=24)(
        delayed(sim)(
            assets_data,
            agent_class,
            cash,
            start,
            end,
            fee,
            tipp_agent,
            tipp_agent_params,
            i,
            params,
            file_suffix,
        )
        for i, params in tqdm(enumerate(agent_params_grid))
    )
