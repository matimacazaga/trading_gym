from collections import deque
from numbers import Number
from re import S
from typing import Any, List, Optional, Tuple, Dict, Union
from numba import prange, njit
import numpy as np
import pandas as pd
from ..agents.base import Agent
from ..agents.hrp_riskfoliolib import HRPAgent
from ..agents.tipp import TippAgent
from ..utils.screener import Screener
from ..envs.trading import TradingEnv
from joblib import Parallel, delayed
from ..envs.spaces import PortfolioVector
from riskfolio import HCPortfolio
from ..utils.summary import stats


def create_individual(genome_spec: Dict[str, Tuple[str, Number, Number]]):
    genome = {}

    for k, v in genome_spec.items():
        genome[k] = (
            np.random.randint(v[1], v[2] + 1)
            if v[0] == int
            else np.random.uniform(v[1], v[2] + 0.1)
        )

    return genome


def evaluate_individual(
    individual: np.ndarray,
    assets_data: Dict[str, pd.DataFrame],
    window: int = 180,
    cash: bool = False,
    fee: float = 0.01,
    eval_period: int = 30,
    rebalance_each_n_obs: int = 7,
    n_obs_volume: int = 15,
    n_assets_volume: int = 200,
) -> float:

    env = TradingEnv(assets_data=assets_data, cash=cash, fee=fee)

    agent = HRPAgent(
        action_space=env.action_space,
        window=window,
        screener=(
            Screener("volume", n_assets_volume, n_obs_volume),
            Screener(
                "volatility",
                individual["n_assets_volatility"],
                individual["n_obs_volatility"],
            ),
            Screener(
                "returns", individual["n_assets_returns"], individual["n_obs_returns"]
            ),
        ),
        objective="Sharpe",
        risk_measure="MDD",
        rebalance_each_n_obs=rebalance_each_n_obs,
        w_min=0.05,
        w_max=0.35,
    )

    env.register(agent)

    ob = env.reset()

    reward = 0

    done = False

    while not done:

        agent.observe(
            observation=ob,
            action=None,
            reward=reward,
            done=done,
            next_reward=None,
        )

        action = agent.act(ob)

        ob, reward, done, _ = env.step({agent.name: action})

    returns = env.agents[agent.name].rewards.iloc[-eval_period:].sum(axis=1)

    returns.iloc[0] = 0.0

    statistics = stats(returns)

    return statistics["Sharpe Ratio"]


def tournament_selection(
    population: Tuple[Dict[str, Number]],
    fitnesses: List[float],
    tournament_contestants: int,
) -> Dict[str, Number]:

    indexes = np.random.choice(
        np.arange(0, len(population), dtype=int),
        size=tournament_contestants,
        replace=False,
    )

    participants = [population[i] for i in indexes]

    fitnesses_ = [fitnesses[i] for i in indexes]

    best_fit = participants[np.argmax(fitnesses_)]

    return best_fit


def crossover(
    individual_1: Dict[str, Number],
    individual_2: Dict[str, Number],
    crossover_prob: float,
) -> Dict[str, Number]:

    if np.random.uniform(0.0, 1.0) < crossover_prob:
        ind_length = len(individual_1)
        keys = list(individual_1.keys())
        dice = np.random.randint(1, ind_length)
        child_1 = {
            **{k: individual_1[k] for k in keys[:dice]},
            **{k: individual_2[k] for k in keys[dice:]},
        }

        child_2 = {
            **{k: individual_2[k] for k in keys[:dice]},
            **{k: individual_1[k] for k in keys[dice:]},
        }

        return child_1, child_2

    return individual_1, individual_2


def mutation(
    individual: Dict[str, Number],
    mutation_prob: float,
    genome_spec: Dict[str, Tuple[str, Number, Number]],
):

    if np.random.uniform(0.0, 1.0) < mutation_prob:
        individual_ = {}
        for k in individual:
            individual_[k] = (
                np.random.randint(genome_spec[k][1], genome_spec[k][2])
                if genome_spec[k][0] == int
                else np.random.uniform(genome_spec[k][1], genome_spec[k][2])
            )

        return individual_

    return individual


def run_genetic_algorithm(
    genome_spec: Dict[str, Tuple[str, Number, Number]],
    assets_data: Dict[str, pd.DataFrame],
    pop_size: int = 100,
    generations: int = 100,
    crossover_prob: float = 0.75,
    mutation_prob: float = 0.25,
    tournament_contestants: int = 20,
    window: int = 180,
    cash: bool = False,
    fee: float = 0.01,
    eval_period: int = 15,
    rebalance_each_n_obs: int = 7,
    n_obs_volume: int = 15,
    n_assets_volume: int = 200,
) -> Dict[str, Number]:

    population = tuple(create_individual(genome_spec) for _ in range(pop_size))

    best_fitness = np.nan
    best_individual = {}

    for _ in range(generations):
        # fitnesses = [
        #     evaluate_individual(
        #         individual,
        #         assets_data,
        #         window,
        #         cash,
        #         fee,
        #         eval_period,
        #         rebalance_each_n_obs,
        #     )
        #     for individual in population
        # ]
        fitnesses = Parallel(n_jobs=24)(
            delayed(evaluate_individual)(
                individual,
                assets_data,
                window,
                cash,
                fee,
                eval_period,
                rebalance_each_n_obs,
                n_obs_volume,
                n_assets_volume,
            )
            for individual in population
        )

        best_index = np.argmax(fitnesses)

        if fitnesses[best_index] > best_fitness or np.isnan(best_fitness):

            best_fitness = fitnesses[best_index]

            best_individual = population[best_index]

        selected = Parallel(n_jobs=24)(
            delayed(tournament_selection)(population, fitnesses, tournament_contestants)
            for _ in range(pop_size)
        )

        np.random.shuffle(selected)

        selected_A = selected[int(0.5 * pop_size) :]
        selected_B = selected[: int(0.5 * pop_size)]

        next_generation = []

        for i in range(int(0.5 * pop_size)):
            child_1, child_2 = crossover(selected_A[i], selected_B[i], crossover_prob)
            next_generation.append(child_1)
            next_generation.append(child_2)

        next_generation = Parallel(n_jobs=24)(
            delayed(mutation)(individual, mutation_prob, genome_spec)
            for individual in next_generation
        )

        population = next_generation

    return best_individual, best_fitness


class GeneticHRPAgent(Agent):
    def __init__(
        self,
        action_space: PortfolioVector,
        genome_spec: Dict[str, Tuple[str, Number, Number]],
        window: int = 180,
        eval_period: int = 15,
        pop_size: int = 100,
        generations: int = 100,
        crossover_prob: float = 0.75,
        mutation_prob: float = 0.25,
        tournament_contestants: int = 20,
        n_obs_volume: int = 15,
        n_assets_volume: int = 200,
        retrain_each_n_obs: int = 30,
        rebalance_each_n_obs: int = 7,
        w_min: float = 0.05,
        w_max: float = 0.35,
        cash: bool = False,
        fee: float = 0.01,
    ):

        self.pop_size = pop_size

        self.generations = generations

        self.crossover_prob = crossover_prob

        self.mutation_prob = mutation_prob

        self.tournament_contestants = tournament_contestants

        self.cash = cash

        self.fee = fee

        self.n_assets_volume = n_assets_volume

        self.n_obs_volume = n_obs_volume

        self.rebalance_each_n_obs = rebalance_each_n_obs

        self.action_space = action_space

        self.observation_size = self.action_space.shape[0]

        self.memory_rewards = []

        self.memory_returns = deque(maxlen=window + eval_period)

        self.memory_volume = deque(maxlen=window + eval_period)

        self.memory_close = deque(maxlen=window + eval_period)

        self.memory_high = deque(maxlen=window + eval_period)

        self.memory_low = deque(maxlen=window + eval_period)

        self.memory_open = deque(maxlen=window + eval_period)

        self.window = window

        self.w = self.action_space.sample()

        self.rebalance_counter = 0

        self.retrain_counter = 0

        self.w_min = w_min

        self.w_max = w_max

        self.retrain_each_n_obs = retrain_each_n_obs

        self.eval_period = eval_period

        self.genome_spec = genome_spec

        self.rebalance_each_n_obs = 1

        self._id = "genetic_hrp"

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        reward = kwargs["reward"]

        self.memory_rewards.append(
            reward if isinstance(reward, int) else reward[self.name]
        )

        self.memory_returns.append(observation["returns"])

        self.memory_volume.append(observation["volume"])

        self.memory_close.append(observation["close"])

        self.memory_open.append(observation["open"])

        self.memory_low.append(observation["low"])

        self.memory_high.append(observation["high"])

    def act(self, observation: Dict[str, pd.Series]) -> pd.Series:

        if len(self.memory_returns) < self.memory_returns.maxlen:

            self.w = self.action_space.sample()

            return self.w

        returns = pd.DataFrame(self.memory_returns).dropna(axis=1)

        returns = returns.iloc[-self.window :]

        if self.retrain_counter % self.retrain_each_n_obs == 0.0:

            assets_data = {
                "close": pd.DataFrame(self.memory_close),
                "low": pd.DataFrame(self.memory_low),
                "high": pd.DataFrame(self.memory_high),
                "open": pd.DataFrame(self.memory_open),
                "volume": pd.DataFrame(self.memory_volume),
            }

            best_individual, best_fitness = run_genetic_algorithm(
                self.genome_spec,
                assets_data,
                self.pop_size,
                self.generations,
                self.crossover_prob,
                self.mutation_prob,
                self.tournament_contestants,
                self.window,
                self.cash,
                self.fee,
                self.eval_period,
                self.rebalance_each_n_obs,
                self.n_obs_volume,
                self.n_assets_volume,
            )

            self.n_assets_volatility = best_individual["n_assets_volatility"]

            self.n_obs_volatility = best_individual["n_obs_volatility"]

            self.n_assets_returns = best_individual["n_assets_returns"]

            self.n_obs_returns = best_individual["n_obs_returns"]

            print(best_individual, best_fitness)

        self.retrain_counter += 1

        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0.0
            or self.observation_size != returns.shape[1]
        ):

            self.observation_size = returns.shape[1]

            screeners = [
                Screener("volume", self.n_assets_volume, self.n_obs_volume),
                Screener("volatility", self.n_assets_volatility, self.n_obs_volatility),
                Screener("returns", self.n_assets_returns, self.n_obs_returns),
            ]

            volume = pd.DataFrame(self.memory_volume).loc[:, returns.columns]

            for screener in screeners:

                assets_list = screener.filter(returns, volume)

                returns = returns.loc[:, assets_list]

                volume = volume.loc[:, assets_list]

            w_min_series = pd.Series(
                np.full(returns.shape[1], self.w_min), index=returns.columns
            )

            w_max_series = pd.Series(
                np.full(returns.shape[1], self.w_max), index=returns.columns
            )

            hrp_algo = HCPortfolio(returns, w_max=w_max_series, w_min=w_min_series)

            w = hrp_algo.optimization(
                model="HRP",
                obj="Sharpe",
                codependence="pearson",
                rm="MDD",
                rf=0.0,
                covariance="hist",
                leaf_order=True,
            )

            w = pd.Series(
                w.loc[:, "weights"],
                index=w.index,
                name=observation["returns"].name,
            )

            if np.any(w < 0):
                w += np.abs(w.min())

            if w.sum() > 1.0 + self.tol:
                w /= w.sum()

            self.w = w

        self.rebalance_counter += 1

        return self.w
