import itertools
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from collections import deque
from .base import Agent
from dataclasses import dataclass
from joblib import Parallel, delayed
from ..envs.spaces import PortfolioVector
from numba import prange, int64, float64, typed, types
from numba.experimental import jitclass

eps = np.finfo(float).eps

spec = [
    ("mean_returns", float64[:]),
    ("sigma_returns", float64[:, :]),
    ("pop_size", int64),
    ("generations", int64),
    ("mutation_prob", float64),
    ("crossover_prob", float64),
    ("tournament_contestants", int64),
    ("max_weight", float64),
]


@jitclass(spec)
class GeneticAlgorithm:
    def __init__(
        self,
        mean_returns: np.ndarray,
        sigma_returns: np.ndarray,
        pop_size: int = 100,
        generations: int = 10,
        mutation_prob: float = 0.25,
        crossover_prob: float = 0.75,
        tournament_contestants: int = 25,
        max_weight: float = 1.0,
    ):
        self.mean_returns = mean_returns
        self.sigma_returns = sigma_returns
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_contestants = tournament_contestants
        self.max_weight = max_weight

    def _create_individual(self) -> Dict[str, Any]:
        genome = np.zeros(self.mean_returns.shape[0])
        for i in range(len(genome)):
            genome[i] = np.random.uniform(0.0, 100.0)

        individual = (genome, np.nan)

        return individual

    def create_population(self) -> List[Tuple[np.ndarray, float]]:
        pop = []
        for _ in range(self.pop_size):
            pop.append(self._create_individual())
        return pop

    def mutation(
        self, individual: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:
        if np.random.uniform(0.0, 1.0) < self.mutation_prob:
            for gen in range(len(individual[0])):
                individual[0][gen] = individual[0][gen] * np.random.uniform(0.5, 1.5)

        return individual

    def crossover(
        self,
        individual_1: Tuple[np.ndarray, float],
        individual_2: Tuple[np.ndarray, float],
    ) -> Tuple[Tuple[np.ndarray, float], Tuple[np.ndarray, float]]:

        if np.random.uniform(0.0, 1.0) < self.crossover_prob:
            dice = np.random.randint(1, len(individual_1[0]))
            child_1 = (
                np.hstack((individual_1[0][:dice], individual_2[0][dice:])),
                np.nan,
            )
            child_2 = (
                np.hstack((individual_2[0][:dice], individual_1[0][dice:])),
                np.nan,
            )

            return child_1, child_2

        return individual_1, individual_2

    def tournament_selection(
        self, pop: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, float]:

        indexes = np.random.choice(
            np.arange(0, self.pop_size),
            size=self.tournament_contestants,
            replace=False,
        )

        tournament = []
        for i in indexes:
            tournament.append(pop[i])

        fitnesses = np.zeros(len(tournament))
        for i in range(len(tournament)):
            fitnesses[i] = tournament[i][1]

        best_fit = tournament[np.argmax(fitnesses)]

        return best_fit

    def evaluation_function(
        self, individual: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:

        constrain = False
        norm_w = individual[0] / individual[0].sum()
        if norm_w.sum() == 1.0:
            if np.any(norm_w > self.max_weight):
                fitness = 0.0
                mu_portfolio = 0.0
                constrain = True
            if not constrain:
                mu_portfolio = np.dot(self.mean_returns, norm_w)
                sigma_portfolio = np.sqrt(
                    np.dot(np.dot(norm_w, self.sigma_returns), norm_w)
                )
                fitness = mu_portfolio / (sigma_portfolio + eps)

        else:
            fitness = 0.0
            mu_portfolio = 0.0

        return (individual[0], fitness)

    def evolve(self):

        pop = self.create_population()

        hof = (np.zeros(self.mean_returns.shape[0]), np.nan)

        for _ in range(self.generations):

            for i in prange(self.pop_size):
                pop[i] = self.evaluation_function(pop[i])

            fitnesses = np.zeros(self.pop_size)
            for i in prange(self.pop_size):
                fitnesses[i] = pop[i][1]

            best_in_generation = pop[np.argmax(fitnesses)]

            if best_in_generation[1] > hof[1] or np.isnan(hof[1]):
                hof = best_in_generation

            selected = []
            for i in prange(self.pop_size):
                selected.append(self.tournament_selection(pop))

            selected_A = []
            selected_B = []
            indexes = np.arange(0, self.pop_size)
            np.random.shuffle(indexes)

            for i in range(0, int(self.pop_size * 0.5)):
                selected_A.append(selected[indexes[i]])

            for i in range(int(self.pop_size * 0.5), self.pop_size):
                selected_B.append(selected[indexes[i]])

            next_generation = []
            for i in prange(len(selected_A)):
                child_1, child_2 = self.crossover(selected_A[i], selected_B[i])
                next_generation.append(child_1)
                next_generation.append(child_2)

            for i in prange(len(next_generation)):
                next_generation[i] = self.mutation(next_generation[i])

            pop = next_generation

        return hof[0]


class GeneticAgent(Agent):

    _id = "geneticagent"

    def __init__(self, action_space: PortfolioVector, window: int, *args, **kwargs):

        self.action_space = action_space

        self.memory = deque(maxlen=window)

        self.w = self.action_space.sample()

        self.generations = kwargs.get("generations", 100)

        self.pop_size = kwargs.get("pop_size", 500)

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs) -> None:

        self.memory.append(observation["returns"].values)

    def act(self, observation: Dict[str, pd.Series]) -> np.ndarray:

        memory = np.array(self.memory)

        mu = np.mean(memory, axis=0)

        if len(self.memory) != self.memory.maxlen:

            sigma = np.eye(self.action_space.shape[0])

        else:

            sigma = np.cov(memory.T)

        genetic_algo = GeneticAlgorithm(mu, sigma, self.pop_size, self.generations)

        self.w = genetic_algo.evolve()

        self.w /= self.w.sum()

        self.w = pd.Series(
            self.w,
            index=observation["returns"].index,
            name=observation["returns"].name,
        )

        return self.w
