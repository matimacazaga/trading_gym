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
from numba import prange, int64, float64, typed, types, njit
from numba.experimental import jitclass

eps = np.finfo(float).eps


@njit(nogil=True)
def create_individual(genome_size: int):
    genome = np.zeros(genome_size)
    for i in range(genome_size):
        genome[i] = np.random.uniform(0.0, 100.0)

    return genome


@njit(nogil=True)
def evaluate_individual(
    individual: np.ndarray,
    mean_returns: np.ndarray,
    sigma_returns: np.ndarray,
    max_weight: float,
) -> float:

    constrain = False
    norm_w = individual / individual.sum()
    if norm_w.sum() == 1.0:
        if np.any(norm_w > max_weight):
            fitness = 0.0
            mu_portfolio = 0.0
            constrain = True
        if not constrain:
            mu_portfolio = np.dot(mean_returns, norm_w)
            sigma_portfolio = np.sqrt(np.dot(np.dot(norm_w, sigma_returns), norm_w))
            fitness = mu_portfolio / (sigma_portfolio + eps)

    else:
        fitness = 0.0
        mu_portfolio = 0.0

    return fitness


@njit(nogil=True)
def tournament_selection(
    participants: np.ndarray,
    fitnesses: np.ndarray,
) -> np.ndarray:

    best_fit = participants[np.argmax(fitnesses), :]

    return best_fit


@njit(nogil=True)
def crossover(
    individual_1: np.ndarray, individual_2: np.ndarray, crossover_prob: float
) -> Tuple[np.ndarray]:
    if np.random.uniform(0.0, 1.0) < crossover_prob:
        dice = np.random.randint(1, len(individual_1))
        child_1 = np.hstack((individual_1[:dice], individual_2[dice:]))
        child_2 = np.hstack((individual_2[:dice], individual_1[dice:]))

        return child_1, child_2
    return individual_1, individual_2


@njit(nogil=True)
def mutation(individual: np.ndarray, mutation_prob: float):

    if np.random.uniform(0.0, 1.0) < mutation_prob:
        individual_ = np.zeros(len(individual))
        for gen in range(len(individual)):
            individual_[gen] = individual[gen] * np.random.uniform(0.5, 1.5)

        return individual_
    return individual


@njit(nogil=True, parallel=False)
def run_genetic_algorithm(
    mean_returns: np.ndarray,
    sigma_returns: np.ndarray,
    pop_size: int = 100,
    generations: int = 100,
    mutation_prob: float = 0.25,
    crossover_prob: float = 0.75,
    tournament_contestants: int = 25,
    max_weight: float = 1.0,
):

    genome_size = mean_returns.shape[0]
    best_individual = np.zeros(genome_size)
    best_fitness = np.nan
    population = np.zeros((pop_size, genome_size))

    for i in prange(pop_size):
        population[i, :] = create_individual(genome_size)

    for _ in range(generations):
        fitnesses = np.zeros(pop_size)
        for i in prange(pop_size):
            fitnesses[i] = evaluate_individual(
                population[i], mean_returns, sigma_returns, max_weight
            )

        best_index = np.argmax(fitnesses)

        if fitnesses[best_index] > best_fitness or np.isnan(best_fitness):
            best_fitness = fitnesses[best_index]
            best_individual = population[best_index]

        selected = np.zeros((pop_size, genome_size))
        for i in prange(pop_size):
            indexes = np.random.choice(
                np.arange(0, pop_size),
                size=tournament_contestants,
                replace=False,
            )
            selected[i, :] = tournament_selection(
                population[indexes], fitnesses[indexes]
            )

        np.random.shuffle(selected)

        selected_A = selected[int(0.5 * pop_size) :]

        selected_B = selected[: int(0.5 * pop_size)]

        next_generation_1 = np.zeros((int(0.5 * pop_size), genome_size))

        next_generation_2 = np.zeros((int(0.5 * pop_size), genome_size))

        for i in prange(int(0.5 * pop_size)):
            child_1, child_2 = crossover(selected_A[i], selected_B[i], crossover_prob)
            next_generation_1[i, :] = child_1
            next_generation_2[i, :] = child_2

        next_generation = np.concatenate((next_generation_1, next_generation_2))

        for i in prange(pop_size):
            next_generation[i] = mutation(next_generation[i], mutation_prob)

        population = next_generation[:, :]

    return best_individual


class GeneticAgent(Agent):

    _id = "geneticagent"

    def __init__(self, action_space: PortfolioVector, window: int, *args, **kwargs):

        self.action_space = action_space

        self.memory = deque(maxlen=window)

        self.w = self.action_space.sample()

        self.generations = kwargs.get("generations", 100)

        self.pop_size = kwargs.get("pop_size", 500)

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs) -> None:

        self.memory.append(observation["returns"])

    def act(self, observation: Dict[str, pd.Series]) -> np.ndarray:

        memory = pd.DataFrame(self.memory)

        memory.dropna(axis=1, inplace=True)

        mu = np.mean(memory.values, axis=0)

        if len(self.memory) != self.memory.maxlen:

            return self.action_space.sample()

        else:

            sigma = np.cov(memory.values, rowvar=False)

        # genetic_algo = GeneticAlgorithm(mu, sigma, self.pop_size, self.generations)

        # self.w = genetic_algo.evolve()

        w = run_genetic_algorithm(mu, sigma, self.pop_size, self.generations)

        w /= w.sum()

        self.w = pd.Series(
            w,
            index=memory.columns,
            name=observation["returns"].name,
        )

        return self.w
