import numpy as np
import pandas as pd
from collections import deque
from math import log, ceil
from .base import Agent
from pypfopt.cla import CLA


class CLAAgent(Agent):
    def __init__(self, action_space, J, window, *args, **kwargs):

        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.J = J
        self._id = "cla" + J

    def observe(self, observation, action, reward, done, next_observation):

        self.memory.append(observation["returns"])

    def act(self, observation):

        memory = pd.DataFrame(self.memory)

        memory.dropna(axis=1, inplace=True)

        if len(self.memory) != self.memory.maxlen:
            return self.action_space.sample()

        expected_returns = memory.mean()

        cov_matrix = memory.cov()

        cla_algo = CLA(expected_returns=expected_returns, cov_matrix=cov_matrix)

        w = (
            cla_algo.max_sharpe()
            if self.J == "max_shapre"
            else cla_algo.min_volatility()
        )

        w = pd.Series(
            list(w.values()),
            index=list(w.keys()),
            name=observation["returns"].name,
        )

        if np.any(w < 0):
            w += np.abs(w.min())

        if w.sum() > 1.0 + 1e-2:
            w /= w.sum()

        self.w = w

        return self.w
