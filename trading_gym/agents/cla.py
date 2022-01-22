from typing import Optional
import numpy as np
import pandas as pd
from collections import deque
from .base import Agent
from pypfopt.efficient_frontier import EfficientFrontier
from ..utils.screener import Screener


class CLAAgent(Agent):
    def __init__(
        self,
        action_space,
        J,
        window,
        screener: Optional[Screener] = None,
        rebalance_each_n_obs: int = 1,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_returns = deque(maxlen=window)
        self.memory_volume = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.J = J
        self._id = "cla_" + J
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0

    def observe(self, observation, action, reward, done, next_observation):

        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def act(self, observation):

        if len(self.memory_returns) != self.memory_returns.maxlen:
            return self.action_space.sample()

        returns = pd.DataFrame(self.memory_returns).dropna(axis=1)

        volume = pd.DataFrame(self.memory_volume).loc[:, returns.columns]

        if self.screener:

            assets_list = self.screener.filter(returns, volume)
            returns = returns.loc[:, assets_list]

        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0
            or self.observation_size != returns.shape[1]
        ):
            expected_returns = returns.mean()

            cov_matrix = returns.cov()

            ef_algo = EfficientFrontier(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
            )

            w = (
                ef_algo.max_sharpe()
                if self.J == "max_sharpe"
                else ef_algo.min_volatility()
            )

            w = pd.Series(
                list(w.values()),
                index=list(w.keys()),
                name=observation["returns"].name,
            )

            if np.any(w < 0):
                w += np.abs(w.min())

            if w.sum() > 1.0 + self.tol:
                w /= w.sum()

            self.w = w

        self.rebalance_counter += 1

        return self.w
