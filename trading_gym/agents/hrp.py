from typing import Dict, Optional
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from ..utils.screener import Screener
from .base import Agent
from collections import deque
from ..envs.spaces import PortfolioVector
from pypfopt.hierarchical_portfolio import HRPOpt


class HRPAgent(Agent):

    _id = "hrp"

    def __init__(
        self,
        action_space: PortfolioVector,
        window: int = 50,
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
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def act(self, observation: Dict[str, pd.Series]) -> np.ndarray:

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

            self.observation_size = returns.shape[1]

            cov_matrix = returns.cov()

            hrp_algo = HRPOpt(returns, cov_matrix)

            w = hrp_algo.optimize()

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
