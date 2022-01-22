from typing import Deque, Dict, Optional
from numpy.core.fromnumeric import shape
from ..envs.spaces import PortfolioVector
from .base import Agent
import numpy as np
from collections import deque
import pandas as pd
from ..utils.screener import Screener


class EqualWeightAgent(Agent):

    _id = "ew"

    def __init__(
        self,
        action_space: PortfolioVector,
        window=50,
        screener: Optional[Screener] = None,
        rebalance_each_n_obs: int = 1,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.w = self.action_space.sample()
        self.memory_returns = deque(maxlen=window)
        self.memory_volume = deque(maxlen=window)
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0

    def observe(self, observation: Dict[str, pd.DataFrame], *args, **kwargs):

        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def act(self, observation: Dict[str, pd.DataFrame]) -> np.ndarray:

        if len(self.memory_returns) < self.memory_returns.maxlen:
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

            w = np.full(
                shape=len(returns.columns),
                fill_value=1.0 / len(returns.columns),
            )

            self.w = pd.Series(
                w,
                index=returns.columns,
                name=observation["returns"].name,
            )

        self.rebalance_counter += 1

        return self.w
