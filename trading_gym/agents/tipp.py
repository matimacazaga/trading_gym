from .base import Agent
from ..envs.spaces import PortfolioVector
from typing import Optional, Dict
from ..utils.screener import Screener
import pandas as pd
from collections import deque
import numpy as np


class TippAgent(Agent):
    def __init__(
        self,
        action_space: PortfolioVector,
        agent: Agent,
        multiplier: int,
        floor_pct: float,
        window: int = 50,
        rebalance_each_n_obs: int = 1,
        w_risk_min: float = 0.1,
        w_risk_max: float = 0.95,
    ):
        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_rewards = []
        self.memory_returns = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0
        self.multiplier = multiplier
        self.floor_pct = floor_pct
        self.agent = agent
        self.window = window
        self.portfolio_pnl = 1.0
        self.floor_value = floor_pct
        self.w_risk_min = w_risk_min
        self.w_risk_max = w_risk_max
        self._id = "tipp_" + self.agent.name

    def observe(self, observation: Dict[str, pd.DataFrame], *args, **kwargs) -> None:

        self.agent.observe(observation, *args, **kwargs)
        reward = kwargs["reward"]
        self.memory_rewards.append(
            reward if isinstance(reward, int) else reward[self.name]
        )

    def act(self, observation: Dict[str, pd.DataFrame]) -> pd.Series:

        w = self.agent.act(observation)

        if len(self.memory_rewards) < self.window:

            self.w = w

            return w

        rewards = pd.Series(self.memory_rewards[self.window :])

        portfolio_pnl = (1.0 + rewards).cumprod().iloc[-1] if len(rewards) > 1 else 1.0

        floor_value_updated = portfolio_pnl * self.floor_pct

        if floor_value_updated > self.floor_value:

            self.floor_value = floor_value_updated

        cushion = (portfolio_pnl - self.floor_value) / portfolio_pnl

        w_risk = max(
            min(self.multiplier * cushion, self.w_risk_max, 1.0), self.w_risk_min
        )

        w_riskless = 1.0 - w_risk
        print(portfolio_pnl, self.floor_value, cushion)
        print(w_risk, w_riskless)

        N = w_risk / w.sum()

        w = N * w

        print(w_risk, w_riskless)

        if "Cash" in w.index:
            w["Cash"] += w_riskless
        else:
            w["Cash"] = w_riskless

        self.w = w

        return self.w
