from typing import Dict, Optional
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from collections import deque
from ..envs.spaces import PortfolioVector
from ..utils.screener import Screener
from .base import Agent


class DeepLPortfolio:
    def __init__(
        self, input_shape: int, hidden_units: int, outputs: int, returns: np.ndarray
    ):

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.outputs = outputs
        self.returns = tf.cast(tf.constant(returns), float)

    def build_model(self) -> tf.keras.Model:

        model = keras.models.Sequential()

        model.add(
            keras.layers.Dense(
                self.hidden_units,
                activation="relu",
                batch_input_shape=(None, self.input_shape),
            )
        )

        model.add(
            keras.layers.Dense(
                self.hidden_units,
                activation="relu",
            ),
        )

        model.add(
            keras.layers.Dense(
                self.outputs,
                activation="softmax",
            ),
        )

        model.compile(loss=self.sharpe_loss, optimizer="adam")

        return model

    @tf.function
    def sharpe_loss(self, _, y_pred):

        portfolio_returns = tf.reduce_sum(tf.multiply(self.returns, y_pred), axis=1)

        sharpe = keras.backend.mean(portfolio_returns) / keras.backend.std(
            portfolio_returns
        )

        return -sharpe


class DeepLPortfolioAgent(Agent):

    _id = "deeplportfolio"

    def __init__(
        self,
        action_space: PortfolioVector,
        window: int,
        hidden_units: int = 20,
        epochs=200,
        screener: Optional[Screener] = None,
        rebalance_each_n_obs: int = 7,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_returns = deque(maxlen=window)
        self.memory_volume = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def act(self, observation: Dict[str, pd.Series]) -> pd.Series:
        if len(self.memory_returns) != self.memory_returns.maxlen:

            return self.action_space.sample()

        returns = pd.DataFrame(self.memory_returns).dropna(axis=1)

        volume = pd.DataFrame(self.memory_volume).loc[:, returns.columns]

        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0
            or self.observation_size != returns.shape[1]
        ):

            self.observation_size = returns.shape[1]

            if self.screener:
                if isinstance(self.screener, tuple) or isinstance(self.screener, list):
                    for screener in self.screener:
                        assets_list = screener.filter(returns, volume)
                        returns = returns.loc[:, assets_list]
                        volume = volume.loc[:, assets_list]
                else:
                    assets_list = self.screener.filter(returns, volume)
                    returns = returns.loc[:, assets_list]

            dlpopt = DeepLPortfolio(
                self.memory_returns.maxlen * returns.shape[1],
                self.hidden_units,
                returns.shape[1],
                returns.values,
            )

            model = dlpopt.build_model()

            # ravel("F") ravels the array column first
            model.fit(
                returns.values.ravel("F")[np.newaxis, :],
                np.zeros(
                    (
                        1,
                        self.action_space.shape[0],
                    )
                ),
                epochs=self.epochs,
                shuffle=False,
                verbose=0,
            )

            w = model.predict(
                tf.constant(returns.values.ravel("F")[np.newaxis, :], float)
            )[0]

            if w.sum() > 1.0 + self.tol:

                w /= w.sum()

            self.w = pd.Series(
                w,
                index=returns.columns,
                name=observation["returns"].name,
            )

        self.rebalance_counter += 1

        return self.w
