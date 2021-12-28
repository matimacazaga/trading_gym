from typing import Dict
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from collections import deque
from trading_gym.envs.spaces import PortfolioVector
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
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.hidden_units = hidden_units
        self.epochs = epochs

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        self.memory.append(observation["returns"].values)

    def act(self, observation: Dict[str, pd.Series]) -> pd.Series:

        memory = np.array(self.memory)

        if len(self.memory) != self.memory.maxlen:
            return self.action_space.sample()
        else:

            dlpopt = DeepLPortfolio(
                self.memory.maxlen * self.action_space.shape[0],
                self.hidden_units,
                self.action_space.shape[0],
                memory,
            )

            model = dlpopt.build_model()

            model.fit(
                memory.ravel("F")[np.newaxis, :],
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

            w = model.predict(tf.constant(memory.ravel("F")[np.newaxis, :], float))[0]

            self.w = pd.Series(
                w,
                index=observation["returns"].index,
                name=observation["returns"].name,
            )

            return w
