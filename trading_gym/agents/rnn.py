from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque

from ..utils.screener import Screener
from .base import Agent
from ..envs.spaces import PortfolioVector
from ..utils.numpy_utils import softmax
from sklearn.preprocessing import StandardScaler


class RnnAgent(Agent):
    def __init__(
        self,
        action_space: PortfolioVector,
        window,
        past_n_obs: int,
        retrain_each_n_obs: int,
        hidden_units=50,
        policy="softmax",
        batch_size=32,
        epochs=50,
        screener: Optional[Screener] = None,
        rebalance_each_n_obs: int = 1,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_returns = deque(maxlen=window)  # []
        self.memory_volume = deque(maxlen=window)
        self.batch_size = batch_size
        self.epochs = epochs
        self.policy = policy
        self.window = window
        self.past_n_obs = past_n_obs
        self.retrain_each_n_obs = retrain_each_n_obs
        self.retrain_counter = 0
        self.hidden_units = hidden_units
        self.scaler = StandardScaler()
        self.w = self.action_space.sample()
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0

    def observe(self, observation, action, reward, done, next_reward):
        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def split_sequences(self, sequences):
        X, y = [], []
        for i in range(len(sequences)):
            end_ix = i + self.past_n_obs
            if end_ix > len(sequences) - 1:
                break
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def reshape_memory(self, memory, columns):
        return memory.reshape((1, self.past_n_obs, columns))

    def reshape_training_data(self, X):
        return X

    def build_model(self, hidden_units):
        raise NotImplementedError

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

            if (
                self.retrain_counter % self.retrain_each_n_obs == 0
                or self.observation_size != returns.shape[1]
            ):

                self.observation_size = returns.shape[1]

                self.model = self.build_model(returns.shape[1], self.hidden_units)

                memory_ = self.scaler.fit_transform(returns.values)

                X, y = self.split_sequences(memory_)

                X = self.reshape_training_data(X)

                self.model.fit(
                    X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0
                )

            self.retrain_counter += 1
            self.rebalance_counter += 1

            prediction = self.model.predict(
                self.reshape_memory(
                    self.scaler.transform(
                        returns.values[-self.past_n_obs :, :]
                    ).ravel(),
                    returns.shape[1],
                )
            )

            if self.policy == "softmax":

                w = self.scaler.inverse_transform(prediction.reshape(1, -1)).ravel()

                w += w.min()

                w[np.isnan(w)] = 0.0

                w = softmax(w)

                if np.all(w == 0.0) or np.any(np.isnan(w)):

                    w = np.random.uniform(0, 1, w.shape)
                    w /= w.sum()

            elif self.policy == "best":

                w = np.zeros_like(
                    self.scaler.inverse_transform(prediction.reshape(1, -1))
                ).ravel()

                w[np.argmax(prediction)] = 1.0

            if w.sum() > 1.0 + self.tol:
                w /= w.sum()

            self.w = pd.Series(
                w,
                index=returns.columns,
                name=observation["returns"].name,
            )

        return self.w


class RnnLSTMAgent(RnnAgent):

    _id = "rnn_lstm"

    def build_model(self, n_features, hidden_units):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                hidden_units,
                activation="relu",
                return_sequences=True,
                input_shape=(self.past_n_obs, n_features),
            )
        )
        model.add(tf.keras.layers.LSTM(hidden_units, activation="relu"))
        model.add(tf.keras.layers.Dense(n_features))
        model.compile(optimizer="adam", loss="mse")
        return model


# TODO: rnn classification model
class RnnGRUAgent(RnnAgent):

    _id = "rnn_gru"

    def build_model(self, hidden_units):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.GRU(
                hidden_units,
                activation="relu",
                return_sequences=True,
                input_shape=(self.past_n_obs, self.observation_size),
            )
        )
        model.add(tf.keras.layers.GRU(hidden_units, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size))
        model.compile(optimizer="adam", loss="mse")
        return model


class RnnConvGRUAgent(Agent):

    _id = "rnn-conv-gru"

    def __init__(
        self,
        action_space: PortfolioVector,
        window,
        past_n_obs: int,
        retrain_each_n_obs: int,
        hidden_units=50,
        policy="softmax",
        batch_size=32,
        epochs=200,
        *args,
        **kwargs
    ):
        super().__init__(
            action_space,
            window,
            past_n_obs,
            retrain_each_n_obs,
            hidden_units,
            policy,
            batch_size,
            epochs,
        )

        self.n_seq = kwargs["n_seq"]

    def build_model(self, hidden_units):

        model = tf.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(
                    filters=64, kernel_size=1, padding="causal", activation="relu"
                ),
                input_shape=(
                    None,
                    int(self.past_n_obs / self.n_seq),
                    self.observation_size,
                ),
            )
        )

    def reshape_memory(self, memory):
        return memory.reshape(
            (1, self.n_seq, int(self.past_n_obs / self.n_seq), self.observation_size)
        )
