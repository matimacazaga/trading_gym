import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
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
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.policy = policy
        self.window = window
        self.past_n_obs = past_n_obs
        self.retrain_each_n_obs = retrain_each_n_obs
        self.retrain_counter = 0
        self.model = self.build_model(hidden_units)
        self.scaler = StandardScaler()
        self.w = self.action_space.sample()

    def observe(self, observation, action, reward, done, next_reward):
        self.memory.append(observation["returns"])

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

    def reshape_memory(self, memory):
        return memory.reshape((1, self.past_n_obs, self.observation_size))

    def reshape_training_data(self, X):
        return X

    def build_model(self, hidden_units):
        raise NotImplementedError

    def act(self, observation):

        memory = pd.DataFrame(self.memory)

        memory.dropna(axis=1, inplace=True)

        if len(self.memory) < self.window:
            return self.action_space.sample()

        if self.retrain_counter % self.retrain_each_n_obs == 0:

            memory_ = self.scaler.fit_transform(memory.values)

            X, y = self.split_sequences(memory_)

            X = self.reshape_training_data(X)

            self.model.fit(
                X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0
            )

        self.retrain_counter += 1

        prediction = self.model.predict(
            self.reshape_memory(
                self.scaler.transform(
                    memory.values[-self.past_n_obs :, :].reshape(
                        -1, self.action_space.shape[0]
                    )
                ).ravel()
            )
        )

        if self.policy == "softmax":

            w = self.scaler.inverse_transform(prediction.reshape(1, -1)).ravel()

            w = softmax(w)

        elif self.policy == "best":

            w = np.zeros_like(
                self.scaler.inverse_transform(prediction.reshape(1, -1))
            ).ravel()

            w[np.argmax(prediction)] = 1.0

        self.w = pd.Series(
            w,
            index=memory.columns,
            name=observation["returns"].name,
        )

        return self.w


class RnnLSTMAgent(RnnAgent):

    _id = "rnn-lstm"

    def build_model(self, hidden_units):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                hidden_units,
                activation="relu",
                return_sequences=True,
                input_shape=(self.past_n_obs, self.observation_size),
            )
        )
        model.add(tf.keras.layers.LSTM(hidden_units, activation="relu"))
        model.add(tf.keras.layers.Dense(self.observation_size))
        model.compile(optimizer="adam", loss="mse")
        return model


# TODO: rnn classification model
class RnnGRUAgent(RnnAgent):

    _id = "rnn-gru"

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
