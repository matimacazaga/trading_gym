import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from .base import Agent
from ..envs.spaces import PortfolioVector
from ..utils.numpy_utils import softmax


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
        epochs=200,
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
        self.model = self.build_model(hidden_units)
        self.past_n_obs = past_n_obs
        self.retrain_each_n_obs = retrain_each_n_obs

    def observe(self, observation, action, reward, done, next_reward):
        self.memory.append(observation["returns"].values)

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

    def build_model(self, hidden_units):
        raise NotImplementedError

    def act(self, observation):

        memory = np.array(self.memory)
        print(memory.shape)

        if len(self.memory) < self.window:
            return self.action_space.sample()

        X, y = self.split_sequences(memory)

        self.model.fit(X, y, batch_size=self.batch_size,
                       epochs=self.epochs, verbose=0)

        prediction = self.model.predict(
            self.reshape_memory(memory[-self.past_n_obs:, :]))

        if self.policy == "softmax":

            action = pd.Series(
                prediction.ravel(),
                index=observation["returns"].index,
                name=observation["returns"].name,
            )
            action = softmax(action)

            return action

        elif self.policy == "best":

            action = np.zeros_like(prediction).ravel()
            action[np.argmax(prediction)] = 1.0
            action = pd.Series(
                action,
                index=observation["returns"].index,
                name=observation["returns"].name,
            )

            return action


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


# DONE: add retrain_each_n_days and past_n_observations,
# DONE: set windows to the minimum amount of observations needed to start training (should be greater than past_n_observations)