import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, df, length, cols):
        super(TradingEnv, self).__init__()
        self.timestep = 0
        obs, rewards_action = [], []
        for i in range(len(df) - length):
            curr_price, next_price = df["close"].iloc[length + i - 1], df["close"].iloc[length + i]
            diff_percent = (next_price - curr_price) / next_price * 100
            rewards_action.append([-0.01, diff_percent, diff_percent * -1])
            obs.append(MinMaxScaler().fit_transform(df[cols].iloc[i:i + length]).ravel())
        self.observations, self.rewards_action = obs, rewards_action
        self.action_space = spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(length * len(cols),), dtype=np.float64)

    def step(self, action):
        observation = self.observations[self.timestep]
        reward = self.rewards_action[self.timestep][np.argmax(action)]
        done = self.timestep >= len(self.observations) - 1
        self.timestep += 1
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset()
        self.timestep = 0
        return self.observations[self.timestep], {}


class RangeTradingEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, df, length, cols):
        super(RangeTradingEnv, self).__init__()
        self.timestep = 0
        self.percent_steps = np.arange(0, 4, 0.25)
        obs, rewards_action = [], []
        for i in range(len(df) - length):
            curr_price, next_price = df["close"].iloc[length + i - 1], df["close"].iloc[length + i]
            diff_percent = abs((next_price - curr_price) / next_price * 100)
            n = len([i for i in self.percent_steps if i <= diff_percent]) - 1
            des = [round(-0.1 * i, 1) for i in range(1, 15)]
            rewards_action.append((des[::-1] + [0, 1, 0] + des)[len(des) - n + 1:None if -n > -1 else -n])
            obs.append(MinMaxScaler().fit_transform(df[cols].iloc[i:i + length]).ravel())
        self.observations, self.rewards_action = obs, rewards_action
        self.action_space = spaces.Box(low=0, high=1, shape=(16,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(length * len(cols),), dtype=np.float64)

    def step(self, action):
        observation = self.observations[self.timestep]
        reward = self.rewards_action[self.timestep][np.argmax(action)]
        done = self.timestep >= len(self.observations) - 1
        self.timestep += 1
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset()
        self.timestep = 0
        return self.observations[self.timestep], {}
