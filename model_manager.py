from datetime import datetime as dt
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import environment
import json
import os


class Model:
    def __init__(self):
        self.models_dir = "models"
        os.mkdir(self.models_dir) if not os.path.exists(self.models_dir) else ...
        self.models = os.listdir(path=self.models_dir)

    def create(self, name, env_name, df, length, cols, actor_arch, critic_arch):
        if name in self.models:
            return
        env = getattr(environment, env_name)(df, length, cols)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        net_arch = dict(pi=actor_arch, qf=critic_arch)
        model = TD3("MlpPolicy", env, action_noise=action_noise, policy_kwargs={"net_arch": net_arch},
                    buffer_size=1_500_000, verbose=1)
        os.mkdir(f"{self.models_dir}//{name}")
        model.save(f"{self.models_dir}//{name}//model")
        with open(f"{self.models_dir}//{name}//config.json", "w") as f:
            json.dump({"model": {
                "name": name, "actor_arch": actor_arch, "critic_arch": critic_arch},
                "env": {"name": env_name, "length": length, "cols": cols}
            }, f, indent=4)
        with open(f"{self.models_dir}//{name}//learn_results.json", "w") as f:
            json.dump([], f, indent=4)
        self.models.append(name)

    def load(self, name):
        return TD3.load(f"{self.models_dir}//{name}//model")

    def learn(self, name, df, steps, epoch=1):
        with open(f"{self.models_dir}//{name}//config.json", "r") as f:
            config = json.load(f)
        env = getattr(environment, config["env"]["name"])(df, config["env"]["length"], config["env"]["cols"])
        model = TD3.load(f"{self.models_dir}//{name}//model", env)
        model.save(self.get_path(name) + "-backup")
        for e in range(max([epoch, 1])):
            model.learn(total_timesteps=steps, log_interval=10)
            model.save(self.get_path(name))
            with open(f"{self.models_dir}//{name}//learn_results.json", "r") as f:
                learn_results = json.load(f)
            learn_result = self.test(name, df, False, False)
            learn_results.append(learn_result)
            with open(f"{self.models_dir}//{name}//learn_results.json", "w") as f:
                json.dump(learn_results, f, indent=4)
        return model.get_env()

    def predict(self, name, df):
        with open(f"{self.models_dir}//{name}//config.json", "r") as f:
            config = json.load(f)
        cols = config["env"]["cols"]
        length = config["env"]["length"]
        obs = MinMaxScaler().fit_transform(df[cols].iloc[len(df) - length - 1:-1]).ravel()
        action, _ = self.load(name).predict(obs.reshape((1, -1)))
        obs = MinMaxScaler().fit_transform(df[cols].iloc[len(df) - length:]).ravel()
        next_action, _ = self.load(name).predict(obs.reshape((1, -1)))
        return np.argmax(action[0]), np.argmax(next_action[0])

    def test(self, name, df, show=True, plot=True):
        with open(f"{self.models_dir}//{name}//config.json", "r") as f:
            config = json.load(f)
        env = getattr(environment, config["env"]["name"])(df, config["env"]["length"], config["env"]["cols"])
        model = TD3.load(f"{self.models_dir}//{name}//model", env)
        env = model.get_env()
        obs, rewards, actions, done = env.reset(), [], [], False
        while not done:
            action, _states = model.predict(obs)
            actions.append(np.argmax(action[0]))
            obs, reward, done, _ = env.step(action)
            rewards.append(float(reward[0]))
        steps = len(actions)
        positive = [num for num in rewards if num > 0]
        negative = [num for num in rewards if num < 0]
        n_positive, n_negative = len(positive), len(negative)
        max_positive, max_negative = round(max([0] + positive), 2), round(min([0] + negative), 2)
        positive_percent = round(n_positive / steps * 100, 2)
        avg_rewards = round(sum(rewards) / len(rewards), 4)
        reward_progress = np.cumsum(rewards).tolist()
        if plot:
            close = df["close"].iloc[config["env"]["length"] - 1:]
            plt.style.use('dark_background')
            fig = plt.figure()
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[2, 1])
            axs = gs.subplots(sharex="col")
            axs[0].plot(close.tolist(), color="#f7fffc")
            if config["env"]["name"] == "TradingEnv":
                for n, action in enumerate(actions):
                    color = "#62ff3b" if action == 1 else "#ff0808" if action == 2 else "#e1dbff"
                    marker = "^" if action == 1 else "v" if action == 2 else ">"
                    axs[0].plot(n, close.iloc[n], marker, color=color, markersize=4)
            elif config["env"]["name"] == "RangeTradingEnv" and plot:
                for n, action in enumerate(actions):
                    colors = ["#000000", "#262222", "#473b3b", "#594545", "#694b4b",
                              "#784e4e", "#875252", "#965353", "#a65353", "#b05151",
                              "#bf4b4b", "#c44343", "#cf3c3c", "#d43333", "#d92929", "#ff0000"]
                    axs[0].plot(n, close.iloc[n], "o", color=colors[action], markersize=4)
            axs[1].plot([reward_progress[-1] / 2] * steps, color="#424242")
            axs[1].plot((np.array(rewards) * 10 + (reward_progress[-1] / 2)).tolist())
            axs[1].plot(reward_progress)
            plt.savefig(f"{self.models_dir}//{name}//figure.png", dpi=200)
            plt.show() if show else ...
            figure = os.path.abspath(f"{self.models_dir}//{name}//figure.png")
        else:
            figure = None
        result = {"steps": steps, "positive_percent": positive_percent,
                  "avg_rewards": avg_rewards, "max_positive": max_positive, "max_negative": max_negative}
        with open(f"{self.models_dir}//{name}//test_result.json", "w") as f:
            json.dump(result, f, indent=4)
        print(f"steps: {steps}\n"
              f"positive_percent: {positive_percent}\n"
              f"avg_rewards: {avg_rewards}\n"
              f"max_positive: {max_positive}\n"
              f"max_negative: {max_negative}\n"
              f"figure: {figure}\n")
        return result

    def get_path(self, name):
        return f"{self.models_dir}//{name}//model"


