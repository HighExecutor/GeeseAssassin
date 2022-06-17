from wrappers.geese_wrapping_2_1 import GeeseEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
from kaggle_environments.envs.hungry_geese.hungry_geese import greedy_agent
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import os.path
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from agents.ppo.ppo_v5.ppo_v5 import Agent as ppo5
# import torch as T


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)



def save_model(model, path):
    model.save(path)

def load_model(path):
    model = PPO.load(path)
    return model

def get_env():
    opp1 = ppo5()
    opp2 = ppo5()
    opp3 = ppo5()
    env = GeeseEnv(debug=False, opponents=[opp1, opp2, opp3])
    m_env = Monitor(env, "./checkpoints/vec/env_{}.monitor.csv".format(np.random.randint(0, 10000)))
    return m_env

def main(output_path, alg, version, name, extractor, feature_dim, net_arch, steps, envs):
    model_name = f"{alg}_{version}_{name}"

    num_envs = envs
    envs = SubprocVecEnv([get_env for i in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=dict(features_dim=feature_dim),
        net_arch=net_arch,
        normalize_images=False
    )
    model_tmp = load_model("E:\share\projects\geese_assassin\\agents\ppo\ppo_v5\ppo_vec_0.0.5")
    model_policy_state_dict = model_tmp.policy.state_dict()
    del model_tmp

    model = PPO('MlpPolicy', envs, policy_kwargs=policy_kwargs,
                n_steps=1024, n_epochs=5, batch_size=32, gamma=0.95,
                verbose=1, ent_coef=0.01, tensorboard_log=os.path.join(output_path, "geese_tb"))
    model.policy.load_state_dict(model_policy_state_dict)
    log_dir = os.path.join(output_path, "vec")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000000, log_dir=log_dir)
    model.learn(total_timesteps=steps, callback=callback, tb_log_name="ppo_vec_log_v6")
    save_model(model, os.path.join(output_path, model_name))
    plot_results([log_dir], steps, results_plotter.X_TIMESTEPS, "PPO vec geese")
    plt.savefig(f"{os.path.join(output_path, model_name)}.png")

