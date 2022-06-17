from wrappers.geese_wrapping_3 import GeeseEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def save_model(model, path):
    model.save(path)

def load_model(path):
    model = PPO.load(path)
    return model

def main(output_path, alg, version, name, opponents, extractor, feature_dim, net_arch, steps):
    model_name = f"{alg}_{version}_{name}"
    env = GeeseEnv(debug=False, opponents=opponents)
    m_env = Monitor(env, os.path.join(output_path, model_name), allow_early_resets=True)

    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=dict(features_dim=feature_dim),
        net_arch=net_arch,
        normalize_images=False
    )
    model = PPO('MlpPolicy', m_env, policy_kwargs=policy_kwargs,
                verbose=1, ent_coef=0.01)
    model.learn(total_timesteps=steps)
    save_model(model, os.path.join(output_path, model_name))

    print("Results...")
    df = pd.read_csv(f'{os.path.join(output_path, model_name)}.monitor.csv', header=1, index_col='t')
    df.rename(columns={'r': 'Episode Reward', 'l': 'Episode Length'}, inplace=True)
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, y='Episode Reward', x=df.index, color='blue')
    plt.savefig(f"{os.path.join(output_path, model_name)}.png")