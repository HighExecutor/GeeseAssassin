import sys
sys.path.append(__file__)
from learning.dqn_launch import main as dqn_main
from learning.ppo_launch import main as ppo_main
from learning.ppo_vec_launch import main as ppo_vec_main
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as T
from torch import nn
import gym
import os.path as path
from kaggle_environments.envs.hungry_geese.hungry_geese import greedy_agent
from agents.dqn.dqn_v1.dqn_v001 import Agent as dqn1
from agents.ppo.ppo_v1.ppo_v1 import Agent as ppo1

class Conv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.conv0 = Conv2d(3, 16, 3, 0)
        self.blocks = nn.ModuleList([Conv2d(16, 16, 3) for _ in range(2)])
        self.conv1 = Conv2d(16, 16, 3, 0)
        self.fc1 = nn.Linear(336, 128)


    def forward(self, obs):
        x = T.relu(self.conv0(obs))
        for l in self.blocks:
            x = T.relu(x + l(x))
        x = T.relu(self.conv1(x))
        x = T.flatten(x, start_dim=1)
        x = T.relu(self.fc1(x))
        return x


if __name__ == "__main__":

    alg = "ppo_vec"
    name = ""
    version = "0.0.6"
    # opponents = [greedy_agent, dqn1(), ppo1()]
    opponents = None
    output_path = path.join(".", "checkpoints")
    feature_dim = 128
    net_arch = None
    if "ppo" in alg:
        net_arch = [64, dict(vf=[16], pi=[16])]
    else:
        net_arch = [64, 16]
    envs = 4
    steps = 20000000

    if alg == "ppo_vec":
        ppo_vec_main(output_path, alg, version, name, CustomCNN, feature_dim, net_arch, steps, envs)
    elif alg == "ppo":
        ppo_main(output_path, alg, version, name, opponents, CustomCNN, feature_dim, net_arch, steps)
    elif alg == "dqn":
        dqn_main(output_path, alg, version, name, opponents, CustomCNN, feature_dim, net_arch, steps)
    else:
        raise Exception("Wrong algorithm name")



