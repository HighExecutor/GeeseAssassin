from stable_baselines3.ppo import PPO
from wrappers.geese_wrapping_4 import GeeseWrapper
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class Agent:
    def __init__(self):
        print("PPO v002")
        self.network = PPO.load("./agents/ppo/ppo_v2/ppo_v2.zip")
        self.wrapper = GeeseWrapper(11, 7)
        self.prev_act = None

    def __call__(self, observation, config):
        if observation.step == 0:
            self.prev_act = Action.NORTH
        state = self.wrapper.convert_obs(observation, self.prev_act)
        action = self.network.predict(state)[0]
        action2 = self.wrapper.validate_action(state, action)
        env_action = self.wrapper.head_action_map[(action2, self.prev_act)]
        self.prev_act = env_action
        return env_action.name


# class Conv2d(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size, padding=1):
#         super().__init__()
#         self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)
#         self.bn = nn.BatchNorm2d(output_dim)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x

# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         self.conv0 = Conv2d(3, 16, 3, 0)
#         self.conv1 = Conv2d(16, 32, 3, 0)
#         self.blocks = nn.ModuleList([Conv2d(32, 32, 3) for _ in range(2)])
#         self.conv2 = Conv2d(32, 16, 3, 0)
#         self.fc1 = nn.Linear(400, 128)
#
#
#     def forward(self, obs):
#         x = T.relu(self.conv0(obs))
#         x = T.relu(self.conv1(x))
#         for l in self.blocks:
#             x = T.relu(x + l(x))
#         x = T.relu(self.conv2(x))
#         x = T.flatten(x, start_dim=1)
#         x = T.relu(self.fc1(x))
#         return x


# net_arch = [64, dict(vf=[16], pi=[16])]
