from stable_baselines3.dqn import DQN
from wrappers.geese_wrapping_3 import GeeseWrapper
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import numpy as np


class Agent:
    def __init__(self):
        print("DQN v001")
        self.network = DQN.load("./agents/dqn/dqn_v1/heads_dqn_test.zip")
        self.network.policy.eval()
        self.wrapper = GeeseWrapper(11, 7)
        self.prev_act = None

    def __call__(self, observation, config):
        if observation.step == 0:
            self.prev_act = Action.NORTH
        state = self.wrapper.convert_obs(observation, self.prev_act)
        action = self.network.policy.predict(state)[0]
        action2 = self.wrapper.validate_action(state, action)
        env_action = self.wrapper.head_action_map[(action2, self.prev_act)]
        self.prev_act = env_action
        return env_action.name