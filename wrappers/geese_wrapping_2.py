import gym
from kaggle_environments import make
from gym import spaces
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, adjacent_positions, \
    row_col, greedy_agent
import numpy as np


class GeeseWrapper:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.center_column = int(columns / 2)
        self.center_row = int(rows / 2)

    def displace_board(self, board, goose_head):
        row_diff = self.center_row - goose_head[0]
        col_diff = self.center_column - goose_head[1]
        board = np.roll(board, row_diff, axis=1)
        board = np.roll(board, col_diff, axis=2)
        return board

    def move_position(self, position, move):
        row = position[0] + move[0]
        column = position[1] + move[1]
        return row, column

    def convert_obs(self, obs):
        player_idx = obs.index
        head_board = np.zeros((self.columns * self.rows), dtype=np.float32)
        body_board = np.zeros((self.columns * self.rows), dtype=np.float32)
        food_board = np.zeros((self.columns * self.rows), dtype=np.float32)
        my_goose = obs.geese[player_idx]
        my_goose_len = len(my_goose)
        if my_goose_len > 0:
            my_goose_head_position = row_col(my_goose[0], self.columns)

            head_board[my_goose[0]] = 0.5
            if my_goose_len > 1:
                my_goose_values = np.linspace(0.9, 0.5, my_goose_len - 1)
                for i in range(1, my_goose_len):
                    body_board[my_goose[i]] = my_goose_values[i - 1]

        for g in range(len(obs.geese)):
            if g == player_idx:
                continue
            opp_goose = obs.geese[g]
            opp_goose_len = len(opp_goose)
            if opp_goose_len > 0:
                head_board[opp_goose[0]] = 1.0
                if opp_goose_len > 1:
                    opp_goose_values = np.linspace(0.9, 0.5, opp_goose_len - 1)
                    for i in range(1, opp_goose_len):
                        body_board[opp_goose[i]] = opp_goose_values[i - 1]

        for f in obs.food:
            f_adj = adjacent_positions(f, self.columns, self.rows)
            has_head = False
            for a in f_adj:
                if head_board[a] == 1.0:
                    has_head = True
                    break
            if has_head:
                food_board[f] = 0.5
            else:
                food_board[f] = 1.0

        head_board = head_board.reshape((1, self.rows, self.columns))
        body_board = body_board.reshape((1, self.rows, self.columns))
        food_board = food_board.reshape((1, self.rows, self.columns))
        if my_goose_len > 0:
            head_board = self.displace_board(head_board, my_goose_head_position)
            body_board = self.displace_board(body_board, my_goose_head_position)
            food_board = self.displace_board(food_board, my_goose_head_position)
        board = np.concatenate((head_board, body_board, food_board))
        return board

    def offset_reward(self, reward, offset):
        return float(reward) - offset - 100.0

    def transform_action(self, action):
        if action == 0:
            return Action.NORTH
        if action == 1:
            return Action.EAST
        if action == 2:
            return Action.WEST
        if action == 3:
            return Action.SOUTH

    def invalid_actions(self, state, prev_act):
        head = [self.center_row, self.center_column]
        invalid_acts = []
        for a in range(4):
            a_trans = self.transform_action(a)
            if prev_act is not None:
                if a_trans == prev_act.opposite():
                    invalid_acts.append(a)
                    continue
            a_move = a_trans.to_row_col()
            a_pos = self.move_position(head, a_move)
            if state[0, a_pos[0], a_pos[1]] == 1.0 or state[1, a_pos[0], a_pos[1]] > 0.5:
                invalid_acts.append(a)
        return invalid_acts

    def validate_action(self, state, action, prev_act):
        invalid_acts = self.invalid_actions(state, prev_act)
        if action not in invalid_acts:
            return action
        else:
            possible_acts = []
            for a in range(4):
                if a not in invalid_acts:
                    possible_acts.append(a)
            if len(possible_acts) > 0:
                action = np.random.choice(possible_acts, 1)
        return action


class GeeseEnv(gym.Env):
    def __init__(self, opponents=[greedy_agent, greedy_agent, greedy_agent], debug=False):
        super().__init__()
        self.num_envs = 1
        self.num_previous_observations = 1
        self.debug = debug
        self.actions = [action for action in Action]
        self.env = make("hungry_geese", debug=self.debug)
        rows = self.env.configuration.rows
        columns = self.env.configuration.columns
        self.trainer = self.env.train([None, *opponents])
        self.config = self.env.configuration
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(3, rows, columns),
                                            dtype=np.float32)

        self.wrapper = GeeseWrapper(columns, rows)
        self.state = None
        self.obs = None
        self.prev_act = None
        self.reward_offset = None

    def step(self, action_idx):
        action = self.wrapper.transform_action(action_idx)
        next_obs, obs_reward, obs_done, obs_info = self.trainer.step(action.name)

        base_reward = 0.1
        collision_reward = 0.0
        lose_reward = 0.0
        win_reward = 0.0
        food_reward = 0.0

        invalid_acts = self.wrapper.invalid_actions(self.state, self.prev_act)
        if action_idx in invalid_acts:
            collision_reward = -1.0

        player_idx = self.obs.index
        my_head = [self.wrapper.center_row, self.wrapper.center_column]

        act_move = action.to_row_col()
        act_pos = self.wrapper.move_position(my_head, act_move)

        if self.debug:
            print("Obs reward = {}".format(obs_reward))
        obs_offset_reward = self.wrapper.offset_reward(obs_reward, self.reward_offset)
        if self.debug:
            print("Offset reward = {}".format(obs_offset_reward))

        next_state = self.wrapper.convert_obs(next_obs)
        active_agents = len(list(filter(lambda x: len(x) > 0, next_obs.geese)))

        if obs_done:
            if self.debug:
                print("Done with active agents = {}".format(active_agents))
            # we died
            if len(next_obs.geese[player_idx]) == 0:
                if self.debug:
                    print("We died")
                lose_reward = -1.0
            # we won
            else:
                if self.debug:
                    print("We won")
                win_reward = 2.0
        else:
            if obs_offset_reward == 1.0:
                if self.debug:
                    print("Found a food")
                # found a food
                food_reward = self.state[2, act_pos[0], act_pos[1]]

        self.state = next_state
        self.obs = next_obs
        self.prev_act = action
        self.reward_offset = 0.0
        reward = base_reward + collision_reward + lose_reward + win_reward + food_reward
        if self.debug:
            print("Result reward = {}".format(reward))

        return self.state, reward, obs_done, obs_info

    def reset(self):
        self.obs = self.trainer.reset()
        self.state = self.wrapper.convert_obs(self.obs)
        self.prev_act = None
        self.reward_offset = 101.0
        return self.state

    def render(self, **kwargs):
        self.env.render(**kwargs)
