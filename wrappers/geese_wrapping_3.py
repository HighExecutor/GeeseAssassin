import gym
from kaggle_environments import make
from gym import spaces
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, adjacent_positions, \
    row_col, greedy_agent
import numpy as np
import matplotlib.pyplot as plt


class EnvRender:
    def __init__(self):
        plt.ion()
        orig_board = np.zeros(shape=(3, 7, 11)).transpose(1, 2, 0)
        state_board = np.zeros(shape=(3, 11, 11)).transpose(1, 2, 0)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        self.orig_image = self.ax[0].imshow(orig_board)
        self.state_image = self.ax[1].imshow(state_board)
        self.probs_text = self.ax[0].text(3, -1, "[0.1, 0.2, 0.3]")
        self.val_text = self.ax[0].text(6, -2, "200.0")
        self.prev_act_text = self.ax[1].text(4, -1, "North")
        plt.autoscale(True, tight=True)
        plt.show()

    def stop_interactive(self):
        plt.ioff()

    def update_boards(self, state, obs, prev_act, prob, val):
        orig_board = self.orig_board_convert(obs)
        orig_board = orig_board.transpose(1, 2, 0)
        state_board = state.transpose(1, 2, 0)
        self.orig_image.set_data(orig_board)
        self.state_image.set_data(state_board)
        self.probs_text.set_text(str(np.round(prob, 4)))
        self.val_text.set_text(str(np.round(val, 2)))
        if prev_act is not None:
            self.prev_act_text.set_text(prev_act.name)
        plt.pause(0.001)

    def orig_board_convert(self, obs):
        player_idx = obs.index
        my_board = np.zeros((11 * 7), dtype=np.float32)
        opp_board = np.zeros((11 * 7), dtype=np.float32)
        food_board = np.zeros((11 * 7), dtype=np.float32)
        my_goose = obs.geese[player_idx]
        my_goose_len = len(my_goose)
        if my_goose_len > 0:
            my_goose_head_position = row_col(my_goose[0], 11)

            my_board[my_goose[0]] = 1.0
            if my_goose_len > 1:
                my_goose_values = np.linspace(0.9, 0.5, my_goose_len - 1)
                for i in range(1, my_goose_len):
                    my_board[my_goose[i]] = my_goose_values[i - 1]

        for g in range(len(obs.geese)):
            if g == player_idx:
                continue
            opp_goose = obs.geese[g]
            opp_goose_len = len(opp_goose)
            if opp_goose_len > 0:
                opp_board[opp_goose[0]] = 1.0
                if opp_goose_len > 1:
                    opp_goose_values = np.linspace(0.9, 0.5, opp_goose_len - 1)
                    for i in range(1, opp_goose_len):
                        opp_board[opp_goose[i]] = opp_goose_values[i - 1]

        for f in obs.food:
            food_board[f] = 1.0

        my_board = my_board.reshape((1, 7, 11))
        opp_board = opp_board.reshape((1, 7, 11))
        food_board = food_board.reshape((1, 7, 11))
        if my_goose_len > 0:
            my_board = self.displace_board(my_board, my_goose_head_position)
            opp_board = self.displace_board(opp_board, my_goose_head_position)
            food_board = self.displace_board(food_board, my_goose_head_position)
        board = np.concatenate((my_board, opp_board, food_board))
        return board

    def displace_board(self, board, goose_head):
        row_diff = 3 - goose_head[0]
        col_diff = 5 - goose_head[1]
        board = np.roll(board, row_diff, axis=1)
        board = np.roll(board, col_diff, axis=2)
        return board


class GeeseWrapper:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.center_column = int(columns / 2)
        self.center_row = int(rows / 2)
        self.head_action_map = self.make_head_action_map()

    # 0 - left, 1 - forward, 2 - right
    def make_head_action_map(self):
        mapping = dict()
        mapping[(0, Action.NORTH)] = Action.WEST
        mapping[(1, Action.NORTH)] = Action.NORTH
        mapping[(2, Action.NORTH)] = Action.EAST
        mapping[(0, Action.EAST)] = Action.NORTH
        mapping[(1, Action.EAST)] = Action.EAST
        mapping[(2, Action.EAST)] = Action.SOUTH
        mapping[(0, Action.WEST)] = Action.SOUTH
        mapping[(1, Action.WEST)] = Action.WEST
        mapping[(2, Action.WEST)] = Action.NORTH
        mapping[(0, Action.SOUTH)] = Action.EAST
        mapping[(1, Action.SOUTH)] = Action.SOUTH
        mapping[(2, Action.SOUTH)] = Action.WEST
        return mapping

    def displace_board(self, board, goose_head):
        row_diff = self.center_row - goose_head[0]
        col_diff = self.center_column - goose_head[1]
        board = np.roll(board, row_diff, axis=1)
        board = np.roll(board, col_diff, axis=2)
        return board

    def rotation_m(self, prev_act):
        if prev_act == Action.NORTH:
            return 0
        if prev_act == Action.EAST:
            return 1
        if prev_act == Action.WEST:
            return 3
        if prev_act == Action.SOUTH:
            return 2

    def rotate_board(self, board, previous_action):
        board = np.append([board[0, 5:7, :]], board, axis=1)
        board = np.append(board, [board[0, 2:4, :]], axis=1)
        if previous_action is not None:
            k = self.rotation_m(previous_action)
            board = np.rot90(board, k=k, axes=(1, 2))
        return board

    def convert_obs(self, obs, prev_act):
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
            head_board = self.rotate_board(head_board, prev_act)
            body_board = self.rotate_board(body_board, prev_act)
            food_board = self.rotate_board(food_board, prev_act)
        board = np.concatenate((head_board, body_board, food_board))
        return board

    def offset_reward(self, reward, offset):
        return float(reward) - offset - 100.0

    def transform_head_action(self, action, prev_action):
        if prev_action is None:
            return self.head_action_map[(action, Action.NORTH)]
        return self.head_action_map[(action, prev_action)]

    def move_position(self, action):
        if action == 0:
            return (5, 4)
        if action == 1:
            return (4, 5)
        if action == 2:
            return (5, 6)

    def invalid_actions(self, state):
        invalid_acts = []
        for a in range(3):
            a_pos = self.move_position(a)
            if state[0, a_pos[0], a_pos[1]] == 1.0 or state[1, a_pos[0], a_pos[1]] > 0.5:
                invalid_acts.append(a)
        return invalid_acts

    def validate_action(self, state, action):
        invalid_acts = self.invalid_actions(state)
        if action not in invalid_acts:
            return action
        else:
            possible_acts = []
            for a in range(3):
                if a not in invalid_acts:
                    possible_acts.append(a)
            if len(possible_acts) > 0:
                action = np.random.choice(possible_acts, 1)[0]
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
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(3, columns, columns),
                                            dtype=np.float32)

        self.wrapper = GeeseWrapper(columns, rows)
        self.state = None
        self.obs = None
        self.prev_act = None
        self.reward_offset = None

    def step(self, action_idx):
        action = self.wrapper.transform_head_action(action_idx, self.prev_act)
        next_obs, obs_reward, obs_done, obs_info = self.trainer.step(action.name)

        base_reward = 0.1
        collision_reward = 0.0
        lose_reward = 0.0
        win_reward = 0.0
        food_reward = 0.0

        invalid_acts = self.wrapper.invalid_actions(self.state)
        if action_idx in invalid_acts:
            collision_reward = -1.0

        player_idx = self.obs.index
        act_pos = self.wrapper.move_position(action_idx)

        if self.debug:
            print("Obs reward = {}".format(obs_reward))
        obs_offset_reward = self.wrapper.offset_reward(obs_reward, self.reward_offset)
        if self.debug:
            print("Offset reward = {}".format(obs_offset_reward))

        next_state = self.wrapper.convert_obs(next_obs, self.prev_act)
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
        self.prev_act = None
        self.obs = self.trainer.reset()
        self.state = self.wrapper.convert_obs(self.obs, self.prev_act)
        self.reward_offset = 101.0
        return self.state

    def render(self, **kwargs):
        self.env.render(**kwargs)
