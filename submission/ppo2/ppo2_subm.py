import torch as T
from torch import nn
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, adjacent_positions, \
    row_col
import numpy as np
from torch.distributions import Categorical
import pickle

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
                opp_head_adj = adjacent_positions(opp_goose[0], self.columns, self.rows)
                for a in opp_head_adj:
                    if head_board[a] == 0.0:
                        head_board[a] = 0.1
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

class Conv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv2d(3, 16, 3, 0)
        self.conv1 = Conv2d(16, 32, 3, 0)
        self.blocks = nn.ModuleList([Conv2d(32, 32, 3) for _ in range(2)])
        self.conv2 = Conv2d(32, 16, 3, 0)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 3)


    def forward(self, obs):
        x = T.relu(self.conv0(obs))
        x = T.relu(self.conv1(x))
        for l in self.blocks:
            x = T.relu(x + l(x))
        x = T.relu(self.conv2(x))
        x = T.flatten(x, start_dim=1)
        x = T.relu(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        x = self.fc4(x)
        print(x)
        dist = Categorical(logits=x)
        act = dist.sample()
        return act


weights = pickle.loads(weights_s)


network = Network()
network.load_state_dict(weights)
network.to('cpu')
network.eval()
wrapper = GeeseWrapper(11, 7)
prev_act = None

def agent(observation, config):
    print(observation.step)
    global prev_act
    if observation.step == 0 or prev_act is None:
        prev_act = Action.NORTH
    state = wrapper.convert_obs(observation, prev_act)
    state_t = T.tensor(state).view(1, 3, 11, 11)
    action = network.forward(state_t)[0].item()
    print(action)
    action2 = wrapper.validate_action(state, action)
    print(action2)
    env_action = wrapper.head_action_map[(action2, prev_act)]
    prev_act = env_action
    return env_action.name