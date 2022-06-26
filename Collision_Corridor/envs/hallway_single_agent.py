import numpy as np

class SingleHallway:
    def __init__(self, args):
        n_agents = 1
        rows, cols = 5, 7
        self.obs_shape = 5 + 7     # one-hot encoding
        args.obs_shape = 5 + 7
        self.state_shape = 5 + 7      # one-hot encoded positions of all agents, except all targets
        self.action_shape = 5
        self.episode_limit = args.episode_len
        self.n_agents = n_agents
        args.n_agents = n_agents
        self.n_actions = 5
        args.n_actions = 5
        self.rows = rows
        self.cols = cols
        # {0: stop, 1: up, 2: down, 3: left, 4: right}
        self.action_space = [0, 1, 2, 3, 4]
        self.state = None
        self.obs = None
        self.array = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.record = False
        self.index = None
        self.sparse = True
        self.collide = True
        self.hallway = [2, 3]
        self.stop, self.up, self.down, self.left, self.right = np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]),\
                                                               np.array([0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 0, 1])

    def reset(self):
        # the initial positions of all agents and targets
        # initial_positions = [[0, 0], [0, 6]]
        # initial_targets = [[4, 6], [4, 0]]
        # ind = np.random.choice(len(initial_positions))
        # self.index = initial_positions[ind]
        # self.targets = initial_targets[ind]
        # order_0 for agent 0
        self.index = [0, 0]
        self.targets = [4, 6]
        # order_1 for agent 1
        # self.index = [0, 6]
        # self.targets = [4, 0]
        self._update_obs()
        return self.get_obs()

    def _update_obs(self):
        if self.record:
            for i in range(self.n_agents):
                self.array[self.index[i][0]][self.index[i][1]] += 1

        obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]

        obs_1[0][self.index[0]] = 1
        obs_1[1][self.index[1]] = 1
        obs_1 = obs_1[0] + obs_1[1]

        # self.state should contain the position of the target
        self.state = obs_1
        self.obs = [obs_1]

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def save(self, num, path):
        np.save(path + '/plt_hot_{}'.format(num), self.array)

    def step(self, actions):
        # {0: stop, 1: up, 2: down, 3: left, 4: right}
        # 如果输入的actions[0]是one_hot编码的，则先将其转换
        # normal_actions = []
        # for action in actions:
        #     if (action==self.stop).all():
        #         normal_actions.append(0)
        #     elif (action==self.up).all():
        #         normal_actions.append(1)
        #     elif (action == self.down).all():
        #         normal_actions.append(2)
        #     elif (action == self.left).all():
        #         normal_actions.append(3)
        #     elif (action == self.right).all():
        #         normal_actions.append(4)
        normal_actions = actions[0]
        done = False
        # rewards = [0.0] * self.n_agents  # [0.0, 0.0, 0.0]
        reward = 0.0

        # 如果有一agent执行stop动作，则其不移动。
        if normal_actions == 1:       # up
            if self.index[0] != 0:
                if not (self.index[0] == 2 and self.index[1] == 3):
                    self.index[0] -= 1
        elif normal_actions == 2:       # down
            if self.index[0] != 4:
                if not (self.index[0] == 2 and self.index[1] == 3):
                    self.index[0] += 1
        elif normal_actions == 3:       # left
            if self.index[1] != 0:
                if not (self.index[1] == 4 and self.index[0] in [0, 1, 3, 4]):
                    self.index[1] -= 1
        elif normal_actions == 4:       # right
            if self.index[1] != 6:
                if not (self.index[1] == 2 and self.index[0] in [0, 1, 3, 4]):
                    self.index[1] += 1

        self._update_obs()
        if not self.sparse:
            reward1 = np.sqrt(
                (self.index[0] - self.targets[0]) ** 2 + (self.index[1] - self.targets[1]) ** 2)
            assistant_reward = reward1 * (-1)
            reward += assistant_reward * 0.1

        if self.index == self.targets:
            done = True
            reward += 30
        rewards = [reward for _ in range(self.n_agents)]
        info = {}
        return self.get_obs(), rewards, done, info


