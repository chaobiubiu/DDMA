import numpy as np

class MultiHallway:
    def __init__(self, args):
        n_agents = 2
        rows, cols = 5, 7
        self.obs_shape = 5 + 7    # one-hot encoding
        args.obs_shape = 5 + 7
        self.state_shape = (5 + 7) * 2      # one-hot encoded positions of all agents, except all targets
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
        self.record = args.record_visitation
        self.index = None
        self.sparse = True
        self.collide = True
        self.evaluate = False
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
        self.index = [[0, 0], [0, 6]]
        self.targets = [[4, 6], [4, 0]]
        self._update_obs()
        return self.get_obs()

    def _update_obs(self):
        if self.record and (not self.evaluate):
            for i in range(self.n_agents):
                self.array[self.index[i][0]][self.index[i][1]] += 1

        obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]

        obs_1[0][self.index[0][0]] = 1
        obs_1[1][self.index[0][1]] = 1
        obs_1 = obs_1[0] + obs_1[1]

        obs_2 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]

        obs_2[0][self.index[1][0]] = 1
        obs_2[1][self.index[1][1]] = 1
        obs_2 = obs_2[0] + obs_2[1]

        # self.state should contain the position of the target
        self.state = obs_1 + obs_2
        self.obs = [obs_1, obs_2]

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def save(self, path):
        np.save(path + '/visitation_hot', self.array)

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
        normal_actions = actions
        done = False
        # rewards = [0.0] * self.n_agents  # [0.0, 0.0, 0.0]
        reward = 0.0
        for agent_id in range(self.n_agents):
            if self.index[agent_id] == self.targets[agent_id]:
                continue
            # 如果有一agent执行stop动作，则其不移动。
            if normal_actions[agent_id] == 0:
                continue
            if normal_actions[agent_id] == 1:       # up
                if self.index[agent_id][0] != 0:
                    if not (self.index[agent_id][0] == 2 and self.index[agent_id][1] == 3):
                        self.index[agent_id][0] -= 1
            elif normal_actions[agent_id] == 2:       # down
                if self.index[agent_id][0] != 4:
                    if not (self.index[agent_id][0] == 2 and self.index[agent_id][1] == 3):
                        self.index[agent_id][0] += 1
            elif normal_actions[agent_id] == 3:       # left
                if self.index[agent_id][1] != 0:
                    if not (self.index[agent_id][1] == 4 and self.index[agent_id][0] in [0, 1, 3, 4]):
                        if not self.collide:
                            self.index[agent_id][1] -= 1
                        else:
                            temp_target = [self.index[agent_id][0], self.index[agent_id][1] - 1]
                            is_crowded = (self.index[0] == self.hallway or self.index[1] == self.hallway)
                            if is_crowded and temp_target == self.hallway:
                                reward -= 10
                            else:
                                self.index[agent_id][1] -= 1
            elif normal_actions[agent_id] == 4:       # right
                if self.index[agent_id][1] != 6:
                    if not (self.index[agent_id][1] == 2 and self.index[agent_id][0] in [0, 1, 3, 4]):
                        if not self.collide:
                            self.index[agent_id][1] += 1
                        else:
                            temp_target = [self.index[agent_id][0], self.index[agent_id][1] + 1]
                            is_crowded = (self.index[0] == self.hallway or self.index[1] == self.hallway)
                            if is_crowded and temp_target == self.hallway:
                                reward -= 10
                            else:
                                self.index[agent_id][1] += 1
        self._update_obs()
        if not self.sparse:
            reward1 = np.sqrt(
                (self.index[0][0] - self.targets[0][0]) ** 2 + (self.index[0][1] - self.targets[0][1]) ** 2)
            reward2 = np.sqrt(
                (self.index[1][0] - self.targets[1][0]) ** 2 + (self.index[1][1] - self.targets[1][1]) ** 2)
            assistant_reward = (reward1 + reward2) * (-1)
            reward += assistant_reward * 0.1

        if self.index[0] == self.targets[0] and self.index[1] == self.targets[1]:
            done = True
            reward += 30
        rewards = [reward for _ in range(self.n_agents)]
        info = {}
        return self.get_obs(), rewards, done, info


