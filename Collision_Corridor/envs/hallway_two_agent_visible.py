import time
import sys
import numpy as np
import tkinter as tk

# red rectangle: agent, black rectangle: obstacle, yellow circle: target
class MultiHallwayVisible(tk.Tk, object):
    def __init__(self, args):
        super(MultiHallwayVisible, self).__init__()
        n_agents = 2
        rows, cols = 5, 7
        self.unit = 60
        self.obs_shape = 5 + 7
        args.obs_shape = 5 + 7
        self.state_shape = self.obs_shape * 2
        self.action_shape = 5
        self.episode_limit = args.episode_len
        self.n_agents = n_agents
        self.n_actions = 5
        args.n_agents = n_agents
        args.n_actions = 5
        self.rows = rows
        self.cols = cols
        # {0: up, 1: down, 2: left, 3: right}
        self.action_space = [0, 1, 2, 3, 4]
        self.state = None
        self.obs = None
        self.sparse = True
        self.array = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.record = False
        self.index = None
        self.collide = True
        self.hallway = [2, 3]
        self.title('HallWay')
        self.geometry('{0}x{1}'.format(self.cols * self.unit, self.rows * self.unit))
        self._build_maze()
        self.stop, self.up, self.down, self.left, self.right = np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), np.array([0, 0, 1, 0, 0]),\
                                                               np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 0, 1])

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.rows * self.unit,
                                width=self.cols * self.unit)
        # create grids
        # 注意create_line方法后坐标列数在前，行数在后
        for c in range(0, self.cols * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.rows * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.rows * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.cols * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([30, 30])

        # create obstacles
        obstacle1_center = origin + np.array([0, self.unit * 3])
        self.obstacle1 = self.canvas.create_rectangle(obstacle1_center[1] - 30, obstacle1_center[0] - 30,
                                                      obstacle1_center[1] + 30, obstacle1_center[0] + 30,
                                                      fill='black')
        obstacle2_center = origin + np.array([self.unit, self.unit * 3])
        self.obstacle2 = self.canvas.create_rectangle(obstacle2_center[1] - 30, obstacle2_center[0] - 30,
                                                      obstacle2_center[1] + 30, obstacle2_center[0] + 30,
                                                      fill='black')
        obstacle3_center = origin + np.array([self.unit * 3, self.unit * 3])
        self.obstacle3 = self.canvas.create_rectangle(obstacle3_center[1] - 30, obstacle3_center[0] - 30,
                                                      obstacle3_center[1] + 30, obstacle3_center[0] + 30,
                                                      fill='black')
        obstacle4_center = origin + np.array([self.unit * 4, self.unit * 3])
        self.obstacle4 = self.canvas.create_rectangle(obstacle4_center[1] - 30, obstacle4_center[0] - 30,
                                                      obstacle4_center[1] + 30, obstacle4_center[0] + 30,
                                                      fill='black')

        # create yellow and red targets corresponding to different agents
        target1_center = origin + np.array([self.unit * 4, self.unit * 6])
        self.target1 = self.canvas.create_oval(target1_center[1] - 15, target1_center[0] - 15,
                                               target1_center[1] + 15, target1_center[0] + 15,
                                               fill='red')
        target2_center = origin + np.array([self.unit * 4, 0])
        self.target2 = self.canvas.create_oval(target2_center[1] - 15, target2_center[0] - 15,
                                               target2_center[1] + 15, target2_center[0] + 15,
                                               fill='yellow')

        agent1_center = origin
        self.agent1 = self.canvas.create_rectangle(agent1_center[1] - 15, agent1_center[0] - 15,
                                                   agent1_center[1] + 15, agent1_center[0] + 15,
                                                   fill='red')
        agent2_center = origin + np.array([0, self.unit * 6])
        self.agent2 = self.canvas.create_rectangle(agent2_center[1] - 15, agent2_center[0] - 15,
                                                   agent2_center[1] + 15, agent2_center[0] + 15,
                                                   fill='yellow')
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.agent1)
        self.canvas.delete(self.agent2)
        origin = np.array([30, 30])
        agent1_center = origin
        self.agent1 = self.canvas.create_rectangle(agent1_center[1] - 15, agent1_center[0] - 15,
                                                   agent1_center[1] + 15, agent1_center[0] + 15,
                                                   fill='red')
        agent2_center = origin + np.array([0, self.unit * 6])
        self.agent2 = self.canvas.create_rectangle(agent2_center[1] - 15, agent2_center[0] - 15,
                                                   agent2_center[1] + 15, agent2_center[0] + 15,
                                                   fill='yellow')
        self.index = [[0, 0], [0, 6]]
        self.targets = [[4, 6], [4, 0]]
        self._update_obs()
        return self.get_obs()

    def _update_obs(self):
        if self.record:
            for i in range(self.n_agents):
                self.array[self.index[i][0]][self.index[i][1]] += 1

        obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
        obs_2 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]

        obs_1[0][self.index[0][0]] = 1
        obs_1[1][self.index[0][1]] = 1
        obs_1 = obs_1[0] + obs_1[1]

        obs_2[0][self.index[1][0]] = 1
        obs_2[1][self.index[1][1]] = 1
        obs_2 = obs_2[0] + obs_2[1]

        # self.state = obs_1 + obs_2 + self.one_hot_target1 + self.one_hot_target2
        self.state = obs_1 + obs_2
        self.obs = [obs_1, obs_2]

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def save(self, num, path):
        np.save(path + '/plot_hot_{}'.format(num), self.array)

    def step(self, actions):
        # {0: stop, 1: up, 2: down, 3: left, 4: right}
        # normal_actions = []
        # for action in actions:
        #     if (action == self.stop).all():
        #         normal_actions.append(0)
        #     elif (action == self.up).all():
        #         normal_actions.append(1)
        #     elif (action == self.down).all():
        #         normal_actions.append(2)
        #     elif (action == self.left).all():
        #         normal_actions.append(3)
        #     elif (action == self.right).all():
        #         normal_actions.append(4)

        normal_actions = actions
        reward = 0.0
        done = False
        obs1 = self.canvas.coords(self.agent1)
        obs2 = self.canvas.coords(self.agent2)
        # delta x, delta y
        temp_pos1 = np.array([0, 0])
        temp_pos2 = np.array([0, 0])
        for agent_id in range(self.n_agents):
            if self.index[agent_id] == self.targets[agent_id]:
                continue
            if normal_actions[agent_id] == 0:
                continue
            if normal_actions[agent_id] == 1:          # up
                if self.index[agent_id][0] != 0:
                    if not (self.index[agent_id][0] == 2 and self.index[agent_id][1] == 3):
                        self.index[agent_id][0] -= 1
                        if obs1[1] > self.unit and agent_id == 0:
                            temp_pos1[1] -= self.unit
                        if obs2[1] > self.unit and agent_id == 1:
                            temp_pos2[1] -= self.unit
            elif normal_actions[agent_id] == 2:        # down
                if self.index[agent_id][0] != 4:
                    if not (self.index[agent_id][0] == 2 and self.index[agent_id][1] == 3):
                        self.index[agent_id][0] += 1
                        if obs1[1] < (self.rows - 1) * self.unit and agent_id == 0:
                            temp_pos1[1] += self.unit
                        if obs2[1] < (self.rows - 1) * self.unit and agent_id == 1:
                            temp_pos2[1] += self.unit
            elif normal_actions[agent_id] == 3:        # left
                if self.index[agent_id][1] != 0:
                    if not (self.index[agent_id][1] == 4 and self.index[agent_id][0] in [0, 1, 3, 4]):
                        if not self.collide:
                            self.index[agent_id][1] -= 1
                            if obs1[0] > self.unit and agent_id == 0:
                                temp_pos1[0] -= self.unit
                            if obs2[0] > self.unit and agent_id == 1:
                                temp_pos2[0] -= self.unit
                        else:
                            temp_target = [self.index[agent_id][0], self.index[agent_id][1] - 1]
                            is_crowded = (self.index[0] == self.hallway or self.index[1] == self.hallway)
                            if not (is_crowded and temp_target == self.hallway):
                                self.index[agent_id][1] -= 1
                                if obs1[0] > self.unit and agent_id == 0:
                                    temp_pos1[0] -= self.unit
                                if obs2[0] > self.unit and agent_id == 1:
                                    temp_pos2[0] -= self.unit
                            else:
                                print('Collision occurs')
                                reward -= 10
            elif normal_actions[agent_id] == 4:        # right
                if self.index[agent_id][1] != 6:
                    if not (self.index[agent_id][1] == 2 and self.index[agent_id][0] in [0, 1, 3, 4]):
                        if not self.collide:
                            self.index[agent_id][1] += 1
                            if obs1[0] < self.unit * (self.cols - 1) and agent_id == 0:
                                temp_pos1[0] += self.unit
                            if obs2[0] < self.unit * (self.cols - 1) and agent_id == 1:
                                temp_pos2[0] += self.unit
                        else:
                            temp_target = [self.index[agent_id][0], self.index[agent_id][1] + 1]
                            is_crowded = (self.index[0] == self.hallway or self.index[1] == self.hallway)
                            if not (is_crowded and temp_target == self.hallway):
                                self.index[agent_id][1] += 1
                                if obs1[0] < self.unit * (self.cols - 1) and agent_id == 0:
                                    temp_pos1[0] += self.unit
                                if obs2[0] < self.unit * (self.cols - 1) and agent_id == 1:
                                    temp_pos2[0] += self.unit
                            else:
                                print('Collision occurs')
                                reward -= 10
        self._update_obs()
        # 这里给我的感觉是 coords() 这个方法返回是[纵坐标，横坐标]？待会回来修改一下
        self.canvas.move(self.agent1, temp_pos1[0], temp_pos1[1])
        self.canvas.move(self.agent2, temp_pos2[0], temp_pos2[1])
        time.sleep(0.5)
        # next state
        obs1_next = self.canvas.coords(self.agent1)
        obs2_next = self.canvas.coords(self.agent2)
        if not self.sparse:
            reward1 = np.sqrt(
                (self.index[0][0] - self.targets[0][0]) ** 2 + (self.index[0][1] - self.targets[0][1]) ** 2)
            reward2 = np.sqrt(
                (self.index[1][0] - self.targets[1][0]) ** 2 + (self.index[1][1] - self.targets[1][1]) ** 2)
            assistant_reward = (reward1 + reward2) * (-1)
            reward += assistant_reward * 0.1
        # reward function
        # if obs1_next == self.canvas.coords(self.target1) and obs2_next == self.canvas.coords(self.target2):
        if self.index[0] == self.targets[0] and self.index[1] == self.targets[1]:
            reward += 30
            done = True
        rewards = [reward, reward]
        info = {}
        return self.get_obs(), rewards, done, info

    def render(self):
        time.sleep(0.5)
        self.update()


if __name__ == '__main__':
    env = Hallway()
    for i in range(10):
        obs1 = env.reset()
        step = 0
        while True:
            env.render()
            actions = []
            # for i in range(2):
            #     avail_actions = env.get_avail_agent_actions(i)
            #     action = np.random.choice(avail_actions, 1)
            #     actions.append(action)
            if step == 0:
                actions = [1 ,1]
            elif step == 1:
                actions = [1, 1]
            elif step == 2:
                actions = [3, 2]
            elif step == 3:
                actions = [3, 2]
            elif step == 4:
                actions = [3, 2]
            elif step == 5:
                actions = [3, 2]
            elif step == 6:
                actions = [3, 2]
            elif step == 7:
                actions = [3, 2]
            elif step == 7:
                actions = [1, 1]
            elif step == 8:
                actions = [1, 1]
            elif step == 9:
                actions = [0, 1]
            else:
                for i in range(2):
                    avail_actions = env.get_avail_agent_actions(i)
                    action = np.random.choice(avail_actions, 1)
                    actions.append(action)
            print('current step', step)
            r, done, info = env.step(actions)
            step += 1
            if done:
                break



