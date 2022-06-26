import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.maddpg_agent import Maddpg_Agent
from common.replay_buffer import Buffer

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate    # default 0.1 noise rate sampling from a standard normal distribution
        self.epsilon = args.epsilon
        self.epsilon_div = float(0.05 / (self.args.max_time_steps))
        self.episode_limit = args.episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
        self.model_save_path = self.args.model_save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Maddpg_Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        reward_eval = []
        s = self.env.reset()
        for time_step in tqdm(range(1, self.args.max_time_steps + 1)):
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):    # action selections of all adversaries
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)  # 这里的做法是方便q_target的计算，q_target中所有agent的action是根据当前各agent policy sample而来

            if time_step % self.args.evaluate_rate == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('T * %d' % self.args.evaluate_rate)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            self.noise = max(0.05, self.noise - self.epsilon_div)
            self.epsilon = max(0.05, self.epsilon - self.epsilon_div)

        for agent in self.agents:
            agent.policy.save_model(self.model_save_path, train_step=None)
        # saves final episode reward for plotting training curve later
        np.save(self.save_path + '/reward_eval', reward_eval)

        print('...Finished training.')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(1, self.args.evaluate_episode_len + 1):
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                # rewards += r[0]
                for rew in r:
                    rewards += rew
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
