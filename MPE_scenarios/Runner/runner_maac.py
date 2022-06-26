import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.maac_agent import MAAC_Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate    # default 0.1 noise rate sampling from a standard normal distribution
        self.epsilon = args.epsilon
        self.epsilon_div = float(0.05 / (self.args.max_time_steps / 10))
        self.episode_limit = args.episode_len
        self.stage = args.stage
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
        self.model_save_path = self.args.model_save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)

        self.log_path = 'runs/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = MAAC_Agent(i, self.args)
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
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon, time_step)
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
                    # 这里的做法是方便q_target的计算，q_target中所有agent的action是根据当前各agent policy sample而来
                    agent.learn(transitions, other_agents)

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

        if self.logger is not None:
            self.logger.close()
        print('...Finished training.')

    def evaluate(self):
        returns = []
        for episode in range(1, self.args.evaluate_episodes + 1):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(1, self.args.evaluate_episode_len + 1):
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0, time_step)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                # for rew in r:
                #     rewards += rew
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        average_return = float(sum(returns) / self.args.evaluate_episodes)
        return average_return
