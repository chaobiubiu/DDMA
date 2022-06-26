import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.overlap_agent import Overlap_Agent
from common.detec_buffer import DetecBuffer
from common.replay_buffer import Buffer
from common.utils import handle_samples
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate    # default 0.1 noise rate sampling from a standard normal distribution
        self.epsilon = args.epsilon
        self.epsilon_div = float(0.05 / (self.args.training_episodes * self.args.episode_len))
        self.episode_limit = args.episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.detec_buffer = DetecBuffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
        self.model_save_path = self.args.model_save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = 'runs/' + self.args.scenario_name + '/' + self.args.algorithm + '/' + ('order%d'%self.args.order)
        if self.args.log:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Overlap_Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        reward_eval = []
        for episode in tqdm(range(1, self.args.training_episodes + 1)):
            s = self.env.reset()
            for time_step in range(1, self.episode_limit + 1):
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
                self.detec_buffer.store_episode(s[:, self.args.n_agents], u)
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)  # 这里的做法是方便q_target的计算，q_target中所有agent的action是根据当前各agent policy sample而来

                self.noise = max(0.05, self.noise - self.epsilon_div)
                self.epsilon = max(0.05, self.epsilon - self.epsilon_div)

            if episode % self.args.detec_training_rate == 0:
                all_samples = self.detec_buffer.sample_all()
                # shape=(n_agents, detec_buffer_size, o_dim/u_dim)
                o_cat, u_cat = handle_samples(all_samples, self.args.n_agents)
                for i, agent in enumerate(self.agents):
                    obs_inputs, kl_values = agent.make_data(o_cat, u_cat)
                    self.detec_buffer.get_labels(obs_inputs, kl_values, i)
                for iter in range(self.args.detec_num_updates):
                    for i in range(self.args.n_agents):
                        batch_inps, batch_labels = self.detec_buffer.sample(self.args.detec_batch_size, i)
                        if iter == (self.args.detec_num_updates - 1) and self.logger is not None:
                            self.agents[i].update_detec(batch_inps, batch_labels, (episode // self.args.detec_training_rate), logger=self.logger)
                        else:
                            self.agents[i].update_detec(batch_inps, batch_labels, None, None)

            if episode % self.args.evaluate_rate == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('Episode * %d' % self.args.evaluate_rate)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

        for agent in self.agents:
            agent.policy.save_model(self.model_save_path, train_step=None)
        # saves final episode reward for plotting training curve later
        np.save(self.save_path + '/reward_eval', reward_eval)

        if self.logger is not None:
            self.logger.close()
        print('...Finished training.')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
