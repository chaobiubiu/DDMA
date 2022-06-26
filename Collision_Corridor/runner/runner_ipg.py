import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.ipg_agent import IPG_Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / (self.args.max_episodes / (self.args.episodes_per_train)))
        self.episode_limit = args.episode_len
        self.stage = args.stage
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)

        if args.single_map:
            self.save_path = self.args.save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.log_path = 'runs/hallway_1agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        else:
            self.save_path = self.args.save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.log_path = 'runs/hallway_2agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)
        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = IPG_Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        reward_eval = []
        for episode in tqdm(range(1, self.args.max_episodes + 1)):
            s = self.env.reset()
            done = False
            time_step = 1
            while not done:
                u = []
                actions = []
                for agent_id ,agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.epsilon)
                    u.append(action)
                    actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                if time_step + 1 > self.episode_limit:
                    done = True
                self.buffer.store_episode(s, u, r, s_next, done)
                s = s_next
                time_step += 1

            if episode % self.args.episodes_per_train == 0:
                for train_epoch in range(self.args.epoches):
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        agent.learn(transitions, self.epsilon)
                # Decrease exploration only after training
                self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)
                self.buffer = Buffer(self.args)

            if episode % self.args.evaluate_period == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('Episode * %d' % self.args.evaluate_period)
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
            done = False
            time_step = 1
            while not done:
                # print('current_episode', episode, time_step)
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0)
                        actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                # for rew in r:
                #     rewards += rew
                s = s_next
                time_step += 1
                if time_step > self.args.evaluate_episode_len:
                    done = True
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
