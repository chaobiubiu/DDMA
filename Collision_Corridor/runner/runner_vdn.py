import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.vdn_agent import VDN_Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        # self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / (self.args.max_time_steps / 5))
        # Only for qmix
        self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / self.args.max_time_steps)
        self.episode_limit = args.episode_len
        self.env = env
        self.agent = VDN_Agent(args)
        self.buffer = Buffer(args)

        if args.single_map:
            self.save_path = self.args.save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + (
                        'order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + (
                        'order_%d' % self.args.order)
            self.log_path = 'runs/hallway_1agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        else:
            self.save_path = self.args.save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + (
                        'order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + (
                        'order_%d' % self.args.order)
            self.log_path = 'runs/hallway_2agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)
        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

    def run(self):
        reward_eval = []
        s = self.env.reset()
        done = False
        for time_step in tqdm(range(1, self.args.max_time_steps + 1)):
            if time_step % self.episode_limit == 0 or done:
                s = self.env.reset()
                done = False
            u = []
            actions = []
            with torch.no_grad():
                for agent_id in range(self.args.n_agents):
                    action = self.agent.select_action(s, agent_id, self.epsilon)
                    u.append(action)
                    actions.append(action)
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s, u, r, s_next, done)
            s = s_next

            if self.buffer.current_size >= self.args.batch_size and time_step % self.args.steps_per_train == 0:
                transitions = self.buffer.sample(self.args.batch_size)
                self.agent.learn(transitions, self.logger)

            if time_step % self.args.evaluate_rate == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('T * %d' % self.args.evaluate_rate)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

        self.agent.policy.save_model(self.model_save_path, train_step=None)
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
                    for agent_id in range(self.args.n_agents):
                        action = self.agent.select_action(s, agent_id, 0)
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