import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.madqn_agent import MADQN_Agent
from common.replay_buffer import Buffer
from common.detec_buffer import DetecBuffer
from common.utils import handle_samples
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / (self.args.max_time_steps / 5))
        self.episode_limit = args.episode_len
        self.train_from_nothing = args.train_from_nothing
        self.use_overlap = args.use_overlap
        self.stage = args.stage
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        if args.stage > 1 and args.use_overlap:
            self.detec_buffer = DetecBuffer(args)

        if args.single_map:
            self.save_path = self.args.save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + ('order_%d'%self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_1agent' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_load_path = self.args.model_save_dir + '/hallway_1agent' + self.args.algorithm + '/stage_1'
            self.log_path = 'runs/hallway_1agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        else:
            self.save_path = self.args.save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_2agent' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_load_path = self.args.model_save_dir + '/hallway_2agent' + self.args.algorithm + '/stage_1'
            self.log_path = 'runs/hallway_2agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)
        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

        if self.stage > 1 and not self.train_from_nothing:
            for agent in self.agents:
                agent.policy.load_pretrained_model(self.model_load_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = MADQN_Agent(i, self.args)
            agents.append(agent)
        return agents

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
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.epsilon)
                    u.append(action)
                    actions.append(action)
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s, u, r, s_next, done)
            if hasattr(self, 'detec_buffer'):
                self.detec_buffer.store_episode(s[:self.args.n_agents], u)
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents, self.logger)

            if time_step % self.args.detec_training_rate == 0 and self.stage > 1 and self.use_overlap:
                all_samples = self.detec_buffer.sample_all()
                # shape=(n_agents, detec_buffer_size, o_dim/u_dim)
                o_cat, u_cat = handle_samples(all_samples, self.args.n_agents)
                for i, agent in enumerate(self.agents):
                    obs_inputs, kl_values = agent.make_data(o_cat, u_cat)
                    self.detec_buffer.get_labels(obs_inputs, kl_values, i)
                # print('threshold', self.detec_buffer.threshold)
                for iter in range(self.args.detec_num_updates):
                    for i in range(self.args.n_agents):
                        batch_inps, batch_labels = self.detec_buffer.sample(self.args.detec_batch_size, i)
                        if iter == (self.args.detec_num_updates - 1) and self.logger is not None:
                            self.agents[i].update_detec(batch_inps, batch_labels, (time_step // self.args.detec_training_rate), logger=self.logger)
                        else:
                            self.agents[i].update_detec(batch_inps, batch_labels, None, None)

            if time_step % self.args.evaluate_rate == 0:
                reward_eval.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(reward_eval)), reward_eval)
                plt.xlabel('T * %d' % self.args.evaluate_rate)
                plt.ylabel('Average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

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
