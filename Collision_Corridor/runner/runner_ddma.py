import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Agent.ddma_agent import DDMA_Agent
from common.detec_buffer import DetecBuffer
from common.replay_buffer import Buffer
from common.utils import handle_samples
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.anneal_epsilon = float((self.epsilon - self.min_epsilon) / (self.args.max_episodes / (self.args.episodes_per_train * 5)))
        self.episode_limit = args.episode_len
        self.stage = args.stage
        self.train_from_nothing = args.train_from_nothing
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        # 测试为什么效果发生波动
        if self.stage > 1 and self.args.use_overlap and (not self.args.evaluate):
            self.detec_buffer = DetecBuffer(args)

        if args.single_map:
            self.save_path = self.args.save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_1agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.log_path = 'runs/hallway_1agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)
        else:
            self.save_path = self.args.save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.model_save_path = self.args.model_save_dir + '/hallway_2agent/' + self.args.algorithm + '/' + ('order_%d' % self.args.order)
            self.log_path = 'runs/hallway_2agent/' + self.args.algorithm + '/' + ('order%d' % self.args.order)

        self.model_load_path = self.args.model_save_dir + '/hallway_2agent/' + self.args.algorithm + '/stage_1'
        if not os.path.exists(self.save_path) and not self.args.evaluate:
            os.makedirs(self.save_path)
        if self.args.log and (not self.args.evaluate):
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.logger = SummaryWriter(self.log_path)
        else:
            self.logger = None

        if self.stage > 1 and (not self.train_from_nothing) and (not self.args.evaluate):
            for agent in self.agents:
                agent.policy.load_pretrained_model(self.model_load_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = DDMA_Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        reward_eval = []
        # epsilon_record = []
        for episode in tqdm(range(1, self.args.max_episodes + 1)):
            s = self.env.reset()
            done = False
            time_step = 1
            while not done:
                u = []
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s, self.epsilon, episode)
                    u.append(action)
                    actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                if time_step + 1 > self.episode_limit:
                    done = True
                self.buffer.store_episode(s, u, r, s_next, done)
                if hasattr(self, 'detec_buffer'):
                    self.detec_buffer.store_episode(s, u)
                s = s_next
                time_step += 1

            if episode % self.args.episodes_per_train == 0:
                for train_epoch in range(self.args.epoches):
                    for agent in self.agents:
                        transitions = self.buffer.sample(self.args.batch_size)
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, self.epsilon, other_agents)
                # Decrease exploration only after training
                # epsilon_record.append(self.epsilon)
                self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)
                self.buffer = Buffer(self.args)

            # 测试为什么会效果出现波动
            if episode % self.args.detec_episodes_per_train == 0 and self.args.stage > 1 and self.args.use_overlap:
                all_samples = self.detec_buffer.sample_all()
                # o_cat/u_cat.shape=(n_agents, batch_size, obs_shape/1)
                o_cat, u_cat = handle_samples(all_samples, self.args.n_agents)
                for i, agent in enumerate(self.agents):
                    obs_inps, kl_values = agent.make_data(o_cat, u_cat)
                    self.detec_buffer.get_labels(obs_inps, kl_values, i)

                for iter in range(self.args.detec_num_updates):
                    for i in range(self.args.n_agents):
                        batch_inps, batch_labels = self.detec_buffer.sample(self.args.detec_batch_size, i)
                        if iter == (self.args.detec_num_updates - 1) and self.logger is not None:
                            self.agents[i].update_detec(batch_inps, batch_labels, (episode // self.args.detec_episodes_per_train), logger=self.logger)
                        else:
                            self.agents[i].update_detec(batch_inps, batch_labels, None, None)

            # 保存一个模型记录pretrained single_agent policy在前期发挥多少作用
            # if episode == 20:
            #     for agent in self.agents:
            #         agent.policy.save_model(self.model_save_path, train_step=500)

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
        # np.save(self.save_path + '/epsilon_record', epsilon_record)
        if self.args.record_visitation:
            self.env.save(self.save_path)

        if self.logger is not None:
            self.logger.close()
        print('...Finished training.')

    def evaluate(self):
        returns = []
        if self.args.record_visitation:
            self.env.evaluate = True
        # 画热力图时使用
        # kl_values_q_one, kl_values_q_two = [], []
        # kl_values_pi_one, kl_values_pi_two = [], []
        # 评估detector性能时使用
        # index_one = []
        # index_two = []
        # interactive_preds_one = []
        # interactive_preds_two = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0.0
            done = False
            time_step = 1
            while not done:
                # print('current_episode_timestep', episode, time_step)
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s, 0, episode)
                        actions.append(action)
                # print('1', self.env.index)

                # 评估detector效果
                # if episode == 0:
                #     for agent_id, agent in enumerate(self.agents):
                #         inputs = torch.tensor(s, dtype=torch.float32)
                #         input_local = inputs[agent_id]
                #         input_others = inputs[np.arange(self.args.n_agents) != agent_id, :]
                #         input_local = input_local.unsqueeze(0)
                #         input_others = input_others.unsqueeze(0).reshape(1, -1)
                #         if self.args.cuda:
                #             input_local = input_local.cuda()
                #             input_others = input_others.cuda()
                #         if self.stage > 1 and self.args.use_overlap:
                #             interact_strength = agent.policy.pred_strength(torch.cat([input_local, input_others], dim=-1)).detach()
                #             if agent_id == 0:
                #                 index_one.append(self.env.index[0])
                #                 interactive_preds_one.append(interact_strength)
                #             elif agent_id == 1:
                #                 index_two.append(self.env.index[1])
                #                 # print('current agent', agent_id, interact_strength)
                #                 interactive_preds_two.append(interact_strength)

                s_next, r, done, info = self.env.step(actions)
                # 画热力图时使用
                # for agent_id, agent in enumerate(self.agents):
                #     kl_values_q, kl_values_pi = agent.policy.get_kl_values(s, actions)
                #     if agent_id == 0 and episode == 0:
                #         kl_values_q_one.append(kl_values_q[0])
                #         kl_values_pi_one.append(kl_values_pi[0])
                #     elif agent_id == 1 and episode == 0:
                #         kl_values_q_two.append(kl_values_q[0])
                #         kl_values_pi_two.append(kl_values_pi[0])
                #     print('current agent', agent_id, kl_values_q, kl_values_pi)
                # print('reward', r)
                rewards += r[0]
                # for rew in r:
                #     rewards += rew
                s = s_next
                time_step += 1
                if time_step > self.args.evaluate_episode_len:
                    done = True

            returns.append(rewards)
            # print('Returns is', rewards)
        # 画热力图时使用
        # np.save(self.save_path + '/kl_values_q_one', kl_values_q_one)
        # np.save(self.save_path + '/kl_values_pi_one', kl_values_pi_one)
        # np.save(self.save_path + '/kl_values_q_two', kl_values_q_two)
        # np.save(self.save_path + '/kl_values_pi_two', kl_values_pi_two)
        # np.save(self.save_path + '/interactive_preds_one', interactive_preds_one)
        # np.save(self.save_path + '/interactive_preds_two', interactive_preds_two)
        average_return = float(sum(returns) / self.args.evaluate_episodes)
        if self.args.record_visitation:
            self.env.evaluate = False
        # print('Average return is', average_return)
        return average_return
