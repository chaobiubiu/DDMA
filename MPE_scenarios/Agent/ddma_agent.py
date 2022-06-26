import numpy as np
import torch
from algorithm.ddma import DDMA

class DDMA_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = DDMA(args, agent_id)

    def select_action(self, o, noise_rate, epsilon, time_step):
        if np.random.uniform() < epsilon:
            # randomly sample an action between high and low bound.
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            obs_about_self = inputs[:, :self.args.input_dim_self]
            obs_about_others = inputs[:, self.args.input_dim_self:]
            '''这里不再考虑interact_flag判断当前状态处是否为交互状态，而是根据网络的概率输出判断当前的交互强度，之后根据该交互强度来考虑探索问题'''
            # if self.args.stage > 1 and self.args.use_overlap:
            #     interact_flag = self.policy.get_flag(inputs)
            #     if interact_flag:
            #         pi = self.policy.actor_network(obs_about_self, obs_about_others)
            #     else:
            #         # 这里的inputs需要划分为obs_about_self, obs_about_others.
            #         pi = self.policy.expert_actor(obs_about_self, obs_about_others)
            #     # pi = interact_flag * multi_agent_pi + single_agent_pi * torch.logical_not(interact_flag)
            #     pi = pi.squeeze(0)
            # else:
            #     pi = self.policy.actor_network(obs_about_self, obs_about_others).squeeze(0)
            pi = self.policy.actor_network(obs_about_self, obs_about_others).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            if epsilon == 0 and noise_rate == 0:
                # evaluation period
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            else:
                if self.args.stage > 1 and self.args.use_overlap:
                    # training period
                    interact_strength = self.policy.pred_strength(inputs)
                    interact_strength = interact_strength.cpu().numpy()
                    # 如果预测出此状态的交互概率大于0.5则需要探索，否则就不探索该状态
                    explor_strength = np.max([0, interact_strength - 0.5])
                    if explor_strength != 0:
                        # 如果对于当前交互状态预测的交互概率越大，这个地方的探索力度越大
                        explor_strength = interact_strength * 2
                    exploration_rate = np.max([self.args.final_rate, self.args.initial_rate +
                                           (self.args.final_rate - self.args.initial_rate) * time_step * 2 / self.args.max_time_steps])
                    noise = exploration_rate * np.abs(explor_strength) * np.random.randn(*u.shape) * self.args.high_action
                    # if time_step % 500 == 0:
                    #     print(u, noise)
                else:
                    noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)

        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

    def update_detec(self, batch_inps, batch_labels, iter=None, logger=None):
        self.policy.update_detec(batch_inps, batch_labels, iter, logger)

    def make_data(self, o_cat, u_cat):
        obs_inps, kl_values = self.policy.output_detec(o_cat, u_cat)
        return obs_inps, kl_values

