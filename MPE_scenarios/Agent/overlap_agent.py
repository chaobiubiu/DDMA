import numpy as np
import torch
from algorithm.overlap import Overlap

class Overlap_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = Overlap(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            # randomly sample an action between high and low bound.
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            obs_about_self = inputs[:, :self.args.input_dim_self]
            obs_about_others = inputs[:, -self.args.input_dim_others:]
            if self.args.stage > 1 and self.args.use_overlap:
                interact_flag = self.policy.get_flag(inputs)
                if interact_flag:
                    pi = self.policy.actor_network(obs_about_self, obs_about_others)
                else:
                    # 这里的inputs需要划分为obs_about_self, obs_about_others.
                    pi = self.policy.expert_actor(obs_about_self, obs_about_others)
                # pi = interact_flag * multi_agent_pi + single_agent_pi * torch.logical_not(interact_flag)
                pi = pi.squeeze(0)
            else:
                pi = self.policy.actor_network(obs_about_self, obs_about_others).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
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

