import numpy as np
import torch
from torch.distributions import Categorical
from algorithm.overlap import Overlap

class Overlap_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = Overlap(args, agent_id)

    def select_action(self, o, epsilon):
        inputs = torch.tensor(o, dtype=torch.float32)
        input_local = inputs[self.agent_id]
        input_others = inputs[np.arange(self.args.n_agents) != self.agent_id, :]
        input_local = input_local.unsqueeze(0)
        input_others = input_others.unsqueeze(0).reshape(1, -1)
        if self.args.cuda:
            input_local = input_local.cuda()
            input_others = input_others.cuda()

        if self.args.stage > 1 and self.args.use_overlap:
            interact_flag = self.policy.get_flag(torch.cat([input_local, input_others], dim=-1))
            # if self.agent_id == 0:
            #     print('current agent%d'%self.agent_id, interact_flag)
            if interact_flag:
                probs = self.policy.actor(input_local, input_others).squeeze(0)
            else:
                probs = self.policy.expert_actor(input_local, input_others).squeeze(0)
        else:
            probs = self.policy.actor(input_local, input_others).squeeze(0)

        probs = (1 - epsilon) * probs + (epsilon / float(self.args.n_actions))
        if epsilon != 0:
            act_sample = Categorical(probs).sample().long()
        else:
            act_sample = torch.argmax(probs)
        # act_sample = Categorical(probs).sample().long()
        # print('{} : {}'.format(self.name, pi))
        u = act_sample.cpu().numpy()
        return u.copy()

    def learn(self, transitions, epsilon, other_agents):
        self.policy.train(transitions, epsilon, other_agents)

    def update_detec(self, batch_inps, batch_labels, iter=None, logger=None):
        self.policy.update_detec(batch_inps, batch_labels, iter, logger)

    def make_data(self, o_cat, u_cat):
        # obs_inps.shape=(detec_buffer_size, obs_shape*n_agents)
        obs_inps, kl_values = self.policy.output_detec(o_cat, u_cat)
        return obs_inps, kl_values

