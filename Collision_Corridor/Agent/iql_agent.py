import numpy as np
import torch
from algorithm.iql import IQL

class IQL_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = IQL(args, agent_id)

    def select_action(self, o, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.choice(np.arange(self.args.n_actions))
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            q_value = self.policy.q_network(inputs).detach()
            u = torch.argmax(q_value)
            u = u.cpu().numpy()
        return u.copy()

    def learn(self, transitions, logger):
        self.policy.train(transitions, logger)

    def update_detec(self, batch_inps, batch_labels, iter=None, logger=None):
        self.policy.update_detec(batch_inps, batch_labels, iter, logger)

    def make_data(self, o_cat, u_cat):
        obs_inps, kl_values = self.policy.output_detec(o_cat, u_cat)
        return obs_inps, kl_values



