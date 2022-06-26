import numpy as np
import torch
from algorithm.vdn import VDN

class VDN_Agent:
    def __init__(self, args):
        self.args = args
        self.policy = VDN(args)

    def select_action(self, o, agent_id, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.choice(np.arange(self.args.n_actions))
        else:
            inputs = torch.tensor(o, dtype=torch.float32)      # shape=(n_agents, obs_shape)
            input_local = inputs[agent_id]
            input_others = inputs[np.arange(self.args.n_agents) != agent_id, :]
            input_local = input_local.unsqueeze(0)
            input_others = input_others.unsqueeze(0).reshape(1, -1)
            if self.args.cuda:
                input_local = input_local.cuda()
                input_others = input_others.cuda()
            q_value = self.policy.q_utilities[agent_id](input_local, input_others).detach()
            u = torch.argmax(q_value)
            u = u.cpu().numpy()
        return u.copy()

    def learn(self, transitions, logger):
        self.policy.train(transitions, logger)



