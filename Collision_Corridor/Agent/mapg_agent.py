import numpy as np
import torch
from torch.distributions import Categorical
from algorithm.mapg import MAPG

class MAPG_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MAPG(args, agent_id)

    def select_action(self, s, epsilon):
        inputs = torch.tensor(s, dtype=torch.float32)
        input_local = inputs[self.agent_id]     # shape=(obs_shape, )
        input_others = inputs[np.arange(self.args.n_agents) != self.agent_id, :]        # shape=(n_agents-1, obs_shape)
        input_local = input_local.unsqueeze(0)  # shape=(1, obs_shape)
        input_others = input_others.unsqueeze(0).reshape(1, -1)     # shape=(1, (n_agents-1)*obs_shape)
        if self.args.cuda:
            input_local = input_local.cuda()
            input_others = input_others.cuda()
        probs = self.policy.actor(input_local, input_others).squeeze(0)
        if epsilon == 0:
            act_sample = torch.argmax(probs)
        else:
            probs = (1 - epsilon) * probs + (epsilon / float(self.args.n_actions))
            act_sample = Categorical(probs).sample().long()
        u = act_sample.cpu().numpy()
        return u.copy()

    def learn(self, transitions, epsilon, other_agents):
        self.policy.train(transitions, epsilon, other_agents)

