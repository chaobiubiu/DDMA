import numpy as np
import torch
from torch.distributions import Categorical
from algorithm.ipg import IPG

class IPG_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = IPG(args, agent_id)

    def select_action(self, o, epsilon):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        probs = self.policy.actor(inputs).squeeze(0)
        probs = (1 - epsilon) * probs + (epsilon / float(self.args.n_actions))
        if epsilon != 0:
            act_sample = Categorical(probs).sample().long()
        else:
            act_sample = torch.argmax(probs)
        # print('{} : {}'.format(self.name, pi))
        u = act_sample.cpu().numpy()
        return u.copy()

    def learn(self, transitions, epsilon):
        self.policy.train(transitions, epsilon)

