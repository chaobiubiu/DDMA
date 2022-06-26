import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter


class Actor(nn.Module):
    def __init__(self, args, n_h1_self=64, n_h1_others=64, n_h2=64, stage=1):
        super(Actor, self).__init__()
        self.args = args
        self.stage = stage
        self.fc1_self = nn.Linear(args.obs_shape, n_h1_self)
        self.w2_self = Parameter(torch.Tensor(n_h1_self, n_h2))
        if self.stage > 1:
            self.fc1_others = nn.Linear(args.obs_shape, n_h1_others)
            self.w2_others = Parameter(torch.Tensor(n_h1_others, n_h2))
        self.b2 = Parameter(torch.Tensor(n_h2))
        self.fc3 = nn.Linear(n_h2, args.n_actions)
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.w2_self, 0, 0.01)
        nn.init.normal_(self.b2, 0, 0.01)
        if self.stage > 1:
            nn.init.normal_(self.w2_others, 0, 0.01)

    def forward(self, obs_about_self, obs_about_others):
        x1_self = f.relu(self.fc1_self(obs_about_self))
        x2_self = torch.matmul(x1_self, self.w2_self)
        if self.stage > 1:
            x1_others = f.relu(self.fc1_others(obs_about_others))
            x2_others = torch.matmul(x1_others, self.w2_others)
            x3 = f.relu(x2_self + x2_others + self.b2)
        else:
            x3 = f.relu(x2_self + self.b2)
        out = self.fc3(x3)
        probs = f.softmax(out, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, args, n_h1_self=64, n_h1_others=64, n_h2=64, stage=1):
        super(Critic, self).__init__()
        self.args = args
        self.stage = stage
        self.fc1_self = nn.Linear((args.obs_shape + args.n_actions), n_h1_self)
        self.w2_self = Parameter(torch.Tensor(n_h1_self, n_h2))
        if stage > 1:
            self.fc1_others = nn.Linear((args.obs_shape + args.n_actions) * (args.n_agents - 1), n_h1_others)
            self.w2_others = Parameter(torch.Tensor(n_h1_others, n_h2))
        self.q_out = nn.Linear(n_h2, 1)
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.w2_self, 0, 0.01)
        if self.stage > 1:
            nn.init.normal_(self.w2_others, 0, 0.01)

    def forward(self, state_one, act_one, state_others, act_others):
        concat_input = torch.cat([state_one, act_one], dim=-1)
        x1_self = f.relu(self.fc1_self(concat_input))
        x2_self = torch.matmul(x1_self, self.w2_self)
        if self.stage > 1:
            concat_input_others = torch.cat([state_others, act_others], dim=-1)
            x1_others = f.relu(self.fc1_others(concat_input_others))
            x2_others = torch.matmul(x1_others, self.w2_others)
            x3 = f.relu(x2_self + x2_others)
        else:
            x3 = f.relu(x2_self)
        q_value = self.q_out(x3)
        return q_value