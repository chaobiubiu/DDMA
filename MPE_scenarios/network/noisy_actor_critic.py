import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    def __init__(self, in_features, out_features, cuda=False, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda
        self.weight_mean = Parameter(torch.Tensor(out_features, in_features))
        self.weight_mean.data.uniform_(-0.1, 0.1)
        self.weight_std = Parameter(torch.Tensor(out_features, in_features))
        self.weight_std.data.uniform_(-0.1, 0.1)
        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_features))
            self.bias_mean.data.uniform_(-0.1, 0.1)
            self.bias_std = Parameter(torch.Tensor(out_features))
            self.bias_std.data.uniform_(-0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        epsilon_weight = torch.rand([self.out_features, self.in_features])
        epsilon_bias = torch.rand([self.out_features])
        if self.cuda:
            epsilon_weight = epsilon_weight.cuda()
            epsilon_bias = epsilon_bias.cuda()
        self.weight = self.weight_mean + self.weight_std.mul(epsilon_weight)
        self.bias = self.bias_mean + self.bias_std.mul(epsilon_bias)
        return F.linear(input, self.weight, self.bias)


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fc1 = Linear(args.obs_shape[agent_id], 64, args.cuda)
        self.fc2 = Linear(64, 64, args.cuda)
        self.fc3 = Linear(64, 64, args.cuda)
        self.action_out = Linear(64, args.action_shape[agent_id], args.cuda)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fc1 = Linear(self.args.input_dim_self * self.args.n_agents + sum(args.action_shape), 64, args.cuda)
        self.fc2 = Linear(64, 64, args.cuda)
        self.fc3 = Linear(64, 64, args.cuda)
        self.q_out = Linear(64, 1, args.cuda)

    def forward(self, state, action):
        for i in range(len(action)):
            action[i] /= self.max_action
        state = state.permute(1,0,2).reshape(self.args.batch_size, -1)
        action = action.permute(1, 0, 2).reshape(self.args.batch_size, -1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
