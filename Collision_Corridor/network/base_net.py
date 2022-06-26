import torch
import torch.nn as nn
import torch.nn.functional as f

class MLP(nn.Module):
    def __init__(self, args, stage=1):
        super(MLP, self).__init__()
        self.stage = stage
        self.fc1 = nn.Linear(args.obs_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, obs):
        x1 = f.relu(self.fc1(obs))
        x2 = f.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3

class MAMLP(nn.Module):
    def __init__(self, args):
        super(MAMLP, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape + args.obs_shape + args.n_actions, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, obs_i, obs_others, act_others):
        concat_inps = torch.cat([obs_i, obs_others, act_others], dim=1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3

class MAActor(nn.Module):
    def __init__(self, args):
        super(MAActor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, obs_i, obs_others):
        concat_inps = torch.cat([obs_i, obs_others], dim=-1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        probs = f.softmax(out, dim=-1)
        return probs

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, obs):
        # concat_inps = torch.cat([obs_i, obs_others], dim=-1)
        x1 = f.relu(self.fc1(obs))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        probs = f.softmax(out, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear((args.obs_shape + args.n_actions), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs_i, act_i):
        concat_inps = torch.cat([obs_i, act_i], dim=-1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out

class CentralCritic(nn.Module):
    def __init__(self, args):
        super(CentralCritic, self).__init__()
        self.fc1 = nn.Linear((args.obs_shape + args.n_actions) * args.n_agents, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs_all, act_all):
        concat_inps = torch.cat([obs_all, act_all], dim=-1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out

class Detector(nn.Module):
    def __init__(self, args, norm_in=False):
        super(Detector, self).__init__()
        self.args = args
        if norm_in:
            self.in_fn = nn.BatchNorm1d(args.input_shape)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(args.input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, obs_inps):
        obs_i = self.in_fn(obs_inps)
        x1 = f.relu(self.fc1(obs_i))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        out_prob = f.softmax(out, dim=1)
        return out, out_prob

class utility_network(nn.Module):
    def __init__(self, args):
        super(utility_network, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def forward(self, obs_i, obs_others):
        concat_inps = torch.cat([obs_i, obs_others], dim=-1)
        x1 = f.relu(self.fc1(concat_inps))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out

class mixing_network(nn.Module):
    def __init__(self, args):
        super(mixing_network, self).__init__()
        self.n_agents = args.n_agents
        self.num_units = 64
        # state = torch.cat([obs_1, obs_2], dim=-1)
        self.hyper_w1 = nn.Linear(args.obs_shape * 2, args.n_agents * 64)
        self.hyper_w2 = nn.Linear(args.obs_shape * 2, 64 * 1)

        self.hyper_b1 = nn.Linear(args.obs_shape * 2, 64)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.obs_shape * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, q_values, state):
        # state.shape=(batch_size,obs_shape*2), q_values.shape=(batch_size, n_agents)
        batch_size = q_values.shape[0]
        q_values = q_values.view(-1, 1, self.n_agents)  # (batch_size, 1, n_agents)

        w1 = torch.abs(self.hyper_w1(state))  # (batch_size, n_agents*num_units)
        b1 = self.hyper_b1(state)  # (batch_size, num_units)

        w1 = w1.view(-1, self.n_agents, self.num_units)  # (batch_size, n_agents, num_units)
        b1 = b1.view(-1, 1, self.num_units)  # (batch_size, 1, num_units)

        hidden = f.elu(torch.bmm(q_values, w1) + b1)  # (batch_size, 1, num_units)

        w2 = torch.abs(self.hyper_w2(state))  # (batch_size, num_units)
        b2 = self.hyper_b2(state)  # (batch_size, 1)

        w2 = w2.view(-1, self.num_units, 1)     # (batch_size, num_units, 1)
        b2 = b2.view(-1, 1, 1)                  # (batch_size, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # (batch_size, 1, 1)
        q_total = q_total.view(batch_size, 1)
        return q_total

class vdn_network(nn.Module):
    def __init__(self, args):
        super(vdn_network, self).__init__()

    def forward(self, q_values):
        # q_values.shape=(batch_size, n_agents)
        q_total = torch.sum(q_values, dim=1, keepdim=True)
        return q_total