import torch
import torch.nn as nn
import torch.nn.functional as f

class Detector(nn.Module):
    def __init__(self, args, agent_id, norm_in=False):
        super(Detector, self).__init__()
        self.args = args
        if norm_in:
            self.in_fn = nn.BatchNorm1d(args.obs_shape[agent_id])
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, obs_inps):
        obs_i = self.in_fn(obs_inps)
        x1 = f.relu(self.fc1(obs_i))
        x2 = f.relu(self.fc2(x1))
        out = self.fc3(x2)
        out_prob = f.softmax(out, dim=1)
        return out, out_prob