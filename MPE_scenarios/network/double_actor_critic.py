import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Actor(nn.Module):
    # hidden_dim: 64 originally
    def __init__(self, args, agent_id, n_h1_self=64, n_h1_others=64, n_h2=64, stage=1):
        super(Actor, self).__init__()
        self.args = args
        self.stage = stage
        self.max_action = args.high_action
        self.fc1_self = nn.Linear(args.input_dim_self, n_h1_self)
        self.fc2_self = nn.Linear(n_h1_self, n_h1_self)
        self.w3_self = Parameter(torch.Tensor(n_h1_self, n_h2))
        if self.stage > 1:
            self.fc1_others = nn.Linear(args.input_dim_others, n_h1_others)
            self.fc2_others = nn.Linear(n_h1_others, n_h1_others)
            self.w3_others = Parameter(torch.Tensor(n_h1_others, n_h2))
        self.b3 = Parameter(torch.Tensor(n_h2))
        self.action_out = nn.Linear(n_h2, args.action_shape[agent_id])
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.w3_self, 0, 0.01)
        nn.init.normal_(self.b3, 0, 0.01)
        if self.stage > 1:
            nn.init.normal_(self.w3_others, 0, 0.01)

    def forward(self, obs_about_self, obs_about_others):
        x1_self = F.relu(self.fc1_self(obs_about_self))
        x2_self = F.relu(self.fc2_self(x1_self))
        x3_self = torch.matmul(x2_self, self.w3_self)
        if self.stage > 1:
            x1_others = F.relu(self.fc1_others(obs_about_others))
            x2_others = F.relu(self.fc2_others(x1_others))
            x3_others = torch.matmul(x2_others, self.w3_others)
            x3 = F.relu(x3_self + x3_others + self.b3)
        else:
            x3 = F.relu(x3_self + self.b3)
        actions = self.max_action * torch.tanh(self.action_out(x3))
        return actions

class Critic(nn.Module):
    # originally 64
    def __init__(self, args, agent_id, n_h1_self=64, n_h1_others=64, n_h2=64, stage=1):
        super(Critic, self).__init__()
        self.args = args
        self.stage = stage
        self.max_action = args.high_action
        # self.fc1_self = nn.Linear((args.obs_shape[agent_id] + args.action_shape[agent_id]), n_h1_self)
        self.fc1_self = nn.Linear((args.input_dim_self + args.action_shape[agent_id]), n_h1_self)
        self.fc2_self = nn.Linear(n_h1_self, n_h1_self)
        self.w3_self = Parameter(torch.Tensor(n_h1_self, n_h2))
        if stage > 1:
            self.fc1_others = nn.Linear((args.input_dim_self * (args.n_agents - 1) + sum(args.action_shape) - args.action_shape[agent_id]), n_h1_others)
            self.fc2_others = nn.Linear(n_h1_others, n_h1_others)
            self.w3_others = Parameter(torch.Tensor(n_h1_others, n_h2))
        self.q_out = nn.Linear(n_h2, 1)
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.w3_self, 0, 0.01)
        if self.stage > 1:
            nn.init.normal_(self.w3_others, 0, 0.01)

    def forward(self, state_one, act_one, state_others, act_others):
        act_one /= self.max_action
        concat_input = torch.cat([state_one, act_one], dim=-1)
        x1_self = F.relu(self.fc1_self(concat_input))
        x2_self = F.relu(self.fc2_self(x1_self))
        x3_self = torch.matmul(x2_self, self.w3_self)
        if self.stage > 1:
            for i in range(len(act_others)):
                act_others[i] /= self.max_action
            if not self.args.use_overlap:
                state_others = state_others.permute(1, 0, 2).reshape(self.args.batch_size, -1)
                act_others = act_others.permute(1, 0, 2).reshape(self.args.batch_size, -1)
            concat_input_others = torch.cat([state_others, act_others], dim=-1)
            x1_others = F.relu(self.fc1_others(concat_input_others))
            x2_others = F.relu(self.fc2_others(x1_others))
            x3_others = torch.matmul(x2_others, self.w3_others)
            x3 = F.relu(x3_self + x3_others)
        else:
            x3 = F.relu(x3_self)
        q_value = self.q_out(x3)
        return q_value

class Attention_Critic(nn.Module):
    def __init__(self, args, agent_id, n_h1_self=64, n_h1_others=128, n_h2=64, stage=1, attention_heads=4):
        super(Attention_Critic, self).__init__()
        self.args = args
        self.stage = stage
        self.n_agents = args.n_agents
        self.max_action = args.high_action
        self.fc1_self = nn.Linear((args.obs_shape[agent_id] + args.action_shape[agent_id]), n_h1_self)
        self.fc2_self = nn.Linear(n_h1_self, n_h1_self)
        self.w3_self = Parameter(torch.Tensor(n_h1_self, n_h2))
        if stage > 1:
            self.fc1_others = nn.ModuleList()
            for i in range(self.n_agents - 1):
                self.fc1_others.append(nn.Sequential(nn.Linear((sum(args.obs_shape) + sum(args.action_shape) - args.obs_shape[agent_id]
                                         + args.action_shape[agent_id]) // (self.n_agents - 1), n_h1_others), nn.ReLU()))
            attention_dim = n_h1_others // attention_heads
            self.key_extractors = nn.ModuleList()
            self.query_extractors = nn.ModuleList()
            self.value_extractors = nn.ModuleList()
            for j in range(attention_heads):
                self.key_extractors.append(nn.Linear(n_h1_others, attention_dim, bias=False))
                self.query_extractors.append(nn.Linear(n_h1_self, attention_dim, bias=False))
                self.value_extractors.append(nn.Sequential(nn.Linear(n_h1_others, attention_dim), nn.ReLU()))
            self.w3_others = Parameter(torch.Tensor(n_h1_others, n_h2))
        self.q_out = nn.Linear(n_h2, 1)
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.w3_self, 0, 0.01)
        if self.stage > 1:
            nn.init.normal_(self.w3_others, 0, 0.01)

    def forward(self, state_one, act_one, state_others, act_others):
        act_one /= self.max_action
        for i in range(len(act_others)):
            act_others[i] /= self.max_action
        # state_others.shape=((n_agents-1), batch_size, o_dim)
        # act_others.shape=((n_agents-1), batch_size, a_dim)
        concat_input = torch.cat([state_one, act_one], dim=-1)
        x1_self = F.relu(self.fc1_self(concat_input))
        x2_self = F.relu(self.fc2_self(x1_self))
        x3_self = torch.matmul(x2_self, self.w3_self)
        if self.stage > 1:
            # [(batch_size, o_dim+a_dim), (the same), ...]
            inps = [torch.cat([state_others[i, :, :], act_others[i, :, :]], dim=-1) for i in range(self.n_agents - 1)]
            # [(batch_size, n_h1_others), (the same), ...]
            others_sa_encodings = [encoder(inp) for encoder, inp in zip(self.fc1_others, inps)]
            # [[(batch_size, attention_dim), (), (), ...], [], [], []], len(all_head_keys)=4
            all_head_keys = [[k_ext(enc) for enc in others_sa_encodings] for k_ext in self.key_extractors]
            # [[(batch_size, attention_dim), (), (), ...], [], [], []], len(all_head_values)=4
            all_head_values = [[v_ext(enc) for enc in others_sa_encodings] for v_ext in self.value_extractors]
            # [(batch_size, attention_dim), (), (), ()], len(all_head_selectors)=4
            all_head_selectors = [sel_ext(x1_self) for sel_ext in self.query_extractors]
            others_all_values = []
            for curr_head_key, curr_head_value, curr_head_selector in zip(all_head_keys, all_head_values, all_head_selectors):
                selector = curr_head_selector  # shape=(batch_size, attention_dim)
                # torch.stack(curr_head_key).shape=((n_agents-1), batch_size, attention_dim)
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1), torch.stack(curr_head_key).permute(1, 2, 0))
                # attend_logits.shape=(batch_size*n_agents, 1, (n_agents-1))
                scaled_attend_logits = attend_logits / np.sqrt(curr_head_key[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                # torch.stack(curr_head_value).shape=((n_agents-1), batch_size, attention_dim)
                other_values = (torch.stack(curr_head_value).permute(1, 2, 0) * attend_weights).sum(dim=2)
                others_all_values.append(other_values)
            x2_others = torch.cat(others_all_values, dim=-1)  # shape=(batch_size, n_h1_others)
            x3_others = torch.matmul(x2_others, self.w3_others)  # shape=(batch_size, n_h2)
            x3 = F.relu((x3_self + x3_others))
        else:
            x3 = F.relu(x3_self)
        q_value = self.q_out(x3)
        return q_value


















