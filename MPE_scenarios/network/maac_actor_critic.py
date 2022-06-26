import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.action_out = nn.Linear(128, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class Attention_Critic(nn.Module):
    def __init__(self, args, agent_id, hidden_dim=64, attention_heads=2):
        super(Attention_Critic, self).__init__()
        self.args = args
        self.agent_id = agent_id
        self.n_agents = args.n_agents
        self.max_action = args.high_action
        # fc1中包含n_agents个embedding network
        self.fc1 = nn.ModuleList()
        for i in range(self.n_agents):
            # 默认所有agent的action_shape相同
            self.fc1.append(nn.Sequential(nn.Linear(args.input_dim_self + args.action_shape[agent_id], hidden_dim), nn.ReLU()))

        attention_dim = hidden_dim // attention_heads
        self.key_extractors = nn.ModuleList()
        self.query_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for j in range(attention_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attention_dim, bias=False))
            self.query_extractors.append(nn.Linear(hidden_dim, attention_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.ReLU()))
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_cat, act_cat):
        # state_cat.shape=(n_agents, batch_size, input_dim_self)
        # act_cat.shape=(n_agents, batch_size, act_shape)
        for i in range(len(act_cat)):
            act_cat[i] /= self.max_action

        inps = torch.cat([state_cat, act_cat], dim=-1)      # shape=(n_agents, batch_size, input_dim_self+action_shape)
        oa_encodings = [encoder(inp) for encoder, inp in zip(self.fc1, inps)]
        # shape=(n_agents, batch_size, hidden_dim)
        oa_encodings = torch.stack(oa_encodings)
        encoding_self = oa_encodings[self.agent_id]     #shape=(batch_size, hidden_dim)
        # shape=((n_agents-1),batch_size, hidden_dim)
        encoding_others = oa_encodings[np.arange(self.args.n_agents) != self.agent_id]
        # len(all_head_keys)=attention_heads, len(all_head_keys[0])=n_agents-1, all_head_keys[0][0].shape=(batch_size, attention_dim)
        others_head_keys = [[k_ext(enc) for enc in encoding_others] for k_ext in self.key_extractors]
        # len(all_head_values)=attention_heads, len(all_head_values[0])=n_agents-1, all_head_values[0][0].shape=(batch_size, attention_dim)
        others_head_values = [[v_ext(enc) for enc in encoding_others] for v_ext in self.value_extractors]
        # len(self_head_selectors)=attention_heads, self_head_selectors[0].shape=(batch_size, attention_dim)
        self_head_selectors = [sel_ext(encoding_self) for sel_ext in self.query_extractors]
        others_all_values = []
        for curr_head_key, curr_head_value, curr_head_selector in zip(others_head_keys, others_head_values, self_head_selectors):
            # curr_head_value.shape=(n_agents-1, batch_size, attention_dim)
            selector = curr_head_selector       # shape=(batch_size, attention_dim)
            # after view, selector.shape=(batch_size, 1, attention_dim),
            # after stack and permute, curr_head_key.shape=(batch_size, attention_dim, n_agents-1)
            attend_logits = torch.matmul(selector.view(self.args.batch_size, 1, -1), torch.stack(curr_head_key).permute(1, 2, 0))
            # attend_logits.shape=(batch_size, 1, n_agents-1)
            scaled_attend_logits = attend_logits / np.sqrt(curr_head_key[0].shape[1])
            attend_weights = F.softmax(scaled_attend_logits, dim=2)     # shape=(batch_size, 1, n_agents-1)
            # torch.stack(curr_head_value).permute(1, 2, 0).shape=(batch_size, attention_dim, n_agents-1)
            other_values = (torch.stack(curr_head_value).permute(1, 2, 0) * attend_weights).sum(dim=2)
            others_all_values.append(other_values)
        # shape=(batch_size, attention_dim * attention_heads)
        others_all_values = torch.cat(others_all_values, dim=1)
        x1 = torch.cat([encoding_self, others_all_values], dim=-1)
        x2 = F.relu(self.fc2(x1))
        q_value = self.q_out(x2)
        return q_value


















