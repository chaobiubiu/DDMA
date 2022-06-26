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
        self.hidden_dim = hidden_dim
        self.agent_id = agent_id
        self.n_agents = args.n_agents
        self.max_action = args.high_action
        # fc1中包含n_agents个embedding network
        # self.fc1 = nn.ModuleList()
        # for i in range(self.n_agents):
        #     # 默认所有agent的action_shape相同
        #     self.fc1.append(nn.Sequential(nn.Linear(args.input_dim_self + args.action_shape[agent_id], hidden_dim), nn.ReLU()))
        # 这里使用一个网络来extract embedding，之前MAAC三个网络参数量过大
        self.fc1 = nn.Sequential(nn.Linear(args.input_dim_self + args.action_shape[agent_id], hidden_dim), nn.ReLU())
        # hard attention层输入为[h_i, h_j]，因此为hidden_dim * 2, 输出为0或1
        self.hard_encoding = nn.Linear(hidden_dim * 2, 2)
        # 这里不考虑multi-head attention，保证对比算法参数总量相同
        self.key_extractor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_extractor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_extractor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # 输入为当前agent自己经过fc1得到的embedding以及通过hard+soft attention network聚合后的其他agent's contribution
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_cat, act_cat):
        # state_cat.shape=(n_agents, batch_size, input_dim_self)
        # act_cat.shape=(n_agents, batch_size, act_shape)
        for i in range(len(act_cat)):
            act_cat[i] /= self.max_action

        inps = torch.cat([state_cat, act_cat], dim=-1)      # shape=(n_agents, batch_size, input_dim_self+action_shape)
        oa_encodings = self.fc1(inps)       #shape=(n_agents, batch_size, hidden_dim)
        encoding_self = oa_encodings[self.agent_id]     #shape=(batch_size, hidden_dim)
        encoding_others = oa_encodings[np.arange(self.args.n_agents) != self.agent_id]      # shape=(n_agents-1, batch_size, hidden_dim)
        encoding_self_rep = encoding_self.unsqueeze(dim=0).expand((self.args.n_agents - 1), -1, -1) # shape=(n_agents-1, batch_size, hidden_dim)
        '''hard attention part'''
        hard_inps = torch.cat([encoding_self_rep, encoding_others], dim=-1)     # shape=(n_agents-1, batch_size, hidden_dim*2)
        # hard_inps = [[h_self, h_others_0], [h_self, h_others_1], ...]
        hard_weights = self.hard_encoding(hard_inps)         # shape=(n_agents-1, batch_size, 2)
        hard_weights = F.gumbel_softmax(hard_weights, tau=0.01)
        hard_weights = hard_weights[:, :, 1].unsqueeze(dim=2)        # shape=(n_agents-1, batch_size, 1)
        '''soft attention part'''
        others_head_keys = self.key_extractor(encoding_others)  # shape=(n_agents-1, batch_size ,hidden_dim)
        others_head_values = self.value_extractor(encoding_others)  # shape=(n_agents-1, batch_size, hidden_dim)
        self_head_selector = self.query_extractor(encoding_self)    # shape=(batch_size, hidden_dim)
        
        # After view, self_head_selector.shape=(batch_size, 1, hidden_dim)
        # After permute, others_head_keys.shape=(batch_size, hidden_dim, n_agents-1)
        attend_logits = torch.matmul(self_head_selector.view(self.args.batch_size, 1, -1), others_head_keys.permute(1, 2, 0))
        scaled_attend_logits = attend_logits / np.sqrt(self.hidden_dim)
        attend_weights = F.softmax(scaled_attend_logits, dim=2)     # shape=(batch_size, 1, n_agents-1)
        # After permute, hard_weights.shape=(batch_size, 1, n_agents-1)
        hard_weights = hard_weights.permute(1, 2, 0)
        # shape: attend_weights/hard_weights.shape=(batch_size, 1, n_agents-1)
        # shape: others_head_values.permute(1, 2, 0).shape=(batch_size, hidden_dim, n_agents-1)
        others_all_values = (others_head_values.permute(1, 2, 0) * attend_weights * hard_weights).sum(dim=2)
        x1 = torch.cat([encoding_self, others_all_values], dim=-1)      # shape=(batch_size, hidden_dim * 2)
        x2 = F.relu(self.fc2(x1))
        q_value = self.q_out(x2)
        return q_value


















