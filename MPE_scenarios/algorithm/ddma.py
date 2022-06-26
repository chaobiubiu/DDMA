import torch
import os
import numpy as np
from network.detector import Detector
from network.double_actor_critic import Actor, Critic, Attention_Critic
from common.utils import hard_update, soft_update

CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class DDMA:
    def __init__(self, args, agent_id):
        self.args = args
        self.stage = args.stage
        self.agent_id = agent_id
        self.train_step = 0
        '''这里的actions后续需要manual design for calculate the policy distribution'''
        self.actions = np.eye(self.args.action_shape[agent_id])
        self.act_dim = self.args.action_shape[self.agent_id]

        # create the network
        self.actor_network = Actor(args, agent_id, stage=self.stage)
        self.actor_target_network = Actor(args, agent_id, stage=self.stage)

        if not self.args.use_attention:
            self.critic_network = Critic(args, agent_id, stage=self.stage)
            self.critic_target_network = Critic(args, agent_id, stage=self.stage)
        else:
            self.critic_network = Attention_Critic(args, agent_id, stage=self.stage, attention_heads=self.args.attention_heads)
            self.critic_target_network = Attention_Critic(args, agent_id, stage=self.stage, attention_heads=self.args.attention_heads)

        if self.stage > 1 and self.args.use_overlap:
            self.expert_actor = Actor(args, agent_id, stage=1)
            self.expert_critic = Critic(args, agent_id, stage=1)
            self.detec = Detector(args, agent_id)

        if args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            if self.stage > 1 and self.args.use_overlap:
                self.expert_actor.cuda()
                self.expert_critic.cuda()
                self.detec.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)):
                path_actor = self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor_network.load_state_dict(torch.load(path_actor, map_location=map_location))
                '''修改想法之后，在evaluate阶段仅使用actor网络进行动作选择，detec和expert actor网络仅在training阶段使用'''
                # if hasattr(self, 'detec'):
                #     path_detec = self.args.model_save_dir + '/evaluate_model/{}_detec_params.pkl'.format(self.agent_id)
                #     self.detec.load_state_dict(torch.load(path_detec, map_location=map_location))
                # if hasattr(self, 'expert_actor'):
                #     # single-agent policy is shared among all agents
                #     path_expert_actor = self.args.model_save_dir + '/evaluate_model/single_actor_params.pkl'
                #     self.expert_actor.load_state_dict(torch.load(path_expert_actor, map_location=map_location))
                print('Successfully load the evaluation network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.actor_target_network, self.actor_network)
        hard_update(self.critic_target_network, self.critic_network)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        if hasattr(self, 'detec'):
            self.detec_optim = torch.optim.Adam(self.detec.parameters(), lr=self.args.lr_detec)

    def load_pretrained_model(self, model_load_path):
        # Load pretrained model in stage 1, as well as for the self.expert_actor and self.expert_critic
        if os.path.exists(model_load_path + '/stage_1_actor_params.pkl'):
            path_partial_actor = model_load_path + '/stage_1_actor_params.pkl'
            path_partial_critic = model_load_path + '/stage_1_critic_params.pkl'
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            pre_actor_dict = torch.load(path_partial_actor, map_location=map_location)
            pre_critic_dict = torch.load(path_partial_critic, map_location=map_location)
            curr_actor_dict = self.actor_network.state_dict()
            curr_critic_dict = self.critic_network.state_dict()
            curr_actor_dict.update(pre_actor_dict)
            curr_critic_dict.update(pre_critic_dict)
            self.actor_network.load_state_dict(curr_actor_dict)
            self.critic_network.load_state_dict(curr_critic_dict)
            if hasattr(self, 'expert_actor'):
                self.expert_actor.load_state_dict(pre_actor_dict)
                self.expert_critic.load_state_dict(pre_critic_dict)
                print('Have loaded single expert actor and critic')
            print('Successfully load the pretrained models')
        else:
            raise Exception('No pretrained models')

    def get_flag(self, inps):
        out, out_prob = self.detec(inps)
        out_flag = torch.gt(out_prob[:, 1], 0.5).unsqueeze(1)
        return out_flag

    def pred_strength(self, inps):
        out, out_prob = self.detec(inps)
        strength = out_prob[:, 1]
        return strength

    def cal_act_probs(self, state_one, state_others, act_others):
        size = state_one.shape[0]
        # state_one.shape=(detec_buffer_size,input_dim_self), state_others.shape=(n_agents-1,detec_buffer_size,input_dim_self)
        s_rep = state_one.unsqueeze(dim=1).expand(-1, self.act_dim, -1).reshape(-1, self.args.input_dim_self)
        # action这一部分应该对连续动作空间离散化，选择几个离散动作用来表征当前Q网络对应的action distribution
        act_cf = np.tile(self.actions, (size, 1))
        act_cf = torch.from_numpy(act_cf).float()
        if self.args.cuda:
            act_cf = act_cf.cuda()
        s_others_rep = state_others.unsqueeze(dim=2).expand(-1, -1, self.act_dim, -1)
        s_others_rep = s_others_rep.reshape((self.args.n_agents - 1), size * self.act_dim, -1)
        s_others_rep = s_others_rep.permute(1, 0, 2).reshape(size * self.act_dim, -1)
        act_others_rep = act_others.unsqueeze(dim=2).expand(-1, -1, self.act_dim, -1)
        act_others_rep = act_others_rep.reshape((self.args.n_agents - 1), size * self.act_dim, -1)
        act_others_rep = act_others_rep.permute(1, 0, 2).reshape(size * self.act_dim, -1)
        Q_multi_all = self.critic_network(s_rep, act_cf, s_others_rep, act_others_rep).reshape(size, self.act_dim).detach()
        # For single-agent critic, the information about other agents don't care.
        Q_single_all = self.expert_critic(s_rep, act_cf, s_others_rep, act_others_rep).reshape(size, self.act_dim).detach()
        act_probs_multi_Q = torch.nn.functional.softmax(Q_multi_all, dim=-1)
        act_probs_single_Q = torch.nn.functional.softmax(Q_single_all, dim=-1)
        return act_probs_multi_Q, act_probs_single_Q

    def output_detec(self, o_cat, u_cat):
        if self.args.cuda:
            o_cat = o_cat.cuda()
            u_cat = u_cat.cuda()
        state_one = o_cat[self.agent_id][:, :self.args.input_dim_self]
        # state_others, act_others = [], []
        # for agent_id in range(self.args.n_agents):
        #     if agent_id != self.agent_id:
        #         state_others.append(o_cat[agent_id])
        #         act_others.append(u_cat[agent_id])
        # state_others, act_others = torch.stack(state_others), torch.stack(act_others)
        state_others = o_cat[np.arange(self.args.n_agents) != self.agent_id, :, :self.args.input_dim_self]
        act_others = u_cat[np.arange(self.args.n_agents) != self.agent_id, :, :]
        act_probs_multi_Q, act_probs_single_Q = self.cal_act_probs(state_one, state_others, act_others)
        kl_values = torch.sum(act_probs_multi_Q * torch.log(act_probs_multi_Q / act_probs_single_Q), dim=1, keepdim=True)
        inps = o_cat[self.agent_id]
        kl_values = kl_values.cpu().numpy()
        inps = inps.cpu().numpy()
        return inps, kl_values

    def update_detec(self, batch_inps, batch_labels, iter, logger):
        batch_inps = torch.from_numpy(batch_inps).float()
        batch_labels = torch.from_numpy(batch_labels).squeeze(dim=1).long()
        if self.args.cuda:
            batch_inps = batch_inps.cuda()
            batch_labels = batch_labels.cuda()
        pred, pred_prob = self.detec(batch_inps)
        loss_ = CrossEntropyLoss(pred, batch_labels)

        self.detec_optim.zero_grad()
        loss_.backward()
        self.detec_optim.step()

        if logger is not None:
            logger.add_scalar('agent_%d_detec_loss'%self.agent_id, loss_, iter)

    # update the network
    def train(self, transitions, other_agents):
        if self.train_step % 2 == 0:
            # for key in transitions.keys():
            #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
            r = transitions['r_%d' % self.agent_id]
            o, u, o_next = [], [], []
            for agent_id in range(self.args.n_agents):
                o.append(transitions['o_%d' % agent_id])
                u.append(transitions['u_%d' % agent_id])
                o_next.append(transitions['o_next_%d' % agent_id])

            r = torch.from_numpy(np.array(r)).float()
            o = torch.from_numpy(np.array(o)).float()
            u = torch.from_numpy(np.array(u)).float()
            o_next = torch.from_numpy(np.array(o_next)).float()
            if self.args.cuda:
                r = r.cuda()
                o = o.cuda()
                u = u.cuda()
                o_next = o_next.cuda()
            # calculate the target Q value function
            u_next = []
            with torch.no_grad():
                index = 0
                for agent_id in range(self.args.n_agents):
                    obs_about_self_next = o_next[agent_id][:, :self.args.input_dim_self]
                    obs_about_others_next = o_next[agent_id][:, self.args.input_dim_self:]
                    if agent_id == self.agent_id:
                        u_next.append(self.actor_target_network(obs_about_self_next, obs_about_others_next))
                    else:
                        u_next.append(other_agents[index].policy.actor_target_network(obs_about_self_next, obs_about_others_next))
                        index += 1
                u_next = torch.stack(u_next)
                if self.args.cuda:
                    u_next = u_next.cuda()
                local_o_next = o_next[self.agent_id][:, :self.args.input_dim_self]
                local_u_next = u_next[self.agent_id]
                others_o_next = o_next[np.arange(self.args.n_agents) != self.agent_id, :, :self.args.input_dim_self]
                others_u_next = u_next[np.arange(self.args.n_agents) != self.agent_id, :, :]
                if self.args.use_overlap:
                    others_o_next = others_o_next.permute(1, 0, 2).reshape(self.args.batch_size, -1)
                    others_u_next = others_u_next.permute(1, 0, 2).reshape(self.args.batch_size, -1)
                q_next = self.critic_target_network(local_o_next, local_u_next, others_o_next, others_u_next).detach()

                target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

            # the q loss
            local_o = o[self.agent_id][:, :self.args.input_dim_self]
            local_u = u[self.agent_id]
            others_o = o[np.arange(self.args.n_agents) != self.agent_id, :, :self.args.input_dim_self]
            others_u = u[np.arange(self.args.n_agents) != self.agent_id, :, :]
            if self.args.use_overlap:
                others_o = others_o.permute(1, 0, 2).reshape(self.args.batch_size, -1)
                others_u = others_u.permute(1, 0, 2).reshape(self.args.batch_size, -1)
            q_value = self.critic_network(local_o, local_u, others_o, others_u)
            critic_loss = (target_q - q_value).pow(2).mean()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # the actor loss
            u[self.agent_id] = self.actor_network(o[self.agent_id][:, :self.args.input_dim_self], o[self.agent_id][:, self.args.input_dim_self:])
            update_local_u = u[self.agent_id]
            if not self.args.use_diy_credit:
                actor_loss = - self.critic_network(local_o, update_local_u, others_o, others_u).mean()
            else:
                single_Q_as_bias = self.expert_critic(local_o, update_local_u, others_o, others_u).detach()
                actor_loss = - (self.critic_network(local_o, update_local_u, others_o, others_u) - single_Q_as_bias).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        soft_update(self.actor_target_network, self.actor_network, self.args.tau)
        soft_update(self.critic_target_network, self.critic_network, self.args.tau)

        self.train_step += 1

    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        model_path = os.path.join(model_save_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
        if hasattr(self, 'detec'):
            torch.save(self.detec.state_dict(), model_path + '/' + num + '_detec_params.pkl')


