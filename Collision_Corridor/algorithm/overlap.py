import torch
import os
import numpy as np
from torch.distributions import Categorical
from network.double_actor_critic import Actor, Critic
from network.detector import Detector
from common.utils import hard_update, soft_update

CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class Overlap:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.stage = args.stage
        self.actions = np.eye(self.args.n_actions)

        # create the network
        self.actor = Actor(args, stage=self.stage)
        self.critic = Critic(args, stage=self.stage)

        # build up the target network
        self.target_actor = Actor(args, stage=self.stage)
        self.target_critic = Critic(args, stage=self.stage)

        # 测试为什么效果会出现波动
        if self.stage > 1 and self.args.use_overlap:
            self.expert_actor = Actor(args, stage=1)
            self.expert_critic = Critic(args, stage=1)
            self.detec = Detector(args)

        if args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.target_actor.cuda()
            self.target_critic.cuda()
            if hasattr(self, 'expert_actor'):
                self.expert_actor.cuda()
                self.expert_critic.cuda()
                self.detec.cuda()

        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)):
                path_actor = self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)
                # 用于在evaluation阶段计算multi_agent_policy与single_agent_policy之间的kl散度大小
                # path_critic = self.args.model_save_dir + '/evaluate_model/{}_critic_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_actor, map_location=map_location))
                # self.critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                if hasattr(self, 'expert_actor'):
                    print('nihao')
                    path_expert_actor = self.args.model_save_dir + '/evaluate_model/{}_single_actor_params.pkl'.format(self.agent_id)
                    # path_expert_critic = self.args.model_save_dir + '/evaluate_model/{}_single_critic_params.pkl'.format(self.agent_id)
                    path_detector = self.args.model_save_dir + '/evaluate_model/{}_detec_params.pkl'.format(self.agent_id)
                    # 用于在evaluation阶段计算multi-agent-policy和single-agent-policy之间的kl散度
                    self.expert_actor.load_state_dict(torch.load(path_expert_actor, map_location=map_location))
                    # self.expert_critic.load_state_dict(torch.load(path_expert_critic, map_location=map_location))
                    self.detec.load_state_dict(torch.load(path_detector, map_location=map_location))
                print('Successfully load the network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        if hasattr(self, 'detec'):
            self.detec_optim = torch.optim.Adam(self.detec.parameters(), lr=self.args.lr_detec)

    def load_pretrained_model(self, model_load_path):
        # Load pretrained model in stage 1, as well as for the self.expert_actor and self.expert_critic
        if os.path.exists(model_load_path + '/stage_1_0_actor_params.pkl'):
            path_partial_actor = model_load_path + '/stage_1_{}_actor_params.pkl'.format(self.agent_id)
            path_partial_critic = model_load_path + '/stage_1_{}_critic_params.pkl'.format(self.agent_id)
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            pre_actor_dict = torch.load(path_partial_actor, map_location=map_location)
            pre_critic_dict = torch.load(path_partial_critic, map_location=map_location)
            curr_actor_dict = self.actor.state_dict()
            curr_critic_dict = self.critic.state_dict()
            curr_actor_dict.update(pre_actor_dict)
            curr_critic_dict.update(pre_critic_dict)
            self.actor.load_state_dict(curr_actor_dict)
            self.critic.load_state_dict(curr_critic_dict)
            if hasattr(self, 'expert_actor'):
                print('woyehao')
                self.expert_actor.load_state_dict(pre_actor_dict)
                self.expert_critic.load_state_dict(pre_critic_dict)
            print('Successfully load the pretrained models')
        else:
            raise Exception('No pretrained models')

    def get_flag(self, obs_inps):
        out, out_probs = self.detec(obs_inps)
        out_flag = torch.gt(out_probs[:, 1], 0.5).unsqueeze(dim=1)
        return out_flag

    # 在evaluation阶段当agent执行完某一动作之后比较上一状态处的action distribution，因为此时其他agent动作已知。
    def cal_act_probs(self, o_cat, u_cat):
        o_cat = torch.tensor(o_cat, dtype=torch.float32)
        u_cat = torch.from_numpy(np.array(u_cat)).long().unsqueeze(dim=1)

        act_index = np.indices((self.args.n_agents, 1))
        act_onehot = torch.zeros([self.args.n_agents, 1, self.args.n_actions])
        act_onehot[act_index[0], act_index[1], u_cat] = 1

        # o_cat.shape=(n_agents, batch_size, obs_shape)
        o_local = o_cat[self.agent_id].unsqueeze(dim=0)          # shape=(1, obs_shape)
        size = o_local.shape[0]
        o_others = o_cat[np.arange(self.args.n_agents) != self.agent_id, :].unsqueeze(dim=1)      # shape=(n_agents-1, 1, obs_shape)
        o_local_rep = o_local.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(size * self.args.n_actions, -1)
        o_others_rep = o_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
                                                                            size * self.args.n_actions, -1)
        o_others_rep = o_others_rep.permute(1, 0, 2).reshape(size * self.args.n_actions, -1)
        act_others = act_onehot[np.arange(self.args.n_agents) != self.agent_id, :, :]
        act_others_rep = act_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
                                                                            size * self.args.n_actions, -1)
        act_others_rep = act_others_rep.permute(1, 0, 2).reshape(size * self.args.n_actions, -1)
        act_cf = np.tile(self.actions, (size, 1))
        act_cf = torch.from_numpy(act_cf).float()
        if self.args.cuda:
            act_cf = act_cf.cuda()
        Q_multi_all = self.critic(o_local_rep, act_cf, o_others_rep, act_others_rep).reshape(size, self.args.n_actions).detach()
        Q_single_all = self.expert_critic(o_local_rep, act_cf, o_others_rep, act_others_rep).reshape(size, self.args.n_actions).detach()
        act_probs_multi_Q = torch.nn.functional.softmax(Q_multi_all, dim=-1)
        act_probs_single_Q = torch.nn.functional.softmax(Q_single_all, dim=-1)
        return act_probs_multi_Q, act_probs_single_Q

    def cal_probs(self, o_cat, u_cat):
        o_cat = torch.tensor(o_cat, dtype=torch.float32)
        # o_cat.shape=(n_agents, batch_size, obs_shape)
        o_local = o_cat[self.agent_id].unsqueeze(dim=0)         # shape=(batch_size, obs_shape)
        size = o_local.shape[0]
        o_others = o_cat[np.arange(self.args.n_agents) != self.agent_id, :].unsqueeze(dim=1)      # shape=(n_agents-1, batch_size, obs_shape)
        o_others = o_others.permute(1, 0, 2).reshape(size, -1)
        act_probs_multi = self.actor(o_local, o_others).detach()
        act_probs_single = self.expert_actor(o_local, o_others).detach()
        return act_probs_multi, act_probs_single

    def get_kl_values(self, o_cat, u_cat):
        # According to critic network
        act_probs_multi_Q, act_probs_single_Q = self.cal_act_probs(o_cat, u_cat)
        kl_values_Q = torch.sum(act_probs_multi_Q * torch.log(act_probs_multi_Q / act_probs_single_Q), dim=-1, keepdim=True)
        kl_values_Q = kl_values_Q.cpu().numpy()

        act_probs_multi, act_probs_single = self.cal_probs(o_cat, u_cat)
        kl_values_pi = torch.sum(act_probs_multi * torch.log(act_probs_multi / act_probs_single), dim=-1,
                                keepdim=True)
        kl_values_pi = kl_values_pi.cpu().numpy()

        return kl_values_Q, kl_values_pi

    def detec_cal_act_probs(self, o_cat, u_cat):
        # o_cat/u_cat.shape=(n_agents, detec_buffer_size, obs_shape/1)

        u_cat = u_cat.squeeze(dim=-1)       # shape=(n_agents, detec_buffer_size)
        act_index = np.indices((self.args.n_agents, self.args.detec_buffer_size))
        act_onehot = torch.zeros([self.args.n_agents, self.args.detec_buffer_size, self.args.n_actions])
        act_onehot[act_index[0], act_index[1], u_cat] = 1

        o_local = o_cat[self.agent_id]  # shape=(detec_buffer_size, obs_shape)
        size = o_local.shape[0]
        o_others = o_cat[np.arange(self.args.n_agents) != self.agent_id, :, :]  # shape=(n_agents-1, detec_buffer_size, obs_shape)
        o_local_rep = o_local.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(size * self.args.n_actions, -1)
        o_others_rep = o_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
                                                                                            size * self.args.n_actions, -1)
        o_others_rep = o_others_rep.permute(1, 0, 2).reshape(size * self.args.n_actions, -1)
        act_others = act_onehot[np.arange(self.args.n_agents) != self.agent_id, :, :]
        act_others_rep = act_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
                                                                                            size * self.args.n_actions, -1)
        act_others_rep = act_others_rep.permute(1, 0, 2).reshape(size * self.args.n_actions, -1)
        act_cf = np.tile(self.actions, (size, 1))
        act_cf = torch.from_numpy(act_cf).float()
        if self.args.cuda:
            act_cf = act_cf.cuda()
        Q_multi_all = self.critic(o_local_rep, act_cf, o_others_rep, act_others_rep).reshape(size, self.args.n_actions).detach()
        Q_single_all = self.expert_critic(o_local_rep, act_cf, o_others_rep, act_others_rep).reshape(size, self.args.n_actions).detach()
        act_probs_multi_Q = torch.nn.functional.softmax(Q_multi_all, dim=-1)
        act_probs_single_Q = torch.nn.functional.softmax(Q_single_all, dim=-1)
        return act_probs_multi_Q, act_probs_single_Q

    def output_detec(self, o_cat, u_cat):
        # o_cat, u_cat.shape=(n_agents, batch_size, obs_shape/1)
        if self.args.cuda:
            o_cat = o_cat.cuda()
            u_cat = u_cat.cuda()
        o_local = o_cat[self.agent_id]      # shape=(detec_buffer_size, obs_shape)
        o_others = o_cat[np.arange(self.args.n_agents) != self.agent_id, :, :]  # shape=(n_agents-1, detec_buffer_size, obs_shape)
        o_others = o_others.permute(1, 0, 2).reshape(self.args.detec_buffer_size, -1)
        obs_inps = torch.cat([o_local, o_others], dim=-1)       # shape=(detec_buffer_size, obs_shape*2)
        obs_inps = obs_inps.cpu().numpy()
        act_probs_multi_Q, act_probs_single_Q = self.detec_cal_act_probs(o_cat, u_cat)
        kl_values_Q = torch.sum(act_probs_multi_Q * torch.log(act_probs_multi_Q / act_probs_single_Q), dim=-1, keepdim=True)
        kl_values_Q = kl_values_Q.cpu().numpy()
        return obs_inps, kl_values_Q

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

    def target_step(self, o_next, other_o_next, epsilon):
        target_probs = self.target_actor(o_next, other_o_next)
        target_act_probs = (1 - epsilon) * target_probs + (epsilon / float(self.args.n_actions))
        target_act = Categorical(target_act_probs).sample().long()
        return target_act

    # update the network
    def train(self, transitions, epsilon, other_agents):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
        done = transitions['done']

        r = torch.from_numpy(np.array(r)).float()
        o = torch.from_numpy(np.array(o)).float()
        u = torch.from_numpy(np.array(u)).long()
        o_next = torch.from_numpy(np.array(o_next)).float()
        done = torch.from_numpy(done).float()

        if self.args.cuda:
            r = r.cuda()
            o = o.cuda()
            u = u.cuda()
            o_next = o_next.cuda()
            done = done.cuda()
        # o/u/o_next.shape = (n_agents, batch_size, obs_shape/1/obs_shape)
        done = done.unsqueeze(dim=1)
        done_multiplier = - (done - 1)
        # calculate the target Q value function
        u_next = []
        target_act_index = np.indices((self.args.batch_size, ))
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                ind_o_next = o_next[agent_id]
                other_o_next = o_next[np.arange(self.args.n_agents) != agent_id, :, :].reshape(self.args.batch_size, -1)
                if agent_id == self.agent_id:
                    target_act = self.target_step(ind_o_next, other_o_next, epsilon)
                    target_act_onehot = torch.zeros((self.args.batch_size, self.args.n_actions))
                    target_act_onehot[target_act_index, target_act] = 1
                    u_next.append(target_act_onehot)
                else:
                    target_act_other = other_agents[index].policy.target_step(ind_o_next, other_o_next, epsilon)
                    target_act_other_onehot = torch.zeros((self.args.batch_size, self.args.n_actions))
                    target_act_other_onehot[target_act_index, target_act_other] = 1
                    u_next.append(target_act_other_onehot)
                    index += 1
            u_next = torch.stack(u_next)
            if self.args.cuda:
                u_next = u_next.cuda()

            # o_next/u_next.shape=(batch_size, n_agents * obs_shape/n_actions)
            o_next_others = o_next[np.arange(self.args.n_agents) != self.agent_id].permute(1, 0, 2).reshape(self.args.batch_size, -1)
            u_next_others = u_next[np.arange(self.args.n_agents) != self.agent_id].permute(1, 0, 2).reshape(self.args.batch_size, -1)
            q_next = self.target_critic(o_next[self.agent_id], u_next[self.agent_id], o_next_others, u_next_others).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next * done_multiplier).detach()

        # the critic loss
        act_index = np.indices((self.args.n_agents, self.args.batch_size))
        act_onehot = torch.zeros([self.args.n_agents, self.args.batch_size, self.args.n_actions])
        act_onehot[act_index[0], act_index[1], u] = 1
        # o/act_onehot.shape=(n_agents, batch_size, obs_shape/n_actions)
        others_o = o[np.arange(self.args.n_agents) != self.agent_id].permute(1, 0, 2).reshape(self.args.batch_size, -1)
        others_act_onehot = act_onehot[np.arange(self.args.n_agents) != self.agent_id].permute(1, 0, 2).reshape(self.args.batch_size, -1)

        q_eval = self.critic(o[self.agent_id], act_onehot[self.agent_id], others_o, others_act_onehot)

        critic_loss = (target_q - q_eval).pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        o_others = o[np.arange(self.args.n_agents) != self.agent_id, :, :]
        tmp_o_others = o_others.reshape(self.args.batch_size, -1)
        act_probs = self.actor(o[self.agent_id], tmp_o_others)
        act_probs = (1 - epsilon) * act_probs + (epsilon / float(self.args.n_actions))
        # # shape = (batch_size * n_actions, obs_shape)
        o_self_rep = o[self.agent_id].unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.obs_shape)
        # # shape = (n_agents-1, batch_size, obs_shape)
        # o_others = o[np.arange(self.args.n_agents) != self.agent_id, :, :]
        o_others_rep = o_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
                                                                                                 -1, self.args.obs_shape)
        o_others_rep = o_others_rep.permute(1, 0, 2).reshape(-1, self.args.obs_shape * (self.args.n_agents - 1))
        # o_rep = o.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape(self.args.n_agents, -1, self.args.obs_shape)
        # u_onehot_rep = []
        # for agent_id in range(self.args.n_agents):
        #     if agent_id == self.agent_id:
        #         act_cf = np.tile(self.actions, (self.args.batch_size, 1))       # shape=(batch_size * n_actions, n_actions)
        #         act_cf = torch.from_numpy(act_cf).float()
        #         u_onehot_rep.append(act_cf)
        #     else:
        #         u_onehot_other = act_onehot[agent_id]       # shape=(batch_size, n_actions)
        #         u_onehot_other_rep = u_onehot_other.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.n_actions)
        #         u_onehot_rep.append(u_onehot_other_rep)
        # u_onehot_rep = torch.stack(u_onehot_rep)
        # if self.args.cuda:
        #     u_onehot_rep = u_onehot_rep.cuda()
        u_onehot_others = act_onehot[np.arange(self.args.n_agents) != self.agent_id, :, :]
        u_onehot_others_rep = u_onehot_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents-1),
                                                                                                        -1, self.args.n_actions)
        u_onehot_others_rep = u_onehot_others_rep.permute(1, 0, 2).reshape(-1, self.args.n_actions * (self.args.n_agents - 1))
        act_cf = np.tile(self.actions, (self.args.batch_size, 1))
        act_cf = torch.from_numpy(act_cf).float()
        if self.args.cuda:
            act_cf = act_cf.cuda()
        # # shape = (batch_size, n_actions)
        q_cf_all = self.critic(o_self_rep, act_cf, o_others_rep, u_onehot_others_rep).reshape(self.args.batch_size, self.args.n_actions)
        # o_rep = o_rep.permute(1, 0, 2).reshape(self.args.batch_size * self.args.n_actions, -1)
        # u_onehot_rep = u_onehot_rep.permute(1, 0, 2).reshape(self.args.batch_size * self.args.n_actions, -1)
        # q_cf_all = self.critic(o_rep, u_onehot_rep, None, None).reshape(self.args.batch_size, self.args.n_actions)

        probs = self.actor(o[self.agent_id], tmp_o_others)
        probs = (1 - epsilon) * probs + epsilon / float(self.args.n_actions)
        log_probs = torch.log(torch.sum(probs * act_onehot[self.agent_id], dim=1, keepdim=True)+1e-15)

        bias = torch.sum(act_probs * q_cf_all, dim=1, keepdim=True).detach()
        advantage = (q_eval - bias).detach()
        actor_loss = - (log_probs * advantage).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.target_actor, self.actor, self.args.tau)
        soft_update(self.target_critic, self.critic, self.args.tau)

    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        model_path = os.path.join(model_save_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
        if hasattr(self, 'detec'):
            torch.save(self.detec.state_dict(), model_path + '/' + num + '_detec_params.pkl')


