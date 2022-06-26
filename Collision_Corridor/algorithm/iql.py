import torch
import os
import numpy as np
from network.base_net import MLP, Detector
from common.utils import hard_update, soft_update

MSELoss = torch.nn.MSELoss()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class IQL:
    def __init__(self, args, agent_id):
        self.args = args
        self.stage = args.stage
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.q_network = MLP(args, agent_id)
        self.target_q_network = MLP(args, agent_id)

        if self.stage > 1 and args.use_overlap:
            self.expert_single_q = MLP(args, agent_id)
            self.detec = Detector(args)

        if args.cuda:
            self.q_network.cuda()
            self.target_q_network.cuda()
            if hasattr(self, 'expert_single_q'):
                self.expert_single_q.cuda()
                self.detec.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_q_params.pkl'.format(self.agent_id)):
                path_q = self.args.model_save_dir + '/evaluate_model/{}_q_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.q_network.load_state_dict(torch.load(path_q, map_location=map_location))
                if hasattr(self, 'detec'):
                    path_detec = self.args.model_save_dir + '/evaluate_model/{}_detec_params.pkl'.format(self.agent_id)
                    self.detec.load_state_dict(torch.load(path_detec, map_location=map_location))
                print('Successfully load the network: {}'.format(path_q))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.target_q_network, self.q_network)

        # create the optimizer
        self.q_optim = torch.optim.Adam(self.q_network.parameters(), lr=self.args.lr_q)
        if hasattr(self, 'detec'):
            self.detec_optim = torch.optim.Adam(self.detec.parameters(), lr=self.args.lr_detec)

    def load_pretrained_model(self, model_load_path):
        # Load pretrained model in stage 1, as well as for the self.expert_actor and self.expert_critic
        if os.path.exists(model_load_path + '/stage_1_q_params.pkl'):
            path_partial_q = model_load_path + '/stage_1_q_params.pkl'
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            pre_q_dict = torch.load(path_partial_q, map_location=map_location)
            curr_q_dict = self.q_network.state_dict()
            curr_q_dict.update(pre_q_dict)
            self.q_network.load_state_dict(curr_q_dict)
            if self.stage > 1 and self.args.use_overlap:
                self.expert_single_q.load_state_dict(pre_q_dict)
            print('Successfully load the pretrained models')
        else:
            raise Exception('No pretrained models')

    def get_flag(self, inps):
        out, out_prob = self.detec(inps)
        out_flag = torch.gt(out_prob[:, 1], 0.5).unsqueeze(1)
        return out_flag

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
    def train(self, transitions, logger):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]
        done = transitions['done']

        r = torch.from_numpy(r).float()
        o = torch.from_numpy(o).float()
        u = torch.from_numpy(u).long()
        o_next = torch.from_numpy(o_next).float()
        done = torch.from_numpy(done).float()

        if self.args.cuda:
            r = r.cuda()
            o = o.cuda()
            u = u.cuda()
            o_next = o_next.cuda()
            done = done.cuda()

        done = done.unsqueeze(dim=1)
        done_multiplier = - (done - 1)
        # calculate the target Q value function
        with torch.no_grad():
            q_next_all = self.target_q_network(o_next).detach()
            q_next = torch.max(q_next_all, dim=1, keepdim=True)[0]
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next * done_multiplier).detach()

        q_eval_all = self.q_network(o)
        u = u.unsqueeze(dim=1)
        q_eval = torch.gather(q_eval_all, dim=1, index=u)
        td_loss = (target_q - q_eval).pow(2).mean()

        if logger is not None:
            if self.train_step % 1000 == 0:
                logger.add_scalar('agent_%d_td_loss'%self.agent_id, td_loss, self.train_step // 1000)

        self.q_optim.zero_grad()
        td_loss.backward()
        self.q_optim.step()

        self.train_step += 1
        if self.train_step > 0 and (self.train_step % self.args.target_update_cycle) == 0:
            hard_update(self.target_q_network, self.q_network)


    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        model_path = os.path.join(model_save_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.q_network.state_dict(), model_path + '/' + num + '_q_params.pkl')
        if hasattr(self, 'detec'):
            torch.save(self.detec.state_dict(), model_path + '/' + num + '_detec_params.pkl')


