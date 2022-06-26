import torch
import os
import numpy as np
from network.base_net import utility_network, mixing_network
from common.utils import hard_update, soft_update

# 将所有agent的utility network定义在该类中，mixing network仅有一个
class QMIX:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.train_step = 0

        # create all utility network for all agents
        self.q_utilities, self.target_q_utilities = [], []
        for i in range(self.n_agents):
            self.q_utilities.append(utility_network(args))
            self.target_q_utilities.append(utility_network(args))

        self.mixing_network = mixing_network(args)
        self.target_mixing_network = mixing_network(args)

        if args.cuda:
            for q_utility_i in self.q_utilities:
                q_utility_i.cuda()
            for target_q_utility_i in self.target_q_utilities:
                target_q_utility_i.cuda()
            self.mixing_network.cuda()
            self.target_mixing_network.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_q_utility_params.pkl'.format(0)):
                for agent_id in range(self.args.n_agents):
                    path_q = self.args.model_save_dir + '/evaluate_model/{}_q_utility_params.pkl'.format(agent_id)
                    map_location = 'cuda:0' if self.args.cuda else 'cpu'
                    self.q_utilities[agent_id].load_state_dict(torch.load(path_q, map_location=map_location))
                print('Successfully load the network')
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        for target_q_utility_i, q_utility_i in zip(self.target_q_utilities, self.q_utilities):
            hard_update(target_q_utility_i, q_utility_i)
        hard_update(self.target_mixing_network, self.mixing_network)

        # create the optimizer
        self.eval_params = list(self.mixing_network.parameters())
        for q_utility_i in self.q_utilities:
            self.eval_params += list(q_utility_i.parameters())
        self.optimizer = torch.optim.Adam(self.eval_params, lr=self.args.lr_q)

    # update the network
    def train(self, transitions, logger=None):
        # All agents share reward in the hallway_2agent scenarios.
        r = transitions['r_%d' % 0]
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

        act_index = np.indices((self.args.n_agents, self.args.batch_size))
        act_onehot = torch.zeros([self.args.n_agents, self.args.batch_size, self.args.n_actions])
        act_onehot[act_index[0], act_index[1], u] = 1

        if self.args.cuda:
            r = r.cuda()
            o = o.cuda()
            u = u.cuda()
            o_next = o_next.cuda()
            done = done.cuda()
            act_onehot = act_onehot.cuda()
        # o/u/o_next.shape = (n_agents, batch_size, obs_shape/1/obs_shape)
        done = done.unsqueeze(dim=1)
        done_multiplier = - (done - 1)

        # calculate the target Q value function
        target_q_utilities = []
        eval_q_utilities = []
        for agent_id in range(self.args.n_agents):
            ind_o_next = o_next[agent_id]
            other_o_next = o_next[np.arange(self.args.n_agents) != agent_id, :, :].reshape(self.args.batch_size, -1)
            target_q_utilities_all = self.target_q_utilities[agent_id](ind_o_next, other_o_next).detach()     # shape=(batch_size, n_actions)
            max_target_q_utility = target_q_utilities_all.max(dim=-1)[0]
            target_q_utilities.append(max_target_q_utility)


            ind_o_curr = o[agent_id]
            other_o_curr = o[np.arange(self.args.n_agents) != agent_id, :, :].reshape(self.args.batch_size, -1)
            eval_q_utilities_all = self.q_utilities[agent_id](ind_o_curr, other_o_curr)
            eval_q_utility = (eval_q_utilities_all * act_onehot[agent_id]).sum(dim=-1)
            eval_q_utilities.append(eval_q_utility)

        target_q_utilities = torch.stack(target_q_utilities, dim=-1)        # shape=(batch_size, n_agents)
        eval_q_utilities = torch.stack(eval_q_utilities, dim=-1)            # shape=(batch_size, n_agents)

        state_next = o_next.permute(1, 0, 2).reshape(self.args.batch_size, -1)      # shape=(batch_size, n_agents*obs_shape)
        state = o.permute(1, 0, 2).reshape(self.args.batch_size, -1)
        q_total_next = self.target_mixing_network(target_q_utilities, state_next).detach()
        q_total = self.mixing_network(eval_q_utilities, state)

        q_total_target = (r.unsqueeze(1) + self.args.gamma * q_total_next * done_multiplier).detach()
        td_loss = (q_total_target - q_total).pow(2).mean()

        if logger is not None:
            if self.train_step % 1000 == 0:
                logger.add_scalar('td_loss', td_loss, self.train_step // 1000)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step > 0 and (self.train_step % self.args.target_update_cycle) == 0:
            hard_update(self.target_mixing_network, self.mixing_network)
            for target_q_utility, eval_q_utility in zip(self.target_q_utilities, self.q_utilities):
                hard_update(target_q_utility, eval_q_utility)

    def save_model(self, model_save_path, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        for agent_id in range(self.n_agents):
            model_path = os.path.join(model_save_path, 'agent_%d' % agent_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.q_utilities[agent_id].state_dict(), model_path + '/' + num + '_q_utility_params.pkl')


