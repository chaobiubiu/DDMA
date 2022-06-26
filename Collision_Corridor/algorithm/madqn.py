import torch
import os
import numpy as np
from network.base_net import MLP, Detector, MAMLP
from common.utils import hard_update, soft_update

MSELoss = torch.nn.MSELoss()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class MADQN:
    def __init__(self, args, agent_id):
        self.args = args
        self.stage = args.stage
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.q_network = MAMLP(args)
        self.target_q_network = MAMLP(args)

        if args.cuda:
            self.q_network.cuda()
            self.target_q_network.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_q_params.pkl'.format(self.agent_id)):
                path_q = self.args.model_save_dir + '/evaluate_model/{}_q_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.q_network.load_state_dict(torch.load(path_q, map_location=map_location))
                print('Successfully load the network: {}'.format(path_q))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.target_q_network, self.q_network)

        # create the optimizer
        self.q_optim = torch.optim.Adam(self.q_network.parameters(), lr=self.args.lr_q)

    # update the network
    def train(self, transitions, other_agents, logger):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
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
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                ind_o_next = o_next[agent_id]
                if agent_id == self.agent_id:
                    u_next.append(torch.argmax(self.target_q_network()))
                else:
                    u_next.append(torch.argmax(other_agents[index].policy.target_q_network()))


            q_next_all = self.target_q_network(o_next[self.agent_id], o_next).detach()
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


