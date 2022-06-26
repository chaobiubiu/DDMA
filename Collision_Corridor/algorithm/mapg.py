import torch
import os
import numpy as np
from torch.distributions import Categorical
from network.base_net import MAActor, CentralCritic
from common.utils import hard_update, soft_update

class MAPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.actions = np.eye(self.args.n_actions)

        # create the network
        self.actor = MAActor(args)
        self.critic = CentralCritic(args)

        # build up the target network
        self.target_actor = MAActor(args)
        self.target_critic = CentralCritic(args)

        if args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.target_actor.cuda()
            self.target_critic.cuda()

        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)):
                path_actor = self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_actor, map_location=map_location))
                print('Successfully load the network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

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
            o_next = o_next.permute(1, 0, 2).reshape(self.args.batch_size, -1)
            u_next = u_next.permute(1, 0, 2).reshape(self.args.batch_size, -1)
            q_next = self.target_critic(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next * done_multiplier).detach()

        # the critic loss
        act_index = np.indices((self.args.n_agents, self.args.batch_size))
        act_onehot = torch.zeros([self.args.n_agents, self.args.batch_size, self.args.n_actions])
        act_onehot[act_index[0], act_index[1], u] = 1
        # o/act_onehot.shape=(n_agents, batch_size, obs_shape/n_actions)
        o_all = o.permute(1, 0, 2).reshape(self.args.batch_size, -1)
        act_onehot_all = act_onehot.permute(1, 0, 2).reshape(self.args.batch_size, -1)
        q_eval = self.critic(o_all, act_onehot_all)
        critic_loss = (target_q - q_eval).pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        o_others = o[np.arange(self.args.n_agents) != self.agent_id, :, :].reshape(self.args.batch_size, -1)
        act_probs = self.actor(o[self.agent_id], o_others)
        act_probs = (1 - epsilon) * act_probs + (epsilon / float(self.args.n_actions))
        # # shape = (batch_size * n_actions, obs_shape)
        # o_self_rep = o[self.agent_id].unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.obs_shape)
        # # shape = (n_agents-1, batch_size, obs_shape)
        # o_others = o[np.arange(self.args.n_agents) != self.agent_id, :, :]
        # o_others_rep = o_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents - 1),
        #                                                                                          -1, self.args.obs_shape)
        o_rep = o.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape(self.args.n_agents, -1, self.args.obs_shape)
        u_onehot_rep = []
        for agent_id in range(self.args.n_agents):
            if agent_id == self.agent_id:
                act_cf = np.tile(self.actions, (self.args.batch_size, 1))       # shape=(batch_size * n_actions, n_actions)
                act_cf = torch.from_numpy(act_cf).float()
                u_onehot_rep.append(act_cf)
            else:
                u_onehot_other = act_onehot[agent_id]       # shape=(batch_size, n_actions)
                u_onehot_other_rep = u_onehot_other.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.n_actions)
                u_onehot_rep.append(u_onehot_other_rep)
        u_onehot_rep = torch.stack(u_onehot_rep)
        if self.args.cuda:
            u_onehot_rep = u_onehot_rep.cuda()
        # u_onehot_others = act_onehot[np.arange(self.args.n_agents) != self.agent_id, :, :]
        # u_onehot_others_rep = u_onehot_others.unsqueeze(dim=2).expand(-1, -1, self.args.n_actions, -1).reshape((self.args.n_agents-1),
        #                                                                                                 -1, self.args.n_actions)
        # act_cf = np.tile(self.actions, (self.args.batch_size, 1))
        # act_cf = torch.from_numpy(act_cf).float()
        # if self.args.cuda:
        #     act_cf = act_cf.cuda()
        # # shape = (batch_size, n_actions)
        # q_cf_all = self.critic(o_self_rep, act_cf, o_others_rep, u_onehot_others_rep)
        o_rep = o_rep.permute(1, 0, 2).reshape(self.args.batch_size * self.args.n_actions, -1)
        u_onehot_rep = u_onehot_rep.permute(1, 0, 2).reshape(self.args.batch_size * self.args.n_actions, -1)
        q_cf_all = self.critic(o_rep, u_onehot_rep).reshape(self.args.batch_size, self.args.n_actions)

        probs = self.actor(o[self.agent_id], o_others)
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


