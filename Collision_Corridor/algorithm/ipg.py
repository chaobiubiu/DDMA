import torch
import os
import numpy as np
from torch.distributions import Categorical
from network.base_net import Actor, Critic
from common.utils import hard_update, soft_update

class IPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.actions = np.eye(self.args.n_actions)

        # create the network
        self.actor = Actor(args)
        self.critic = Critic(args)

        # build up the target network
        self.target_actor = Actor(args)
        self.target_critic = Critic(args)

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

    # update the network
    def train(self, transitions, epsilon):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]
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

        done = done.unsqueeze(dim=1)
        done_multiplier = - (done - 1)
        # calculate the target Q value function
        with torch.no_grad():
            target_probs = self.target_actor(o_next)
            target_act_probs = (1 - epsilon) * target_probs + epsilon / float(self.args.n_actions)
            target_act = Categorical(target_act_probs).sample().long()
            u_next_onehot = torch.zeros([self.args.batch_size, self.args.n_actions])
            index = np.indices((self.args.batch_size, ))
            u_next_onehot[index, target_act] = 1
            if self.args.cuda:
                u_next_onehot = u_next_onehot.cuda()
            q_next = self.target_critic(o_next, u_next_onehot).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next * done_multiplier).detach()

        # the q loss
        u_onehot = torch.zeros([self.args.batch_size, self.args.n_actions])
        u_onehot[index, u] = 1
        q_eval = self.critic(o, u_onehot)
        critic_loss = (target_q - q_eval).pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        act_probs = self.actor(o)
        act_probs = (1 - epsilon) * act_probs + (epsilon / float(self.args.n_actions))
        # shape = (batch_size * n_actions, obs_shape)
        o_rep = o.unsqueeze(dim=1).expand(-1, self.args.n_actions, -1).reshape(-1, self.args.obs_shape)
        act_cf = np.tile(self.actions, (self.args.batch_size, 1))
        act_cf = torch.from_numpy(act_cf).float()
        if self.args.cuda:
            act_cf = act_cf.cuda()
        # shape = (batch_size, n_actions)
        q_cf_all = self.critic(o_rep, act_cf).reshape(self.args.batch_size, self.args.n_actions)

        probs = self.actor(o)
        probs = (1 - epsilon) * probs + epsilon / float(self.args.n_actions)
        log_probs = torch.log(torch.sum(probs * u_onehot, dim=1, keepdim=True)+1e-15)

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


