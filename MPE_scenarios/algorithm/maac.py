import torch
import os
import numpy as np
from network.maac_actor_critic import Actor, Attention_Critic
from common.utils import hard_update, soft_update

class MAAC:
    def __init__(self, args, agent_id):
        self.args = args
        self.stage = args.stage
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.actor_target_network = Actor(args, agent_id)

        self.critic_network = Attention_Critic(args, agent_id)
        self.critic_target_network = Attention_Critic(args, agent_id)

        if args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # Only done in evaluation
        if self.args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)):
                path_actor = self.args.model_save_dir + '/evaluate_model/{}_actor_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor_network.load_state_dict(torch.load(path_actor, map_location=map_location))
                print('Successfully load the evaluation network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # load the weights into the target networks
        hard_update(self.actor_target_network, self.actor_network)
        hard_update(self.critic_target_network, self.critic_network)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

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
                    ind_obs_next = o_next[agent_id]
                    if agent_id == self.agent_id:
                        u_next.append(self.actor_target_network(ind_obs_next))
                    else:
                        u_next.append(other_agents[index].policy.actor_target_network(ind_obs_next))
                        index += 1
                u_next = torch.stack(u_next)
                if self.args.cuda:
                    u_next = u_next.cuda()

                q_next = self.critic_target_network(o_next[:, :, :self.args.input_dim_self], u_next).detach()

                target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

            # the q loss
            q_value = self.critic_network(o[:, :, :self.args.input_dim_self], u)
            critic_loss = (target_q - q_value).pow(2).mean()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # the actor loss
            u[self.agent_id] = self.actor_network(o[self.agent_id])
            actor_loss = - self.critic_network(o[:, :, :self.args.input_dim_self], u).mean()

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


