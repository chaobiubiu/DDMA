import torch
import os
import numpy as np
from network.detector import Detector
from network.actor_critic import Actor, Critic
from common.utils import onehot_from_logits, gumbel_softmax
from network.relevant_network import Left_Actor, Left_Critic, Right_Actor, Right_Critic

CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class DDPG:
    def __init__(self, args, agent_id, prior_buffer):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.left_actor_network = Left_Actor(args, agent_id)
        # self.right_actor_network = Right_Actor(args, agent_id)

        self.left_critic_network = Left_Critic(args, agent_id)
        # self.right_critic_network = Right_Critic(args, agent_id)

        self.left_actor_target_network = Left_Actor(args, agent_id)
        # self.right_actor_target_network = Right_Actor(args, agent_id)

        self.left_critic_target_network = Left_Critic(args, agent_id)
        # self.right_critic_target_network = Right_Critic(args, agent_id)

        # Interaction detector that indicates whether agents are in interaction now.
        # self.detec = Detector(args, agent_id)

        # Prior Buffer
        self.prior_buffer = prior_buffer

        if args.cuda:
            self.left_actor_network.cuda()
            # self.right_actor_network.cuda()
            self.left_critic_network.cuda()
            # self.right_critic_network.cuda()

            self.left_actor_target_network.cuda()
            # self.right_actor_target_network.cuda()
            self.left_critic_target_network.cuda()
            # self.right_critic_target_network.cuda()

            # self.detec.cuda()

        # load the weights into the target networks
        self.left_actor_target_network.load_state_dict(self.left_actor_network.state_dict())
        # self.right_actor_target_network.load_state_dict(self.right_actor_network.state_dict())
        self.left_critic_target_network.load_state_dict(self.left_critic_network.state_dict())
        # self.right_critic_target_network.load_state_dict(self.right_critic_network.state_dict())

        if args.load_model:
            if os.path.exists(self.args.model_save_dir + '/evaluate_model/' + '/0_left_actor_params.pkl'):
                # path_detec = self.args.model_save_dir + '/evaluate_model/{}_detector_params.pkl'.format(self.agent_id)
                path_actor = self.args.model_save_dir + '/evaluate_model/{}_left_actor_params.pkl'.format(self.agent_id)
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                # self.detec.load_state_dict(torch.load(path_detec, map_location=map_location))
                self.left_actor_network.load_state_dict(torch.load(path_actor, map_location=map_location))
                print('Successfully load the network: {}'.format(path_actor))
            else:
                raise Exception("No network!")

        # create the optimizer
        # self.actor_optim = torch.optim.Adam(list(self.left_actor_network.parameters()) + list(self.right_actor_network.parameters()), lr=self.args.lr_actor)
        self.actor_optim = torch.optim.Adam(self.left_actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.left_critic_network.parameters(), lr=self.args.lr_critic)
        # self.detec_optim = torch.optim.Adam(self.detec.parameters(), lr=self.args.lr_detector)

        # self.left_actor_optim = torch.optim.Adam(self.left_actor_network.parameters(), lr=self.args.lr_actor)
        # self.left_critic_optim = torch.optim.Adam(self.left_critic_network.parameters(), lr=self.args.lr_critic)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.left_actor_target_network.parameters(), self.left_actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        # for target_param, param in zip(self.right_actor_target_network.parameters(), self.right_actor_network.parameters()):
        #     target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.left_critic_target_network.parameters(), self.left_critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        # for target_param, param in zip(self.right_critic_target_network.parameters(), self.right_critic_network.parameters()):
        #     target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # def get_flag(self, obs_i):
    #     out_i, out_i_prob = self.detec(obs_i)
    #     # out[:, 1]如果大于0.5那么认为此处为交互点，此时需要使用其他智能体信息进行决策
    #     out_flag = torch.gt(out_i_prob[:, 1], 0.5).unsqueeze(1)
    #     return out_flag

    # update the network
    def train(self, transitions, other_agents, logger):
        if self.train_step % 2 == 0:
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
            others_o_next, others_u_next = [], []
            others_o, others_u = [], []

            with torch.no_grad():
                # observation = (o_self, o_others)
                # extra_pi_in = self.right_actor_target_network(o_next[agent_id][:, self.args.obs_self_shape:])
                # u_self_next = onehot_from_logits(self.left_actor_target_network(o_next[agent_id][:, :self.args.obs_self_shape], extra_pi_in))
                u_self_next = onehot_from_logits(self.left_actor_target_network(o_next[self.agent_id]))
                if self.args.cuda:
                    u_self_next = u_self_next.cuda()
                # right_critic_target_network的输入为(others_o_next, others_u_next)
                # left_critic_target_network的输入为(o_next_self, u_self_next)
                q_next = self.left_critic_target_network(o_next[self.agent_id].unsqueeze(0), u_self_next.unsqueeze(0), None)
                target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
            # target的相关计算不考虑interactive和individual setting
            # interact_flag = self.get_flag(o[self.agent_id])
            # common_target_q = target_q * interact_flag
            # individual_target_q = target_q - common_target_q
            # critic_loss
            # critic_loss, left_critic_loss = 0.0, 0.0
            # extra_q_in = self.right_critic_network(others_o, others_u)
            # 这里interact_flag相当于mask的作用，将非交互状态处对应的内容置0
            # common_extra_q_in = extra_q_in * interact_flag
            q_value = self.left_critic_network(o[self.agent_id].unsqueeze(0), u[self.agent_id].unsqueeze(0), None)
            critic_loss = (target_q - q_value).pow(2).mean()
            # actor loss
            # extra_pi_in = self.right_actor_network(o[self.agent_id][:, self.args.obs_self_shape:])
            # # 同样这里interact_flag相当于mask的作用，将非交互状态处的内容置0
            # common_extra_pi_in = extra_pi_in * interact_flag
            # u[self.agent_id] = gumbel_softmax(self.left_actor_network(o[self.agent_id][:, :self.args.obs_self_shape], common_extra_pi_in), hard=True)
            u[self.agent_id] = gumbel_softmax(self.left_actor_network(o[self.agent_id]), hard=True)
            support_q = self.left_critic_network(o[self.agent_id].unsqueeze(0), u[self.agent_id].unsqueeze(0), None)
            # actor_loss = - (support_q * interact_flag).mean()
            # left_actor_loss = -(support_q - support_q * interact_flag).mean()
            actor_loss = - (support_q).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # if left_actor_loss != 0:
            #     self.left_actor_optim.zero_grad()
            #     left_actor_loss.backward()
            #     self.left_actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:  # 每2000步save model一次
            self.save_model(self.train_step)
        self.train_step += 1


    def save_model(self, train_step):
        if train_step is not None:
            num = str(train_step // self.args.save_rate)
        else:
            num = 'final'
        if not self.args.hallway:
            model_path = os.path.join(self.args.model_save_dir, self.args.scenario_name)
        else:
            model_path = os.path.join(self.args.model_save_dir, 'hallway')
        model_path = os.path.join(model_path, self.args.algorithm)
        model_path = os.path.join(model_path, 'order_%d'%self.args.order)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.left_actor_network.state_dict(), model_path + '/' + num + '_left_actor_params.pkl')
        # torch.save(self.right_actor_network.state_dict(), model_path + '/' + num + '_right_actor_params.pkl')
        torch.save(self.left_critic_network.state_dict(),  model_path + '/' + num + '_left_critic_params.pkl')



