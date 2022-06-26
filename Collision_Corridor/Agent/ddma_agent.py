import numpy as np
import torch
from torch.distributions import Categorical
from algorithm.ddma import DDMA

class DDMA_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = DDMA(args, agent_id)

    def select_action(self, o, epsilon, curr_episode):
        inputs = torch.tensor(o, dtype=torch.float32)
        input_local = inputs[self.agent_id]
        input_others = inputs[np.arange(self.args.n_agents) != self.agent_id, :]
        input_local = input_local.unsqueeze(0)
        input_others = input_others.unsqueeze(0).reshape(1, -1)
        if self.args.cuda:
            input_local = input_local.cuda()
            input_others = input_others.cuda()

        probs = self.policy.actor(input_local, input_others).squeeze(0)
        if epsilon == 0:
            act_sample = torch.argmax(probs)
        else:
            if self.args.stage > 1 and self.args.use_overlap:
                # training period
                interact_strength = self.policy.pred_strength(torch.cat([input_local, input_others], dim=-1))
                # interact_strength = interact_strength.cpu().numpy()
                # 如果预测出此状态的交互概率大于0.5则需要探索，否则就不探索该状态
                explor_strength = np.max([0, interact_strength - 0.5])
                if explor_strength != 0:
                    # 如果对于当前交互状态预测的交互概率越大，这个地方的探索力度越大
                    explor_strength = interact_strength * 2
                    # explor_strength = interact_strength
                # extra_explor = explor_strength * ((curr_episode * 2) / self.args.max_episodes)
                extra_explor = explor_strength / (np.sqrt(curr_episode))
                epsilon += extra_explor
                if epsilon > 1:
                    print('label')
                epsilon = np.min([epsilon, 1])
            else:
                epsilon = epsilon

            probs = (1 - epsilon) * probs + (epsilon / float(self.args.n_actions))
            act_sample = Categorical(probs).sample().long()
        # act_sample = Categorical(probs).sample().long()
        u = act_sample.cpu().numpy()
        return u.copy()

    def learn(self, transitions, epsilon, other_agents):
        self.policy.train(transitions, epsilon, other_agents)

    def update_detec(self, batch_inps, batch_labels, iter=None, logger=None):
        self.policy.update_detec(batch_inps, batch_labels, iter, logger)

    def make_data(self, o_cat, u_cat):
        # obs_inps.shape=(detec_buffer_size, obs_shape*n_agents)
        obs_inps, kl_values = self.policy.output_detec(o_cat, u_cat)
        return obs_inps, kl_values

