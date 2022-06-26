import threading
import numpy as np

class DetecBuffer:
    def __init__(self, args):
        self.size = args.detec_buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        self.labels = np.zeros((self.args.n_agents, self.size, 1))
        self.kl_values = np.zeros((self.args.n_agents, self.size, 1))
        self.pos_idx = [None for _ in range(self.args.n_agents)]
        self.neg_idx = [None for _ in range(self.args.n_agents)]
        self.pos_t = [0 for _ in range(self.args.n_agents)]
        self.neg_t = [0 for _ in range(self.args.n_agents)]
        self.threshold = [None for _ in range(self.args.n_agents)]
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['inp_%d' % i] = np.zeros([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]

    def get_labels(self, obs_inputs, kl_values, agent_id):
        self.buffer['inp_%d' % agent_id] = obs_inputs
        self.kl_values[agent_id] = kl_values
        self.threshold[agent_id] = np.percentile(self.kl_values[agent_id], self.args.detec_percentile)
        self.pos_idx[agent_id] = np.where(self.kl_values[agent_id] >= self.threshold[agent_id])[0]
        self.neg_idx[agent_id] = np.where(self.kl_values[agent_id] < self.threshold[agent_id])[0]
        self.labels[agent_id][self.pos_idx[agent_id]] = 1
        self.labels[agent_id][self.neg_idx[agent_id]] = 0
        np.random.shuffle(self.pos_idx[agent_id])
        np.random.shuffle(self.neg_idx[agent_id])

    # sample inputs and labels to train the detector
    def sample(self, batch_size, agent_id):
        pos_indexes = np.random.choice(self.pos_idx[agent_id], int(batch_size / 2))
        neg_indexes = np.random.choice(self.neg_idx[agent_id], int(batch_size / 2))
        all_indexes = np.hstack((pos_indexes, neg_indexes))
        out_inps = self.buffer['inp_%d' % agent_id][all_indexes]
        out_labels = self.labels[agent_id][all_indexes]
        return out_inps, out_labels

    # sample inputs and labels to train the detector
    # def sample(self, batch_size, agent_id):
    #     if (self.pos_t[agent_id] + int(batch_size / 2)) < len(self.pos_idx[agent_id]):
    #         # pos_indexes = np.random.choice(self.pos_idx[agent_id], int(batch_size / 2))
    #         pos_indexes = self.pos_idx[agent_id][self.pos_t[agent_id]: (self.pos_t[agent_id] + int(batch_size / 2))]
    #         self.pos_t[agent_id] += int(batch_size / 2)
    #     else:
    #         pos_indexes = self.pos_idx[agent_id][self.pos_t[agent_id]:]
    #         self.pos_t[agent_id] = 0
    #
    #     if (self.neg_t[agent_id] + int(batch_size / 2)) < len(self.neg_idx[agent_id]):
    #         # neg_indexes = np.random.choice(self.neg_idx[agent_id], int(batch_size / 2))
    #         neg_indexes = self.neg_idx[agent_id][self.neg_t[agent_id]: (self.neg_t[agent_id] + int(batch_size / 2))]
    #         self.neg_t[agent_id] += int(batch_size / 2)
    #     else:
    #         neg_indexes = self.neg_idx[agent_id][self.neg_t[agent_id]:]
    #         self.neg_t[agent_id] = 0
    #
    #     all_indexes = np.hstack((pos_indexes, neg_indexes))
    #     out_inps = self.buffer['inp_%d' % agent_id][all_indexes]
    #     out_labels = self.labels[agent_id][all_indexes]
    #     return out_inps, out_labels

    def sample_all(self):
        temp_buffer = {}
        for key in self.buffer.keys():
            if 'o' in key or 'u' in key:
                temp_buffer[key] = self.buffer[key]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            # 当还有足够的空余位置时，直接添加入buffer中
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            # 当剩余空余位置不足时，一部分直接添加入空余位置，另一部分overflow随机覆盖掉之前的old memories
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            # 当没有空余位置时，直接覆盖之前的old memories
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
