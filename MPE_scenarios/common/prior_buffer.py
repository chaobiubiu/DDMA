import numpy as np


class DetecBuffer:
    def __init__(self, args, obs_dim):
        self.args = args
        self.max_size = args.prior_buffer_size
        # 这里所有agent共用一prior_buffer，因此假设所有agent.obs_shape相同
        self.obs_inputs = np.zeros((self.max_size, obs_dim))
        # torch中cross_entropy使用时label不能为one-hot encoding，必须是具体某个数
        self.labels = np.zeros((self.max_size, 1))
        self.kl_values = np.zeros((self.max_size, 1))
        self.current_idx = 0
        self.threshold = 0
        self.percentile = args.prior_training_percentile
        self.pos_idx, self.neg_idx = [], []
        self.pos_t, self.neg_t = 0, 0

    def get_threshold(self):
        self.threshold = np.percentile(self.kl_values, self.percentile)
        return self.threshold

    def get_labels(self):
        threshold = self.get_threshold()
        for i in range(self.max_size):
            if self.kl_values[i] <= threshold:
                # self.labels[i] = [1, 0]，如果小于给定阈值，则表示当前处于非交互位置
                self.labels[i] = 0
                self.neg_idx.append(i)
            else:
                # self.labels[i] = [0, 1]，如果大于给定阈值，则表示当前处于交互位置
                self.labels[i] = 1
                self.pos_idx.append(i)

    def get_shuffle(self):
        np.random.shuffle(self.pos_idx)
        np.random.shuffle(self.neg_idx)

    def clear(self):
        self.pos_idx = []
        self.neg_idx = []

    def insert(self, size, obs_i, kl_values_i):
        # 判断新数据量是否会造成replay_buffer overflow，如果overflow则只填充当前buffer剩余空间量的数据
        if self.current_idx + size <= self.max_size:
            self.obs_inputs[self.current_idx: (self.current_idx + size), :] = obs_i
            self.kl_values[self.current_idx: (self.current_idx + size), :] = kl_values_i
            self.current_idx += size
            return False
        else:
            idx_tmp = self.max_size - self.current_idx
            self.obs_inputs[self.current_idx:, :] = obs_i[0: idx_tmp, :]
            self.kl_values[self.current_idx:, :] = kl_values_i[0: idx_tmp, :]
            self.current_idx = 0
            self.get_labels()
            self.get_shuffle()
            return True

    def get_samples(self, prior_batch_size):
        if (self.pos_t + int(prior_batch_size / 2)) < len(self.pos_idx):
            pos_indexes = self.pos_idx[self.pos_t: (self.pos_t + int(prior_batch_size / 2))]
            self.pos_t += int(prior_batch_size / 2)
        else:
            pos_indexes = self.pos_idx[self.pos_t:]
            self.pos_t = 0
        if (self.neg_t + int(prior_batch_size / 2)) < len(self.neg_idx):
            neg_indexes = self.neg_idx[self.neg_t: (self.neg_t + int(prior_batch_size / 2))]
            self.neg_t += int(prior_batch_size / 2)
        else:
            neg_indexes = self.neg_idx[self.neg_t:]
            self.neg_t = 0
        all_indexes = pos_indexes + neg_indexes
        out_obs = self.obs_inputs[np.array(all_indexes)]
        out_labels = self.labels[np.array(all_indexes)]
        return out_obs, out_labels
