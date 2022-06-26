import threading
import numpy as np

class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape])
            self.buffer['u_%d' % i] = np.empty([self.size])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape])
        self.buffer['done'] = np.empty([self.size])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next, done):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.args.n_agents):
            # with self.lock:
            #     self.buffer['o_%d' % i][idxs] = o[i]
            #     self.buffer['u_%d' % i][idxs] = u[i]
            #     self.buffer['r_%d' % i][idxs] = r[i]
            #     self.buffer['o_next_%d' % i][idxs] = o_next[i]
            self.buffer['o_%d' % i][idxs] = o[i]
            self.buffer['u_%d' % i][idxs] = u[i]
            self.buffer['r_%d' % i][idxs] = r[i]
            self.buffer['o_next_%d' % i][idxs] = o_next[i]
        self.buffer['done'][idxs] = done

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
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
