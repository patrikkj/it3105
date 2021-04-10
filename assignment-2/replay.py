from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size=512):
        self.buffer_size = buffer_size
        self._buffer_x = deque(maxlen=buffer_size)
        self._buffer_y = deque(maxlen=buffer_size)

    def add(self, x, y):
        self._buffer_x.append(x)
        self._buffer_y.append(y)

    def clear(self):
        self._buffer_x.clear()
        self._buffer_y.clear()

    def fetch_minibatch(self, batch_size=32):
        current_size = len(self._buffer_x)
        indices = np.random.choice(current_size, size=min(batch_size, current_size), replace=False)
        x_batch = np.array(self._buffer_x)[indices, ...]
        y_batch = np.array(self._buffer_y)[indices, ...]
        return x_batch, y_batch
