import numpy as np

class CpuSumTree:
    """
    Flat sum-tree on CPU (numpy).
    - capacity = N (not forced to power of 2)
    - internal nodes stored in tree[1..2*L-1], leaves at [L..L+N-1]
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.L = 1 << (self.capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.L, dtype=np.float32)

    def update(self, idx: np.ndarray, new_p: np.ndarray):
        # idx: [B] int64, new_p: [B] float32
        for i, p in zip(idx, new_p):
            pos = self.L + int(i)
            delta = float(p) - float(self.tree[pos])
            self.tree[pos] = float(p)

            pos >>= 1
            while pos >= 1:
                self.tree[pos] += delta
                pos >>= 1

    def sample(self, u: np.ndarray) -> np.ndarray:
        # u: [B] float32 in [0, tree[1])
        out = np.empty(u.shape[0], dtype=np.int64)
        for j, x0 in enumerate(u):
            x = float(x0)
            node = 1
            while node < self.L:
                left = node << 1
                s_left = float(self.tree[left])
                if x <= s_left:
                    node = left
                else:
                    x -= s_left
                    node = left + 1
            leaf = node - self.L
            if leaf >= self.capacity:
                leaf = self.capacity - 1
            out[j] = leaf
        return out

    @property
    def total(self) -> float:
        return float(self.tree[1])
