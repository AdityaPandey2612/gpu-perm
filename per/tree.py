import torch
from ._ext import per_ext

class GpuSumTree:
    """
    Flat segment tree:
      - L = next power of two >= capacity
      - tree is size 2*L, float32 on CUDA
      - root at index 1
      - leaves at [L : L+capacity)
    """
    def __init__(self, capacity: int, device: str = "cuda"):
        assert capacity > 0
        self.capacity = int(capacity)
        self.L = 1 << (self.capacity - 1).bit_length()
        self.device = torch.device(device)
        self.tree = torch.zeros(2 * self.L, device=self.device, dtype=torch.float32)

    @property
    def total(self) -> torch.Tensor:
        return self.tree[1]

    def leaf_values(self) -> torch.Tensor:
        return self.tree[self.L : self.L + self.capacity]

    def update(self, idx: torch.Tensor, new_p: torch.Tensor):
        """
        idx: int64 CUDA tensor of shape [n] in [0, capacity)
        new_p: float32 CUDA tensor of shape [n]
        """
        per_ext.update(self.tree, idx, new_p, self.L)
    
    def sample(self, batch_size: int) -> torch.Tensor:
        # total priority mass on GPU
        total = self.tree[1].item()
        if total <= 0.0:
            raise RuntimeError("Cannot sample: total priority is 0")

        # stratified sampling: u_i in [i*seg, (i+1)*seg)
        seg = total / batch_size
        u = (torch.arange(batch_size, device=self.tree.device, dtype=torch.float32) +
             torch.rand(batch_size, device=self.tree.device, dtype=torch.float32)) * seg

        return per_ext.sample(self.tree, u, self.L, self.capacity)

    def leaf_priorities(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: int64 CUDA tensor [B] in [0, capacity)
        returns: float32 CUDA tensor [B] of leaf priorities
        """
        return self.tree[self.L + idx]

    def total_priority(self) -> torch.Tensor:
        """Return total priority mass (CUDA scalar tensor)."""
        return self.tree[1]

    def update_coalesced(self, idx, new_p):
        per_ext.update_coalesced(self.tree, idx, new_p, self.L)