import torch
from dataclasses import dataclass
from .tree import GpuSumTree

@dataclass
class PERBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor

class GPUReplayPER:
    """
    Minimal GPU PER replay:
      - stores transitions in CUDA tensors
      - priorities in a GPU sum-tree
      - proportional sampling + importance weights
    """
    def __init__(
        self,
        capacity: int,
        obs_shape,
        device: str = "cuda",
        alpha: float = 0.6,
        beta0: float = 0.4,
        eps: float = 1e-6,
        dtype_obs=torch.float32,
    ):
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.eps = float(eps)

        # ring buffer state (keep on CPU for simplicity)
        self.pos = 0
        self.size = 0

        # storage on GPU
        self.obs = torch.empty((capacity, *obs_shape), device=self.device, dtype=dtype_obs)
        self.next_obs = torch.empty((capacity, *obs_shape), device=self.device, dtype=dtype_obs)
        self.act = torch.empty((capacity,), device=self.device, dtype=torch.int64)
        self.rew = torch.empty((capacity,), device=self.device, dtype=torch.float32)
        self.done = torch.empty((capacity,), device=self.device, dtype=torch.float32)

        # priorities
        self.tree = GpuSumTree(capacity, device=device)
        self._max_priority = 1.0  # CPU scalar is fine

    def __len__(self):
        return self.size

    @torch.no_grad()
    def add(self, obs, act, rew, next_obs, done):
        """
        obs/next_obs can be numpy or torch; will be copied to GPU.
        done should be bool or 0/1.
        """
        i = self.pos

        # copy to GPU tensors
        self.obs[i].copy_(torch.as_tensor(obs, device=self.device))
        self.next_obs[i].copy_(torch.as_tensor(next_obs, device=self.device))
        self.act[i] = int(act)
        self.rew[i] = float(rew)
        self.done[i] = float(done)

        # set priority for this slot to current max (common PER trick)
        p = torch.tensor([self._max_priority], device=self.device, dtype=torch.float32)
        idx = torch.tensor([i], device=self.device, dtype=torch.int64)
        self.tree.update(idx, p)

        # advance ring
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, batch_size: int, step: int, beta_steps: int = 200_000) -> PERBatch:
        """
        beta anneals from beta0 to 1.0 over beta_steps.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        # beta schedule
        frac = min(1.0, step / float(beta_steps))
        beta = self.beta0 + frac * (1.0 - self.beta0)

        # sample indices proportional to priorities
        idx = self.tree.sample(batch_size)  # int64 CUDA [B]

        # gather transitions
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        act = self.act[idx]
        rew = self.rew[idx]
        done = self.done[idx]

        # importance weights
        total = self.tree.total_priority()  # CUDA scalar
        prio = self.tree.leaf_priorities(idx).clamp_min(self.eps)
        p = prio / total.clamp_min(self.eps)

        N = float(self.size)
        weights = (N * p).pow(-beta)
        weights = weights / weights.max().clamp_min(self.eps)

        return PERBatch(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done, weights=weights, indices=idx)

    @torch.no_grad()
    def update_priorities(self, indices: torch.Tensor, td_error: torch.Tensor):
        """
        indices: int64 CUDA [B]
        td_error: float tensor [B] on CUDA
        priority = (|td| + eps)^alpha
        """
        p = (td_error.abs() + self.eps).pow(self.alpha).to(torch.float32)
        self.tree.update(indices, p)

        # track max priority on CPU to init new entries
        max_p = float(p.max().item())
        if max_p > self._max_priority:
            self._max_priority = max_p
