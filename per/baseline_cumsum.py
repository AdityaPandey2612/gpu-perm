import torch

class GpuCumsumSampler:
    """
    Baseline GPU sampler using cumsum + searchsorted.
    Priorities stored as a flat CUDA tensor.
    """
    def __init__(self, capacity, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.priorities = torch.zeros(capacity, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, idx, new_p):
        self.priorities[idx] = new_p

    @torch.no_grad()
    def sample(self, batch_size):
        cdf = torch.cumsum(self.priorities, dim=0)
        total = cdf[-1]

        seg = total / batch_size
        u = (torch.arange(batch_size, device=self.device) +
             torch.rand(batch_size, device=self.device)) * seg

        idx = torch.searchsorted(cdf, u, right=False)
        idx.clamp_(0, self.capacity - 1)
        return idx
