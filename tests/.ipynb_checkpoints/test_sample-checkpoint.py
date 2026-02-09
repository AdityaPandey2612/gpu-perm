import torch
from per.tree import GpuSumTree

def test_sampling_distribution_reasonable():
    if not torch.cuda.is_available():
        return

    cap = 128
    t = GpuSumTree(cap)

    idx = torch.arange(cap, device="cuda", dtype=torch.int64)
    p = (idx.float() + 1.0)  # priorities 1..cap
    t.update(idx, p)

    # sample many times
    B = 200_000
    samp = t.sample(B)

    hist = torch.bincount(samp, minlength=cap).float()
    hist /= hist.sum()

    target = p / p.sum()
    l1 = torch.abs(hist - target).sum().item()

    # forgiving threshold to avoid flakiness
    assert l1 < 0.10, f"L1 error too large: {l1}"
