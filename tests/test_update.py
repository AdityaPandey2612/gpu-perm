import torch
from per.tree import GpuSumTree

def test_sum_invariant_after_random_updates():
    if not torch.cuda.is_available():
        return

    cap = 1024
    t = GpuSumTree(cap)

    # random updates (with duplicates)
    n = 5000
    idx = torch.randint(0, cap, (n,), device="cuda", dtype=torch.int64)
    new_p = torch.rand(n, device="cuda", dtype=torch.float32)

    t.update(idx, new_p)

    leaves_sum = t.leaf_values().sum()
    root = t.total

    assert torch.allclose(root, leaves_sum, rtol=1e-4, atol=1e-5), (root.item(), leaves_sum.item())

def test_internal_node_invariant_spotcheck():
    if not torch.cuda.is_available():
        return

    cap = 256
    t = GpuSumTree(cap)

    # set all leaves explicitly (unique update)
    idx = torch.arange(cap, device="cuda", dtype=torch.int64)
    new_p = torch.rand(cap, device="cuda", dtype=torch.float32)
    t.update(idx, new_p)

    # spot check random internal nodes
    # valid internal indices: [1, L-1]
    L = t.L
    internal = torch.randint(1, L, (50,), device="cuda", dtype=torch.int64).tolist()

    for i in internal:
        left = 2 * i
        right = left + 1
        # right child might be out of bounds only if i >= L, but we sampled < L
        s = t.tree[left] + t.tree[right]
        assert torch.allclose(t.tree[i], s, rtol=1e-4, atol=1e-5), f"node {i} mismatch"
