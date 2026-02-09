import time
import torch
import numpy as np

from per.tree import GpuSumTree
from per.baseline_cumsum import GpuCumsumSampler
from per.cpu_tree import CpuSumTree


def time_cuda(fn, iters=100):
    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def time_cpu(fn, iters=200):
    for _ in range(10):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters


def benchmark(capacity, batch_size, iters_gpu=50, iters_cpu=200):
    device = "cuda"

    # GPU random indices + priorities
    idx_gpu = torch.randint(0, capacity, (batch_size,), device=device, dtype=torch.int64)
    p_gpu = torch.rand(batch_size, device=device, dtype=torch.float32)

    # ---------- GPU PER (naive + coalesced)
    tree = GpuSumTree(capacity, device=device)
    tree.update(idx_gpu, p_gpu)

    def gpu_per_sample():
        tree.sample(batch_size)

    def gpu_per_update_naive():
        tree.update(idx_gpu, p_gpu)

    def gpu_per_update_coalesced():
        tree.update_coalesced(idx_gpu, p_gpu)

    t_gpu_sample = time_cuda(gpu_per_sample, iters_gpu)
    t_gpu_update_naive = time_cuda(gpu_per_update_naive, iters_gpu)
    t_gpu_update_coal = time_cuda(gpu_per_update_coalesced, iters_gpu)

    # ---------- GPU cumsum baseline
    base = GpuCumsumSampler(capacity, device=device)
    base.update(idx_gpu, p_gpu)

    def gpu_cumsum_sample():
        base.sample(batch_size)

    def gpu_cumsum_update():
        base.update(idx_gpu, p_gpu)

    t_cumsum_sample = time_cuda(gpu_cumsum_sample, iters_gpu)
    t_cumsum_update = time_cuda(gpu_cumsum_update, iters_gpu)

    # ---------- CPU PER (numpy)
    idx_cpu = idx_gpu.cpu().numpy().astype(np.int64)
    p_cpu = p_gpu.cpu().numpy().astype(np.float32)

    cpu_tree = CpuSumTree(capacity)
    cpu_tree.update(idx_cpu, p_cpu)

    def cpu_update():
        cpu_tree.update(idx_cpu, p_cpu)

    def cpu_sample():
        total = cpu_tree.total
        seg = total / batch_size
        u = (np.arange(batch_size, dtype=np.float32) +
             np.random.rand(batch_size).astype(np.float32)) * seg
        cpu_tree.sample(u)

    t_cpu_sample = time_cpu(cpu_sample, iters_cpu)
    t_cpu_update = time_cpu(cpu_update, iters_cpu)

    # Throughputs
    cpu_sample_sps = batch_size / t_cpu_sample
    cpu_update_ups = batch_size / t_cpu_update

    gpu_sample_sps = batch_size / t_gpu_sample
    gpu_update_naive_ups = batch_size / t_gpu_update_naive
    gpu_update_coal_ups = batch_size / t_gpu_update_coal

    cumsum_sample_sps = batch_size / t_cumsum_sample
    cumsum_update_ups = batch_size / t_cumsum_update

    # Speedups (updates)
    coal_over_naive = gpu_update_coal_ups / max(1e-9, gpu_update_naive_ups)
    naive_over_cpu = gpu_update_naive_ups / max(1e-9, cpu_update_ups)
    coal_over_cpu = gpu_update_coal_ups / max(1e-9, cpu_update_ups)
    cumsum_over_naive = cumsum_update_ups / max(1e-9, gpu_update_naive_ups)

    # Speedups (sampling)
    gpu_sample_over_cpu = gpu_sample_sps / max(1e-9, cpu_sample_sps)
    cumsum_sample_over_gpu = cumsum_sample_sps / max(1e-9, gpu_sample_sps)

    return {
        "capacity": capacity,
        "batch": batch_size,

        "cpu_sample_sps": cpu_sample_sps,
        "cpu_update_ups": cpu_update_ups,

        "gpu_sample_sps": gpu_sample_sps,
        "gpu_update_naive_ups": gpu_update_naive_ups,
        "gpu_update_coal_ups": gpu_update_coal_ups,

        "cumsum_sample_sps": cumsum_sample_sps,
        "cumsum_update_ups": cumsum_update_ups,

        "coal_over_naive": coal_over_naive,
        "naive_over_cpu": naive_over_cpu,
        "coal_over_cpu": coal_over_cpu,
        "cumsum_over_naive": cumsum_over_naive,

        "gpu_sample_over_cpu": gpu_sample_over_cpu,
        "cumsum_sample_over_gpu": cumsum_sample_over_gpu,
    }


def main():
    capacities = [2**14, 2**16, 2**18]
    batches = [32, 64, 128, 256, 512]

    results = []
    for cap in capacities:
        for B in batches:
            print(f"Running cap={cap}, batch={B}")
            results.append(benchmark(cap, B))

    print("\n=== RESULTS (samples/sec, updates/sec) ===")
    header = (
        "capacity batch | "
        "CPU samp  CPU upd | "
        "GPU samp  GPU upd_naive  GPU upd_coal | "
        "cumsum samp  cumsum upd | "
        "coal/naive  naive/CPU  coal/CPU  cumsum/naive  gpuSamp/CPU"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['capacity']:>8} {r['batch']:>5} | "
            f"{r['cpu_sample_sps']:>7.0f} {r['cpu_update_ups']:>7.0f} | "
            f"{r['gpu_sample_sps']:>7.0f} {r['gpu_update_naive_ups']:>13.0f} {r['gpu_update_coal_ups']:>12.0f} | "
            f"{r['cumsum_sample_sps']:>10.0f} {r['cumsum_update_ups']:>10.0f} | "
            f"{r['coal_over_naive']:>10.2f} {r['naive_over_cpu']:>9.2f} {r['coal_over_cpu']:>8.2f} "
            f"{r['cumsum_over_naive']:>12.2f} {r['gpu_sample_over_cpu']:>10.2f}"
        )


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
