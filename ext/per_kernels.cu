#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

static inline void check_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// GPU UPDATE (naive)
__global__ void update_kernel(
    float* tree,              // size: 2*L (index 0 unused, root at 1)
    const int64_t* idx,        // leaf indices in [0, capacity)
    const float* new_p,        // new priorities
    int64_t n,
    int64_t L
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int64_t pos = L + idx[i];

    // atomic exchange at leaf to handle duplicates safely
    float old = atomicExch(&tree[pos], new_p[i]);
    float delta = new_p[i] - old;

    // propagate delta to parents
    pos >>= 1;
    while (pos >= 1) {
        atomicAdd(&tree[pos], delta);
        pos >>= 1;
    }
}

void update_cuda(torch::Tensor tree,
                 torch::Tensor idx,
                 torch::Tensor new_p,
                 int64_t L) {
    check_cuda(tree, "tree");
    check_cuda(idx, "idx");
    check_cuda(new_p, "new_p");

    TORCH_CHECK(tree.dtype() == torch::kFloat32, "tree must be float32");
    TORCH_CHECK(idx.dtype() == torch::kInt64, "idx must be int64");
    TORCH_CHECK(new_p.dtype() == torch::kFloat32, "new_p must be float32");
    TORCH_CHECK(tree.dim() == 1, "tree must be 1D");
    TORCH_CHECK(idx.dim() == 1 && new_p.dim() == 1, "idx/new_p must be 1D");
    TORCH_CHECK(idx.numel() == new_p.numel(), "idx and new_p must match size");

    const int64_t n = idx.numel();
    if (n == 0) return;

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    update_kernel<<<blocks, threads, 0, stream>>>(
        tree.data_ptr<float>(),
        idx.data_ptr<int64_t>(),
        new_p.data_ptr<float>(),
        n,
        L
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// GPU UPDATE (warp-coalesced atomics)
__device__ __forceinline__ void warp_aggregated_atomic_add(float* addr, float val) {
    unsigned mask = 0xFFFFFFFFu;

    // Group threads in this warp that have the same address
    unsigned group = __match_any_sync(mask, (unsigned long long)addr);

    // Reduce val within the group
    float sum = val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(group, sum, offset);
        sum += other;
    }

    // Leader lane does the atomic
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1;
    if (lane == leader) {
        atomicAdd(addr, sum);
    }
}

__global__ void update_kernel_coalesced(
    float* tree,
    const int64_t* idx,
    const float* new_p,
    int64_t n,
    int64_t L
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int64_t pos = L + idx[i];

    // leaf exchange still needed to compute delta correctly
    float old = atomicExch(&tree[pos], new_p[i]);
    float delta = new_p[i] - old;

    // propagate delta to parents (warp-coalesced atomics)
    pos >>= 1;
    while (pos >= 1) {
        warp_aggregated_atomic_add(&tree[pos], delta);
        pos >>= 1;
    }
}

void update_coalesced_cuda(torch::Tensor tree,
                           torch::Tensor idx,
                           torch::Tensor new_p,
                           int64_t L) {
    check_cuda(tree, "tree");
    check_cuda(idx, "idx");
    check_cuda(new_p, "new_p");

    TORCH_CHECK(tree.dtype() == torch::kFloat32, "tree must be float32");
    TORCH_CHECK(idx.dtype() == torch::kInt64, "idx must be int64");
    TORCH_CHECK(new_p.dtype() == torch::kFloat32, "new_p must be float32");
    TORCH_CHECK(tree.dim() == 1, "tree must be 1D");
    TORCH_CHECK(idx.dim() == 1 && new_p.dim() == 1, "idx/new_p must be 1D");
    TORCH_CHECK(idx.numel() == new_p.numel(), "idx and new_p must match size");

    const int64_t n = idx.numel();
    if (n == 0) return;

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    update_kernel_coalesced<<<blocks, threads, 0, stream>>>(
        tree.data_ptr<float>(),
        idx.data_ptr<int64_t>(),
        new_p.data_ptr<float>(),
        n,
        L
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// GPU SAMPLE
__global__ void sample_kernel(
    const float* tree,
    const float* u,
    int64_t* out_idx,
    int64_t n,
    int64_t L,
    int64_t capacity
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = u[i];
    int64_t node = 1;

    while (node < L) {
        int64_t left = node << 1;
        float s_left = tree[left];

        if (x <= s_left) {
            node = left;
        } else {
            x -= s_left;
            node = left + 1;
        }
    }

    int64_t leaf = node - L;
    if (leaf >= capacity) leaf = capacity - 1; // safety
    out_idx[i] = leaf;
}

torch::Tensor sample_cuda(torch::Tensor tree,
                          torch::Tensor u,
                          int64_t L,
                          int64_t capacity) {
    check_cuda(tree, "tree");
    check_cuda(u, "u");

    TORCH_CHECK(tree.dtype() == torch::kFloat32, "tree must be float32");
    TORCH_CHECK(u.dtype() == torch::kFloat32, "u must be float32");
    TORCH_CHECK(tree.dim() == 1, "tree must be 1D");
    TORCH_CHECK(u.dim() == 1, "u must be 1D");

    auto out = torch::empty({u.numel()},
        torch::TensorOptions().device(u.device()).dtype(torch::kInt64));

    const int64_t n = u.numel();
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    sample_kernel<<<blocks, threads, 0, stream>>>(
        tree.data_ptr<float>(),
        u.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        n, L, capacity
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
