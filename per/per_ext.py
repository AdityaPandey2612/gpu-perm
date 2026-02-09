#include <torch/extension.h>

void update_cuda(torch::Tensor tree, torch::Tensor idx, torch::Tensor new_p, int64_t L);
void update_opt_cuda(torch::Tensor tree, torch::Tensor idx, torch::Tensor new_p, int64_t L);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &update_cuda, "Sum-tree update (CUDA)");
m.def("update_opt", &update_opt_cuda, "Sum-tree update optimized with CUB (CUDA)");
// keep sample binding too if you have it
}
