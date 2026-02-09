#include <torch/extension.h>

// CUDA forward decl
void update_cuda(torch::Tensor tree,
                 torch::Tensor idx,
                 torch::Tensor new_p,
                 int64_t L);

torch::Tensor sample_cuda(torch::Tensor tree,
                          torch::Tensor u,
                          int64_t L,
                          int64_t capacity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", &update_cuda, "Sum-tree update (CUDA)");
    m.def("sample", &sample_cuda, "Sum-tree sample (CUDA)");
}
