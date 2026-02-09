from torch.utils.cpp_extension import load
import torch

per_ext = load(
    name="per_ext",
    sources=["ext/per_ext.cpp", "ext/per_kernels.cu"],
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=True,
)

if __name__ == "__main__":
    print("Built per_ext OK")
    # quick smoke test
    assert torch.cuda.is_available()
    print("CUDA available:", torch.cuda.is_available())
