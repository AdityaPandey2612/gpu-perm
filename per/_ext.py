# per/_ext.py
from pathlib import Path
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent

# Unique name avoids collisions if you have multiple builds
_EXT_NAME = "per_ext"

per_ext = load(
    name=_EXT_NAME,
    sources=[
        str(_ROOT / "ext" / "per_ext.cpp"),
        str(_ROOT / "ext" / "per_kernels.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=True,
)
