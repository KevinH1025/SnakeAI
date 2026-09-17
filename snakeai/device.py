"""Device selection and a guard for the failure mode that actually bites on new hardware."""

from __future__ import annotations

import torch

#: Measured on this project's workload (see docs/BENCHMARKS.md): CUDA only beats CPU once the
#: forward pass is batched past ~128 rows. These defaults keep every inference batch above that.
AUTO_NUM_ENVS = {"cuda": 256, "cpu": 8}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") # prefer the GPU if there is one
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "train.device='cuda' but torch.cuda.is_available() is False. This is usually a "
            f"CPU-only wheel: torch {torch.__version__} was built with CUDA "
            f"{torch.version.cuda!r}. Reinstall with a CUDA build, e.g.\n"
            "  pip install --force-reinstall torch"
        )
    return torch.device(spec)


def assert_kernels_available(device: torch.device) -> None:
    """Fail loudly if this torch build has no kernels for this GPU.

    A wheel that predates the card (cu121/cu124 on Blackwell, say) still reports
    ``cuda.is_available() == True``. You then get either a hard "no kernel image is available"
    at the first matmul or, worse, a silent multi second PTX JIT to the new arch that works
    but is slow and easy to mistake for "the GPU just isn't helping".
    """
    if device.type != "cuda":
        return
    major, minor = torch.cuda.get_device_capability(device) # e.g. (12, 0) for Blackwell
    arch, available = f"sm_{major}{minor}", torch.cuda.get_arch_list() # what this wheel was built for
    if arch not in available: # the wheel predates the card
        raise RuntimeError(
            f"torch {torch.__version__} (CUDA {torch.version.cuda}) has no {arch} kernels for "
            f"{torch.cuda.get_device_name(device)}. Built for: {available}. Install a newer wheel:\n"
            "  pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch"
        )


def resolve_num_envs(requested: int, device: torch.device) -> int:
    return requested if requested > 0 else AUTO_NUM_ENVS[device.type] # 0 means "pick for me"


def describe(device: torch.device) -> str:
    if device.type != "cuda":
        return f"cpu (torch {torch.__version__})"
    major, minor = torch.cuda.get_device_capability(device)
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return (
        f"{torch.cuda.get_device_name(device)} sm_{major}{minor}, {total:.1f} GiB, "
        f"torch {torch.__version__} / CUDA {torch.version.cuda}"
    )
