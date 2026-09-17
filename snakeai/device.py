"""Device selection and a guard for the failure mode that actually bites on new hardware."""

from __future__ import annotations

import torch

# Measured on this project's workload (see docs/BENCHMARKS.md): CUDA only beats CPU once
# the forward pass is batched past ~128 rows. These defaults keep every inference batch
# above that.
AUTO_NUM_ENVS = {"cuda": 256, "cpu": 8}


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device spec of auto, cpu or cuda into a torch.device."""
    if spec == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu" # prefer the GPU if there is one
        return torch.device(name) # hand back whichever one won

    if spec == "cuda" and not torch.cuda.is_available(): # asked for a GPU torch cannot see
        raise RuntimeError(
            "train.device='cuda' but torch.cuda.is_available() is False. This is usually a "
            f"CPU-only wheel: torch {torch.__version__} was built with CUDA "
            f"{torch.version.cuda!r}. Reinstall with a CUDA build, e.g.\n"
            "  pip install --force-reinstall torch"
        )

    return torch.device(spec) # cpu or cuda, exactly as asked


def assert_kernels_available(device: torch.device) -> None:
    """Fail loudly if this torch build has no kernels for this GPU.

    A wheel older than the card still reports cuda.is_available() as True, then either dies at
    the first matmul or silently JITs and runs slow.
    """
    if device.type != "cuda":
        return # nothing to check on the CPU

    major, minor = torch.cuda.get_device_capability(device) # e.g. (12, 0) for Blackwell
    arch = f"sm_{major}{minor}" # the kernel name torch uses for this card
    available = torch.cuda.get_arch_list() # the architectures this wheel was built for

    if arch not in available: # the wheel predates the card
        name = torch.cuda.get_device_name(device) # the marketing name of the card
        raise RuntimeError(
            f"torch {torch.__version__} (CUDA {torch.version.cuda}) has no {arch} kernels for "
            f"{name}. Built for: {available}. Install a newer wheel:\n"
            "  pip install --force-reinstall "
            "--index-url https://download.pytorch.org/whl/cu128 torch"
        )


def resolve_num_envs(requested: int, device: torch.device) -> int:
    """Return the requested env count, or this device's default when it is 0."""
    if requested > 0:
        return requested # an explicit count from the config wins

    return AUTO_NUM_ENVS[device.type] # 0 means "pick for me"


def describe(device: torch.device) -> str:
    """One line summary of the device and torch build for the run header."""
    if device.type != "cuda":
        return f"cpu (torch {torch.__version__})" # nothing else worth printing

    name = torch.cuda.get_device_name(device) # e.g. "NVIDIA GeForce RTX 5090"
    major, minor = torch.cuda.get_device_capability(device) # compute capability of the card
    total_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3 # bytes to GiB

    return (
        f"{name} sm_{major}{minor}, {total_gib:.1f} GiB, "
        f"torch {torch.__version__} / CUDA {torch.version.cuda}"
    )
