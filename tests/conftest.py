import pytest
import torch

from snakeai.config import apply_overrides, preset


@pytest.fixture
def cfg():
    """A tiny config that trains in seconds. CPU only, so the suite runs anywhere."""
    return apply_overrides(preset("small"), [
        "train.device=cpu",
        "train.num_envs=4",
        "agent.learning_starts=128",
        "agent.buffer_capacity=5000",
        "agent.hidden=32,32",
    ])


@pytest.fixture
def device():
    return torch.device("cpu")
