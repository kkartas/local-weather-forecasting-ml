"""Reproducibility helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set deterministic seeds for supported libraries.

    Sets ``PYTHONHASHSEED`` and seeds Python's ``random`` and NumPy globals.
    For PyTorch, in addition to the manual seed and the cuDNN deterministic
    flags, this enables ``torch.use_deterministic_algorithms(warn_only=True)``
    so any kernel without a deterministic implementation surfaces as a warning
    instead of silently introducing run-to-run variance. ``CUBLAS_WORKSPACE_CONFIG``
    is set on CUDA to satisfy cuBLAS's determinism contract.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS requires a fixed workspace config to behave deterministically; set
    # it before torch is imported so the runtime picks it up on first use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:
            LOGGER.debug("torch.use_deterministic_algorithms unavailable: %s", exc)
    except Exception as exc:
        LOGGER.debug("Torch determinism setup skipped: %s", exc)
