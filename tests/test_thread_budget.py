"""Tests for the per-worker BLAS/MKL/PyTorch thread budget.

The pipeline caps each worker's thread pool so that
``horizon_workers * threads_per_worker`` stays within the available CPU
budget when running with high outer parallelism. These tests pin the
contract:

- ``_resolve_torch_threads_per_worker`` returns ``None`` for the sequential
  case and a positive integer otherwise.
- An explicit YAML override beats the auto-formula.
- ``_apply_thread_cap`` sets the documented env vars and calls
  ``torch.set_num_threads`` with the configured value.
"""

from __future__ import annotations

import os
from dataclasses import replace

from weather_forecasting_pipeline.training import pipeline as training_pipeline


def _make_training_config(
    *, horizon_workers: int, torch_threads_per_worker: int | None
) -> object:
    """Build a minimal stub config exposing the two fields under test."""

    class _StubTraining:
        def __init__(self, horizon_workers: int, torch_threads_per_worker: int | None):
            self.horizon_workers = horizon_workers
            self.torch_threads_per_worker = torch_threads_per_worker

    class _StubConfig:
        def __init__(self, horizon_workers: int, torch_threads_per_worker: int | None):
            self.training = _StubTraining(horizon_workers, torch_threads_per_worker)

    return _StubConfig(horizon_workers, torch_threads_per_worker)


def test_resolve_torch_threads_returns_none_for_sequential():
    cfg = _make_training_config(horizon_workers=1, torch_threads_per_worker=None)
    assert training_pipeline._resolve_torch_threads_per_worker(cfg, horizon_workers=1) is None


def test_resolve_torch_threads_auto_divides_cpu_count(monkeypatch):
    """When no override is set, the cap is ``cpu_count // horizon_workers``."""
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    cfg = _make_training_config(horizon_workers=6, torch_threads_per_worker=None)
    assert training_pipeline._resolve_torch_threads_per_worker(cfg, horizon_workers=6) == 2


def test_resolve_torch_threads_floor_to_one(monkeypatch):
    """``horizon_workers > cpu_count`` still produces at least one thread per worker."""
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    cfg = _make_training_config(horizon_workers=8, torch_threads_per_worker=None)
    assert training_pipeline._resolve_torch_threads_per_worker(cfg, horizon_workers=8) == 1


def test_resolve_torch_threads_explicit_override_wins(monkeypatch):
    """Explicit YAML value supersedes the auto-formula."""
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    cfg = _make_training_config(horizon_workers=6, torch_threads_per_worker=3)
    assert training_pipeline._resolve_torch_threads_per_worker(cfg, horizon_workers=6) == 3


def test_apply_thread_cap_sets_env_vars_and_torch(monkeypatch):
    """The cap touches OMP/MKL/OPENBLAS env vars and torch.set_num_threads."""
    import torch

    saved_env = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    captured: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: captured.append(int(n)))

    try:
        training_pipeline._apply_thread_cap(2)
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["MKL_NUM_THREADS"] == "2"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
        assert captured == [2]
    finally:
        # Restore process state so the cap does not leak to other tests.
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_apply_thread_cap_none_is_noop(monkeypatch):
    """Passing ``None`` must not touch env vars or torch threads."""
    import torch

    captured: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: captured.append(int(n)))
    before = dict(os.environ)
    training_pipeline._apply_thread_cap(None)
    assert captured == []
    # ``_apply_thread_cap`` returns early before touching the env, so the
    # full environment should remain byte-identical to the pre-call state.
    assert dict(os.environ) == before
