"""Training progress state and heartbeat helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Callable, Iterator


@dataclass
class TrainingProgressTracker:
    total_models: int
    _next_run: int = 1
    _completed: int = 0

    def start_model(self) -> dict[str, int]:
        run = self._next_run
        self._next_run += 1
        return {
            "run": run,
            "total": self.total_models,
            "remaining": max(self.total_models - run, 0),
        }

    def finish_model(self) -> dict[str, int]:
        self._completed += 1
        return {
            "run_completed": self._completed,
            "total": self.total_models,
            "remaining": max(self.total_models - self._completed, 0),
        }


class SharedTrainingProgressTracker:
    def __init__(self, total_models: int, manager: Any) -> None:
        self.total_models = int(total_models)
        self._next_run = manager.Value("i", 1)
        self._completed = manager.Value("i", 0)
        self._lock = manager.Lock()

    def start_model(self) -> dict[str, int]:
        with self._lock:
            run = int(self._next_run.value)
            self._next_run.value = run + 1
            return {
                "run": run,
                "total": self.total_models,
                "remaining": max(self.total_models - run, 0),
            }

    def finish_model(self) -> dict[str, int]:
        with self._lock:
            self._completed.value = int(self._completed.value) + 1
            done = int(self._completed.value)
            return {
                "run_completed": done,
                "total": self.total_models,
                "remaining": max(self.total_models - done, 0),
            }


@contextmanager
def heartbeat_during(
    interval_seconds: int, tick: Callable[[int], None]
) -> Iterator[None]:
    if interval_seconds <= 0:
        yield
        return

    stop = Event()
    started = time.perf_counter()

    def _runner() -> None:
        while not stop.wait(interval_seconds):
            tick(int(time.perf_counter() - started))

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
