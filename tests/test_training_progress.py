"""Tests for training progress tracker primitives."""

from __future__ import annotations

import logging
import time
from multiprocessing import Manager

from weather_forecasting_pipeline.training.progress import (
    SharedTrainingProgressTracker,
    TrainingProgressTracker,
    heartbeat_during,
)


def test_tracker_assigns_monotonic_slots():
    tracker = TrainingProgressTracker(total_models=5)
    first = tracker.start_model()
    second = tracker.start_model()
    assert first["run"] == 1
    assert second["run"] == 2
    assert first["remaining"] == 4
    assert second["remaining"] == 3


def test_tracker_finish_model_increments_completed():
    tracker = TrainingProgressTracker(total_models=3)
    tracker.start_model()
    done = tracker.finish_model()
    assert done["run_completed"] == 1
    assert done["remaining"] == 2


def test_shared_tracker_assigns_monotonic_slots():
    with Manager() as manager:
        tracker = SharedTrainingProgressTracker(total_models=5, manager=manager)
        first = tracker.start_model()
        second = tracker.start_model()
        assert first["run"] == 1
        assert second["run"] == 2
        assert first["remaining"] == 4
        assert second["remaining"] == 3


def test_shared_tracker_finish_model_increments_completed():
    with Manager() as manager:
        tracker = SharedTrainingProgressTracker(total_models=3, manager=manager)
        tracker.start_model()
        done = tracker.finish_model()
        assert done["run_completed"] == 1
        assert done["remaining"] == 2


def test_heartbeat_during_is_valid_context_manager():
    ticks: list[int] = []

    with heartbeat_during(60, ticks.append):
        pass

    assert isinstance(ticks, list)


def test_heartbeat_during_emits_ticks(caplog):
    ticks = []

    def _tick(elapsed: int) -> None:
        ticks.append(elapsed)
        logging.getLogger("weather_forecasting_pipeline.training.pipeline").info(
            "Stage progress: train model family=ml model=svr horizon=h01 heartbeat elapsed=%ss status=fitting",
            elapsed,
        )

    with caplog.at_level(logging.INFO):
        with heartbeat_during(1, _tick):
            time.sleep(2.2)

    assert ticks
    assert any(
        "Stage progress: train model" in r.getMessage() and "heartbeat" in r.getMessage()
        for r in caplog.records
    )


def test_heartbeat_during_noop_for_zero_interval():
    ticks: list[int] = []

    with heartbeat_during(0, ticks.append):
        time.sleep(1.2)

    assert ticks == []
