"""Model-facing dataset utilities built on MetDataPy-prepared data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitMetadata:
    target_column: str
    feature_columns: list[str]
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    n_train: int
    n_validation: int
    n_test: int


def target_column_name(target: str, horizon_steps: int) -> str:
    """Return the MetDataPy supervised target column name."""
    return f"{target}_t+{horizon_steps}"


EXCLUDED_FEATURE_NAMES: frozenset[str] = frozenset({"gap"})


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Select numeric feature columns without future targets or gap markers."""
    feature_columns: list[str] = []
    for col in df.columns:
        col_str = str(col)
        if col == target_col:
            continue
        if "_t+" in col_str:
            continue
        if col_str in EXCLUDED_FEATURE_NAMES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col_str)
    if not feature_columns:
        raise ValueError("No numeric feature columns available after excluding future targets")
    return feature_columns


def arrays_from_split(
    split: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one split to model arrays."""
    if target_col not in split.columns:
        raise ValueError(f"Target column {target_col!r} not present")
    x = split[feature_columns].to_numpy(dtype=np.float32)
    y = split[target_col].to_numpy(dtype=np.float32)
    return x, y


def sequence_arrays_from_split(
    split: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create leakage-free sequence tensors.

    For a row at forecast origin ``t``, the sequence contains rows ending at
    ``t`` and the target is the already shifted MetDataPy target at ``t+h``.
    """
    x_tab, y_tab = arrays_from_split(split, feature_columns, target_col)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if len(split) < sequence_length:
        return (
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for end_pos in range(sequence_length - 1, len(split)):
        start_pos = end_pos - sequence_length + 1
        xs.append(x_tab[start_pos : end_pos + 1])
        ys.append(float(y_tab[end_pos]))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def make_split_metadata(
    splits: dict[str, pd.DataFrame],
    target_col: str,
    feature_columns: list[str],
) -> SplitMetadata:
    """Build serializable split metadata."""
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    return SplitMetadata(
        target_column=target_col,
        feature_columns=feature_columns,
        train_start=str(train.index.min()),
        train_end=str(train.index.max()),
        validation_start=str(val.index.min()),
        validation_end=str(val.index.max()),
        test_start=str(test.index.min()),
        test_end=str(test.index.max()),
        n_train=len(train),
        n_validation=len(val),
        n_test=len(test),
    )


def save_split_metadata(metadata: SplitMetadata, path: str | Path) -> None:
    """Write split metadata as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(asdict(metadata), fh, indent=2)
