"""Model-facing dataset utilities built on MetDataPy-prepared data."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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

# MetDataPy lag columns follow the ``<name>_lag<digits>`` convention. They are
# redundant for sequence models because the sequence axis already encodes the
# same recent history; including them in DL inputs explodes RAM (e.g. 144 ×
# ~1.6k features per timestep) without adding information.
_LAG_COLUMN_PATTERN: re.Pattern[str] = re.compile(r"_lag\d+$")


def _is_lag_column(name: str) -> bool:
    return bool(_LAG_COLUMN_PATTERN.search(str(name)))


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


def select_dl_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    *,
    exclude_lag_features: bool = True,
    feature_allow_list: Sequence[str] | None = None,
) -> list[str]:
    """Select per-timestep features for DL sequence inputs.

    DL models receive a ``(sequence_length, n_features)`` window, so each
    timestep should hold present-time signals rather than the wide tabular
    feature matrix used by ML/baselines. By default this drops MetDataPy
    ``_lag<n>`` columns (redundant with the sequence axis) but keeps canonical
    variables, derived metrics, calendar cyclic features, wind-direction
    encoding, rolling stats at ``t``, and causal QC flags.

    When ``feature_allow_list`` is provided the DL features are restricted to
    that explicit list, supporting transparent dissertation runs that fix the
    DL feature set independent of MetDataPy's evolving column inventory. The
    helper validates that every requested column exists in ``df``.
    """
    if feature_allow_list is not None:
        missing = [c for c in feature_allow_list if c not in df.columns]
        if missing:
            raise ValueError(
                f"DL feature allow-list references missing columns: {missing}"
            )
        return [str(c) for c in feature_allow_list]
    base_features = select_feature_columns(df, target_col)
    if not exclude_lag_features:
        return base_features
    return [c for c in base_features if not _is_lag_column(c)]


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

    .. note::
        Materializes ``(n_sequences, sequence_length, n_features)`` in RAM and
        is therefore only suitable for small fixtures or backwards-compatible
        smoke tests. The training pipeline uses :func:`build_sequence_dataset`
        and :class:`SequenceDataset` instead so DL training does not allocate
        the full train tensor in one block.
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


def sequence_targets(target_values: np.ndarray, sequence_length: int) -> np.ndarray:
    """Return the y-array aligned with sequence end positions.

    Each sequence at index ``i`` ends at row ``i + sequence_length - 1``, so the
    aligned target array is ``target_values[sequence_length - 1 :]``. Returns an
    empty 1-D array when the split has fewer rows than ``sequence_length``.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    arr = np.asarray(target_values, dtype=np.float32)
    if len(arr) < sequence_length:
        return np.empty((0,), dtype=np.float32)
    return arr[sequence_length - 1 :].astype(np.float32, copy=False)


class SequenceDataset:
    """Lazy ``(sequence_length, n_features)`` window dataset.

    Holds the scaled per-timestep feature matrix for a single split and the
    aligned target vector. ``__getitem__`` builds one sequence on demand by
    slicing the feature matrix, so DL training does not need to materialize a
    full ``(n_sequences, sequence_length, n_features)`` tensor.

    Implements the small subset of the PyTorch ``Dataset`` protocol used by
    the project (``__len__`` and ``__getitem__``) without importing torch at
    module import time, so the splits module stays import-cheap and usable
    from non-DL tests.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ):
        if features.ndim != 2:
            raise ValueError(
                f"SequenceDataset expects a 2-D feature matrix, got shape {features.shape}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"SequenceDataset expects a 1-D target vector, got shape {targets.shape}"
            )
        if len(features) != len(targets):
            raise ValueError(
                "Feature matrix and target vector must have the same number of rows"
            )
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        # Ensure compact float32 storage so per-window slices are zero-copy
        # views and the training loop never falls back to float64 paths.
        self._features = np.ascontiguousarray(features, dtype=np.float32)
        self._targets = np.ascontiguousarray(targets, dtype=np.float32)
        self._sequence_length = int(sequence_length)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def n_features(self) -> int:
        return int(self._features.shape[1])

    def __len__(self) -> int:
        return max(0, len(self._features) - self._sequence_length + 1)

    def __getitem__(self, idx: int):
        # Lazily import torch so the splits module remains importable in
        # environments that exercise tabular ML/baselines without PyTorch.
        import torch

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        end = idx + self._sequence_length
        x = self._features[idx:end]
        y = self._targets[end - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def build_sequence_dataset(
    split: pd.DataFrame,
    feature_columns: Sequence[str],
    target_values: np.ndarray | pd.Series,
    sequence_length: int,
) -> SequenceDataset:
    """Build a :class:`SequenceDataset` from a scaled split frame.

    ``target_values`` must already be in the desired training space (e.g.
    target-scaled for DL training, original units when the dataset is used
    for prediction targets). The split frame supplies the per-timestep
    features only; this function does not refit any scalers.
    """
    if any(col not in split.columns for col in feature_columns):
        missing = [c for c in feature_columns if c not in split.columns]
        raise ValueError(f"Sequence features missing from split frame: {missing}")
    features = split[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = np.asarray(target_values, dtype=np.float32).reshape(-1)
    return SequenceDataset(features, targets, sequence_length)


def estimate_sequence_batch_bytes(
    sequence_length: int, n_features: int, batch_size: int
) -> int:
    """Return the float32 byte size of a single training batch tensor.

    Used by the training pipeline to log a conservative RAM estimate before
    DL fit; mirrors PyTorch's batch shape ``(batch, seq, features)`` and
    excludes the per-split feature matrix that is materialized once and
    shared across batches.
    """
    return int(sequence_length) * int(n_features) * int(batch_size) * 4


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
