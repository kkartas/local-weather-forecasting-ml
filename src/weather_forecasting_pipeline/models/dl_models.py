"""PyTorch deep learning models for sequence forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class RecurrentRegressor(nn.Module):
    """LSTM/GRU regressor.

    Input shape is ``(batch, sequence_length, n_features)``. The final recurrent
    hidden state is projected to one scalar forecast target. A dropout layer
    between the recurrent output and the regression head provides a small
    amount of regularisation, which matters because the dissertation's modest
    training-row budget makes the unregularised default prone to overfit.
    """

    def __init__(
        self,
        input_size: int,
        cell: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if cell == "lstm":
            self.recurrent = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif cell == "gru":
            self.recurrent = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported recurrent cell: {cell}")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class TemporalBlock(nn.Module):
    """Simple causal temporal convolution block with dropout."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=2, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=2, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y[..., : x.size(-1)] + x


def _dilations_for_receptive_field(sequence_length: int, kernel_size: int = 2) -> list[int]:
    """Pick exponentially-growing dilations whose receptive field covers the input.

    Each ``TemporalBlock`` applies two stacked dilated convolutions of kernel
    ``k`` and dilation ``d``, contributing ``2 * d * (k-1)`` to the receptive
    field. Starting from RF=1 and doubling dilations (1, 2, 4, ...) we add
    enough blocks to reach ``sequence_length``. Capping at ``d=64`` keeps the
    network compact for the configured 10-min cadence; the previous fixed
    ``(1, 2, 4)`` schedule only covered 15 timesteps and silently ignored the
    bulk of a 144-step input.
    """
    rf = 1
    dilations: list[int] = []
    d = 1
    while rf < max(int(sequence_length), 1):
        dilations.append(d)
        rf += 2 * d * (kernel_size - 1)
        d = min(d * 2, 64)
    if not dilations:
        dilations = [1]
    return dilations


class TCNRegressor(nn.Module):
    """Compact temporal convolutional network for sequence regression."""

    def __init__(
        self,
        input_size: int,
        channels: int = 64,
        sequence_length: int = 144,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        dilations = _dilations_for_receptive_field(sequence_length)
        self.blocks = nn.Sequential(
            *[TemporalBlock(channels, dilation, dropout=dropout) for dilation in dilations]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (batch, features, sequence_length).
        z = x.transpose(1, 2)
        z = self.input_projection(z)
        z = self.blocks(z)
        return self.head(z[:, :, -1]).squeeze(-1)


@dataclass(frozen=True)
class DLTrainingResult:
    model: nn.Module
    best_validation_loss: float
    epochs_trained: int


def make_dl_model(name: str, input_size: int, sequence_length: int = 144) -> nn.Module:
    """Create a deep learning sequence model.

    ``sequence_length`` is forwarded to the TCN so its dilation schedule grows
    a receptive field that actually covers the configured input window.
    """
    if name == "lstm":
        return RecurrentRegressor(input_size=input_size, cell="lstm")
    if name == "gru":
        return RecurrentRegressor(input_size=input_size, cell="gru")
    if name == "tcn":
        return TCNRegressor(input_size=input_size, sequence_length=sequence_length)
    raise ValueError(f"Unknown DL model: {name}")


# Hyperparameters of the LR scheduler attached to the Adam optimiser inside
# :func:`train_dl_model_from_datasets`. They are part of the DL training
# stability bundle and are kept module-level so tests and downstream callers
# can introspect them without re-deriving the values.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-5


def train_dl_model_from_datasets(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    grad_clip_norm: float | None = 1.0,
    on_epoch_end: Callable[[int, int, float, float, int], None] | None = None,
) -> DLTrainingResult:
    """Train a PyTorch sequence model using lazy ``Dataset`` inputs.

    The training and validation datasets must yield ``(x, y)`` pairs with
    ``x`` shaped ``(sequence_length, n_features)`` and ``y`` a scalar tensor.
    Each batch is built on demand by the underlying :class:`DataLoader` so the
    full training tensor is never materialized in memory. Validation loss is
    computed batch-wise as a sample-weighted MSE so the early-stopping anchor
    matches a single-pass forward over the validation set.

    Training stability bundle:

    - A :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` is attached to
      the Adam optimiser and stepped after every validation pass. It halves
      the learning rate when the validation loss has not improved for
      :data:`LR_SCHEDULER_PATIENCE` epochs (down to
      :data:`LR_SCHEDULER_MIN_LR`). This rescues training runs that would
      otherwise hit early-stopping during a transient loss plateau.
    - ``grad_clip_norm`` (default ``1.0``) clips the parameter gradients
      to the given L2 norm before each optimiser step. Pass ``None`` or a
      non-positive value to disable clipping. This protects against the
      exploding-gradient events that can produce abrupt training collapses
      on the longer forecast horizons.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )
    loss_fn = nn.MSELoss()

    # A non-positive or ``None`` clip threshold disables gradient clipping
    # without forcing callers to thread a separate flag through the API.
    clip_enabled = grad_clip_norm is not None and grad_clip_norm > 0.0

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    epochs_done = 0

    for epoch in range(max_epochs):
        epochs_done = epoch + 1
        model.train()
        train_weighted_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if clip_enabled:
                # ``clip_grad_norm_`` rescales gradients in-place when the
                # global L2 norm exceeds ``max_norm``. It is a no-op when
                # gradients are well-behaved, so the cost is negligible.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            batch_count = int(yb.numel())
            train_weighted_loss += float(loss.item()) * batch_count
            train_count += batch_count

        train_loss = train_weighted_loss / max(train_count, 1)

        model.eval()
        sse = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                # Aggregate sum-of-squared-errors so the final mean matches the
                # in-memory MSELoss the previous implementation reported.
                sse += float(((preds - yb) ** 2).sum().item())
                count += int(yb.numel())
        val_loss = sse / max(count, 1)
        # Drive the LR scheduler on the validation-loss anchor. The
        # scheduler decrements the learning rate after
        # ``LR_SCHEDULER_PATIENCE`` non-improving epochs; outer early
        # stopping still applies on top via the ``patience`` counter.
        scheduler.step(val_loss)
        should_stop = False
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                should_stop = True

        patience_left = max(patience - bad_epochs, 0)
        if on_epoch_end is not None:
            on_epoch_end(epochs_done, max_epochs, train_loss, val_loss, patience_left)
        if should_stop:
            break

    model.load_state_dict(best_state)
    model = model.cpu()
    return DLTrainingResult(model=model, best_validation_loss=best_val, epochs_trained=epochs_done)


def train_dl_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    grad_clip_norm: float | None = 1.0,
) -> DLTrainingResult:
    """Train a PyTorch sequence model with validation early stopping.

    Backwards-compatible wrapper that accepts pre-built ``(n, seq, feat)``
    arrays. The training pipeline uses :func:`train_dl_model_from_datasets`
    with on-demand sequence windows to stay memory-safe on the full
    multi-year configuration; this helper remains for unit tests and small
    fixtures that already hold the full tensor in memory. ``grad_clip_norm``
    forwards to the underlying training loop (see its docstring for the
    full stability bundle).
    """
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    return train_dl_model_from_datasets(
        model,
        train_ds,
        val_ds,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed,
        grad_clip_norm=grad_clip_norm,
    )


def predict_dl_model(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run batched prediction for a trained PyTorch model.

    Backwards-compatible helper used by tests with pre-built tensors. The
    pipeline uses :func:`predict_dl_model_from_dataset` so the prediction
    path stays memory-safe for the full configuration.
    """
    model.eval()
    preds: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def predict_dl_model_from_dataset(
    model: nn.Module, dataset: Dataset, batch_size: int = 256
) -> np.ndarray:
    """Run batched prediction across a lazy sequence dataset.

    Iterates the dataset's ``DataLoader`` and ignores the ``y`` element of
    each batch so the same ``(x, y)`` dataset can be used for both training
    and inference without an extra ``x``-only wrapper.
    """
    model.eval()
    preds: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            xb: Any = batch[0] if isinstance(batch, (list, tuple)) else batch
            preds.append(model(xb).detach().cpu().numpy())
    if not preds:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    """Save model weights."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
