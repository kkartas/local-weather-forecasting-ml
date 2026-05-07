"""PyTorch deep learning models for sequence forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
) -> DLTrainingResult:
    """Train a PyTorch sequence model with validation early stopping."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    epochs_done = 0

    for epoch in range(max_epochs):
        epochs_done = epoch + 1
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model = model.cpu()
    return DLTrainingResult(model=model, best_validation_loss=best_val, epochs_trained=epochs_done)


def predict_dl_model(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run batched prediction for a trained PyTorch model."""
    model.eval()
    preds: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    """Save model weights."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
