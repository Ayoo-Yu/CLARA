"""Unified predictor interface for 0427.

Each predictor trains once per (zone, seed, horizon) and outputs predictions
for ALL quantiles needed across the 11 alpha levels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from build_0427_configs import ALL_QUANTILES


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePredictor(ABC):
    """Output shape: (N, len(quantiles)) where columns match ALL_QUANTILES."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        timestamps: np.ndarray | None = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


def rearrange_quantile_predictions(
    predictions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """逐行升序重排分位数，并返回重排前后的诊断。"""
    raw = np.asarray(predictions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(ALL_QUANTILES):
        raise ValueError("分位数预测形状与ALL_QUANTILES不一致")
    if not np.isfinite(raw).all():
        raise ValueError("分位数预测包含非有限值")

    before_diff = np.diff(raw, axis=1)
    before_crossing = before_diff < 0.0
    rearranged = np.sort(raw, axis=1, kind="stable")
    after_crossing = np.diff(rearranged, axis=1) < 0.0
    adjustment = np.abs(rearranged - raw)
    row_adjusted = np.any(adjustment > 0.0, axis=1)
    row_crossing = np.any(before_crossing, axis=1)
    negative_depth = np.maximum(-before_diff, 0.0)

    diagnostics = {
        "method": "deterministic_rowwise_ascending_rearrangement",
        "n_rows": int(raw.shape[0]),
        "n_quantiles": int(raw.shape[1]),
        "rows_with_crossing_before": int(row_crossing.sum()),
        "crossing_row_rate_before": float(row_crossing.mean()) if len(raw) else 0.0,
        "crossing_cells_before": int(before_crossing.sum()),
        "max_adjacent_crossing_before": float(negative_depth.max()) if negative_depth.size else 0.0,
        "rows_adjusted": int(row_adjusted.sum()),
        "adjusted_row_rate": float(row_adjusted.mean()) if len(raw) else 0.0,
        "mean_absolute_adjustment": float(adjustment.mean()) if adjustment.size else 0.0,
        "max_absolute_adjustment": float(adjustment.max()) if adjustment.size else 0.0,
        "rows_with_crossing_after": int(np.any(after_crossing, axis=1).sum()),
        "crossing_cells_after": int(after_crossing.sum()),
    }
    return rearranged, diagnostics


# ---------------------------------------------------------------------------
# Ridge baseline
# ---------------------------------------------------------------------------

class RidgePredictor(BasePredictor):
    """Ridge regression for mean + residual quantile approach.

    Fits a Ridge model for the mean, then computes conformity residuals
    and outputs quantile-adjusted predictions for all required quantiles.
    """

    def __init__(self):
        self._model = None
        self._residuals = None
        self._scaler = StandardScaler()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        timestamps: np.ndarray | None = None,
    ) -> None:
        _ = timestamps
        Xs = self._scaler.fit_transform(X_train)
        self._model = Ridge(alpha=1.0, random_state=seed).fit(Xs, y_train)
        preds = self._model.predict(Xs)
        self._residuals = y_train - preds

    def predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = timestamps
        Xs = self._scaler.transform(X)
        mean_pred = self._model.predict(Xs)
        # For each quantile, use empirical quantile of residuals
        n = len(mean_pred)
        out = np.zeros((n, len(ALL_QUANTILES)))
        for i, q in enumerate(ALL_QUANTILES):
            offset = np.quantile(self._residuals, q)
            out[:, i] = mean_pred + offset
        return out


# ---------------------------------------------------------------------------
# GBR tree ensemble
# ---------------------------------------------------------------------------

class GBRTreePredictor(BasePredictor):
    """LightGBM-based quantile regression (replaces sklearn GBR for speed)."""

    def __init__(self):
        self._models: dict[float, lgb.LGBMRegressor] = {}
        self._scaler = StandardScaler()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        timestamps: np.ndarray | None = None,
    ) -> None:
        _ = timestamps
        Xs = self._scaler.fit_transform(X_train)
        for q in ALL_QUANTILES:
            self._models[q] = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            ).fit(Xs, y_train)

    def predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = timestamps
        Xs = self._scaler.transform(X)
        n = len(Xs)
        out = np.zeros((n, len(ALL_QUANTILES)))
        for i, q in enumerate(ALL_QUANTILES):
            out[:, i] = self._models[q].predict(Xs)
        return out


# ---------------------------------------------------------------------------
# MLP (basic deep learning baseline)
# ---------------------------------------------------------------------------

class MLPPredictor(BasePredictor):
    """MLP-based quantile regression via sklearn MLPRegressor.

    Uses the mean + residual quantile approach similar to Ridge,
    but with a neural network for the mean prediction.
    """

    def __init__(self):
        self._model = None
        self._residuals = None
        self._scaler = StandardScaler()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        timestamps: np.ndarray | None = None,
    ) -> None:
        _ = timestamps
        Xs = self._scaler.fit_transform(X_train)
        self._model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ).fit(Xs, y_train)
        preds = self._model.predict(Xs)
        self._residuals = y_train - preds

    def predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = timestamps
        Xs = self._scaler.transform(X)
        mean_pred = self._model.predict(Xs)
        n = len(mean_pred)
        out = np.zeros((n, len(ALL_QUANTILES)))
        for i, q in enumerate(ALL_QUANTILES):
            offset = np.quantile(self._residuals, q)
            out[:, i] = mean_pred + offset
        return out


# ---------------------------------------------------------------------------
# QRLSTM (sequence deep learning)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _quantile_loss_torch(pred: "torch.Tensor", target: "torch.Tensor",
                         q: float) -> "torch.Tensor":
    err = target - pred
    return torch.maximum((q - 1.0) * err, q * err).mean()


@dataclass
class SequenceTrainConfig:
    lookback: int = 24
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 15
    device: str = "cpu"


class _QuantileLSTM(nn.Module):
    def __init__(self, input_dim: int, cfg: SequenceTrainConfig,
                 n_outputs: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, n_outputs),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    timestamps: np.ndarray | None = None,
    pad_short_segments: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按连续时间片构造序列，禁止序列跨越物理时间缺口。"""
    features = np.asarray(features)
    target = np.asarray(target)
    if len(features) != len(target):
        raise ValueError("特征与目标行数不一致")
    if lookback <= 0:
        raise ValueError("lookback必须为正整数")
    if len(features) == 0:
        return (
            np.empty((0, lookback, features.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    segment_starts = np.zeros(len(features), dtype=np.int64)
    if timestamps is not None:
        time_values = pd.to_datetime(np.asarray(timestamps), errors="raise")
        if len(time_values) != len(features):
            raise ValueError("时间戳与特征行数不一致")
        deltas = pd.Series(time_values).diff()
        positive = deltas.iloc[1:]
        if positive.isna().any() or positive.le(pd.Timedelta(0)).any():
            raise ValueError("QRLSTM时间戳必须严格递增且唯一")
        counts = positive.value_counts()
        nominal = min(counts[counts.eq(counts.max())].index)
        current_start = 0
        for idx in range(1, len(features)):
            if deltas.iloc[idx] != nominal:
                current_start = idx
            segment_starts[idx] = current_start

    xs: list[np.ndarray] = []
    ys: list[float] = []
    output_indices: list[int] = []
    for idx in range(len(features)):
        segment_start = int(segment_starts[idx])
        available = idx - segment_start + 1
        if available < lookback and not pad_short_segments:
            continue
        start = max(segment_start, idx - lookback + 1)
        sequence = features[start: idx + 1]
        if len(sequence) < lookback:
            padding = np.repeat(sequence[0:1], lookback - len(sequence), axis=0)
            sequence = np.vstack([padding, sequence])
        xs.append(sequence)
        ys.append(float(target[idx]))
        output_indices.append(idx)

    if not xs:
        return (
            np.empty((0, lookback, features.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(output_indices, dtype=np.int64),
    )


class QRLSTMPredictor(BasePredictor):
    """LSTM quantile regression for all required quantiles."""

    def __init__(self, cfg: SequenceTrainConfig | None = None):
        self._cfg = cfg or SequenceTrainConfig()
        self._model = None
        self._scaler = StandardScaler()
        self._n_outputs = len(ALL_QUANTILES)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        timestamps: np.ndarray | None = None,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for QRLSTMPredictor")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        Xs = self._scaler.fit_transform(X_train).astype(np.float32)
        X_seq, y_seq, _ = _build_sequences(
            Xs,
            y_train,
            self._cfg.lookback,
            timestamps=timestamps,
            pad_short_segments=False,
        )
        if len(X_seq) == 0:
            raise ValueError("连续训练序列不足，无法训练QRLSTM")

        device = torch.device(self._cfg.device)
        input_dim = X_seq.shape[2]
        self._model = _QuantileLSTM(input_dim, self._cfg, self._n_outputs).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_seq), torch.from_numpy(y_seq.astype(np.float32))
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=self._cfg.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)

        for _ in range(self._cfg.epochs):
            self._model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = self._model(xb)
                loss = torch.tensor(0.0, device=device)
                for i, q in enumerate(ALL_QUANTILES):
                    loss = loss + _quantile_loss_torch(pred[:, i], yb, q)
                # Penalty for crossing lower > upper
                for i in range(len(ALL_QUANTILES) - 1):
                    loss = loss + 0.2 * torch.relu(pred[:, i] - pred[:, i + 1]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = self._model.cpu()

    def predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        Xs = self._scaler.transform(X).astype(np.float32)
        X_seq, _, output_indices = _build_sequences(
            Xs,
            np.zeros(len(Xs)),
            self._cfg.lookback,
            timestamps=timestamps,
            pad_short_segments=True,
        )
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X_seq)).numpy()
        if not np.array_equal(output_indices, np.arange(len(Xs), dtype=np.int64)):
            raise AssertionError("QRLSTM预测序列没有覆盖全部输入行")
        return pred


# ---------------------------------------------------------------------------
# Predictor registry
# ---------------------------------------------------------------------------

PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {
    "Ridge": RidgePredictor,
    "GBR": GBRTreePredictor,
    "MLP": MLPPredictor,
    "QRLSTM": QRLSTMPredictor,
}


def get_predictor(name: str) -> BasePredictor:
    cls = PREDICTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown predictor: {name}")
    return cls()


def extract_quantile_predictions(
    all_preds: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From the full quantile output, extract lower, center, upper for a given alpha."""
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    idx_low = ALL_QUANTILES.index(q_low)
    idx_center = ALL_QUANTILES.index(0.5)
    idx_high = ALL_QUANTILES.index(q_high)
    return (
        all_preds[:, idx_low],
        all_preds[:, idx_center],
        all_preds[:, idx_high],
    )
