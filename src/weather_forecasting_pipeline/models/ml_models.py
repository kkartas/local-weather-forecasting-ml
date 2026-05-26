"""Traditional machine learning model factories."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

# Ridge alpha grid. Spans three orders of magnitude so that the
# cross-validated choice is meaningful for the wide lag plus rolling design
# matrix used by the pipeline.
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
_RIDGE_CV_SPLITS = 5


class ChronologicalRidgeCV(BaseEstimator, RegressorMixin):
    """Ridge regression with leakage-safe chronological alpha selection.

    Scikit-learn's ``RidgeCV`` defaults to generalized leave-one-out CV, which
    uses dense linear algebra paths that allocate large float64 work arrays on
    the dissertation-scale feature matrix. This estimator keeps the same model
    family and alpha grid, but evaluates alpha values with expanding-window
    ``TimeSeriesSplit`` folds and fits ``Ridge(solver="lsqr")`` to avoid the
    memory-heavy SVD/Cholesky routines. The pipeline has already standardised
    the features with a training-only scaler, so this wrapper centers the
    target per fold and fits Ridge with ``fit_intercept=False``. That keeps an
    intercept-equivalent offset without letting sklearn copy and center the
    full wide feature matrix for every fit.
    """

    def __init__(
        self,
        alphas: tuple[float, ...] = _RIDGE_ALPHAS,
        n_splits: int = _RIDGE_CV_SPLITS,
        solver: str = "lsqr",
    ):
        self.alphas = alphas
        self.n_splits = n_splits
        self.solver = solver

    def fit(self, X, y):
        x = np.asarray(X, dtype=np.float32, order="C")
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"ChronologicalRidgeCV expects a 2-D feature matrix, got {x.shape}")
        if len(x) != len(y_arr):
            raise ValueError(f"Shape mismatch: X has {len(x)} rows but y has {len(y_arr)} rows")
        if not self.alphas:
            raise ValueError("ChronologicalRidgeCV requires at least one alpha value")

        cv_scores: dict[float, float] = {}
        n_splits = min(max(2, int(self.n_splits)), max(2, len(x) - 1))
        if len(x) > n_splits:
            splitter = TimeSeriesSplit(n_splits=n_splits)
            for alpha in self.alphas:
                fold_scores: list[float] = []
                for train_idx, val_idx in splitter.split(x):
                    train_sel = _contiguous_selector(train_idx)
                    val_sel = _contiguous_selector(val_idx)
                    model, y_offset = self._fit_model(alpha, x[train_sel], y_arr[train_sel])
                    pred = model.predict(x[val_sel]) + y_offset
                    fold_scores.append(_rmse(y_arr[val_sel], pred))
                cv_scores[float(alpha)] = float(np.mean(fold_scores))
            self.alpha_ = min(cv_scores, key=cv_scores.get)
        else:
            # Degenerate tiny fixtures cannot support chronological CV. Use the
            # strongest regularisation rather than failing the smoke path.
            self.alpha_ = float(self.alphas[-1])
            cv_scores[self.alpha_] = np.nan

        self.cv_scores_ = cv_scores
        self.estimator_, self.y_offset_ = self._fit_model(self.alpha_, x, y_arr)
        self.n_features_in_ = int(x.shape[1])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        x = np.asarray(X, dtype=np.float32, order="C")
        return self.estimator_.predict(x) + self.y_offset_

    def _make_model(self, alpha: float) -> Ridge:
        return Ridge(alpha=float(alpha), solver=self.solver, copy_X=False, fit_intercept=False)

    def _fit_model(self, alpha: float, x: np.ndarray, y: np.ndarray) -> tuple[Ridge, float]:
        y_offset = float(np.mean(y))
        model = self._make_model(alpha)
        model.fit(x, y - y_offset)
        return model, y_offset


def _contiguous_selector(index: np.ndarray) -> slice | np.ndarray:
    if len(index) == 0:
        return index
    start = int(index[0])
    stop = int(index[-1]) + 1
    if stop - start == len(index):
        return slice(start, stop)
    return index


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float32).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def make_ml_model(name: str, random_seed: int, *, rf_n_jobs: int = -1):
    """Create a deterministic scikit-learn regressor.

    ``rf_n_jobs`` controls the inner parallelism of
    :class:`RandomForestRegressor`. The default ``-1`` keeps the historical
    behaviour (use all available cores) for sequential training. When the
    pipeline trains horizons in parallel it passes ``rf_n_jobs=1`` so that
    outer × inner CPU oversubscription does not collapse throughput.

    Notes on roster changes:

    - ``ridge`` is the default linear baseline as of 2026-05-25 because
      OLS produced an RMSE explosion on the wide lag+rolling feature
      matrix in run 180526 (CHANGES.md).
    - ``linear_regression`` (plain OLS) is retained for backwards
      compatibility with run 180526 reproduction; it is no longer part of
      the shipped default configuration.
    - ``svr`` is retained for the same reason but is similarly excluded
      from the shipped default configuration (CHANGES.md, run 180526).
    """
    if name == "ridge":
        return ChronologicalRidgeCV(alphas=_RIDGE_ALPHAS, n_splits=_RIDGE_CV_SPLITS)
    if name == "linear_regression":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=rf_n_jobs,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_seed)
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=0.1)
    raise ValueError(f"Unknown ML model: {name}")
