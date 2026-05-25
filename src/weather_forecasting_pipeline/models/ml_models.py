"""Traditional machine learning model factories."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import SVR

# Ridge alpha grid for ``RidgeCV``. Spans three orders of magnitude so that
# the cross-validated choice is meaningful for the wide (~232 feature) lag
# plus rolling design matrix used by the pipeline. RidgeCV uses an efficient
# leave-one-out routine on the training partition only, which preserves the
# project's chronological-split leakage rules.
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


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
        return RidgeCV(alphas=_RIDGE_ALPHAS)
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
