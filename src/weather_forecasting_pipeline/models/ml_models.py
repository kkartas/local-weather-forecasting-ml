"""Traditional machine learning model factories."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def make_ml_model(name: str, random_seed: int):
    """Create a deterministic scikit-learn regressor."""
    if name == "linear_regression":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_seed)
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=0.1)
    raise ValueError(f"Unknown ML model: {name}")
