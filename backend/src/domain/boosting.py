"""Gradient-boosted quantiles with an exact pinball line search per terminal leaf."""

from collections.abc import Sequence
from dataclasses import dataclass

from .settings import FEATURES
from .trees import Tree, Vector, fit_tree

QUANTILES = (50, 90)


def shot_features(row: dict) -> Vector:
    return tuple(row[field] for field in FEATURES)


def empirical_quantile(values: Sequence[float], percentile: int) -> float:
    """Return the smallest value whose empirical CDF reaches the given percentile."""
    ordered = sorted(values)
    rank = (len(ordered) * percentile + 99) // 100
    return ordered[rank - 1]


@dataclass(frozen=True)
class QuantileModel:
    constant: float
    rate: float
    trees: tuple[Tree, ...]

    def predict(self, features: Vector) -> float:
        prediction = self.constant
        for tree in self.trees:
            prediction += self.rate * tree.predict(features)
        return prediction


def fit_quantile(features: Sequence[Vector], actual: Sequence[float], percentile: int,
                 settings: dict) -> QuantileModel:
    constant = empirical_quantile(actual, percentile)
    rate = settings["learning_rate"]
    alpha = percentile / 100
    fitted = [constant] * len(actual)
    trees = []
    for _ in range(settings["estimators"]):
        residuals = [observed - estimate for observed, estimate in zip(actual, fitted)]
        # Zero is a valid subgradient at an exact hit.
        gradients = [alpha if residual > 0 else alpha - 1 if residual < 0 else 0.0 for residual in residuals]

        def correction(members: list[int]) -> float:
            return empirical_quantile([residuals[index] for index in members], percentile)

        tree = fit_tree(features, gradients, correction, settings["max_depth"], settings["min_leaf"])
        fitted = [estimate + rate * tree.predict(vector) for estimate, vector in zip(fitted, features)]
        trees.append(tree)
    return QuantileModel(constant, rate, tuple(trees))


def predict_shot(row: dict, models: tuple[QuantileModel, QuantileModel]) -> dict:
    vector = shot_features(row)
    raw_p50, raw_p90 = (model.predict(vector) for model in models)
    p50 = round(max(0.0, raw_p50), 6)
    p90 = round(max(0.0, raw_p50, raw_p90), 6)
    return {**row, "p50_minutes": p50, "p90_minutes": p90, "crossing_adjusted": raw_p90 < raw_p50}
