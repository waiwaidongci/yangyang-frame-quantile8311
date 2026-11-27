"""Gradient-boosted quantiles with an exact pinball line search per terminal leaf."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from .settings import FEATURES
from .trees import Branch, Tree, Vector, fit_tree, tree_from_dict, tree_to_dict

QUANTILES = (50, 90)


def shot_features(row: dict) -> Vector:
    return tuple(row[field] for field in FEATURES)


def feature_boxes(row: dict) -> tuple[tuple[float, float], ...]:
    """Per-feature (lower, upper) bounds; exactly measured features collapse to a point."""
    return (
        (row["frames"], row["frames"]),
        (row["occlusion_pct"], row.get("occlusion_pct_high", row["occlusion_pct"])),
        (row["overlapping_characters"], row["overlapping_characters"]),
        (row["rig_marks"], row.get("rig_marks_high", row["rig_marks"])),
    )


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


def split_thresholds(tree: Tree, feature: int) -> set[float]:
    if isinstance(tree, Branch):
        cuts = split_thresholds(tree.left, feature) | split_thresholds(tree.right, feature)
        return cuts | {tree.threshold} if tree.feature == feature else cuts
    return set()


def model_max(model: QuantileModel, boxes: tuple[tuple[float, float], ...]) -> float:
    """Exact maximum of the piecewise-constant ensemble over an axis-aligned box."""
    axes = []
    for feature, (lower, upper) in enumerate(boxes):
        if lower == upper:
            axes.append([lower])
            continue
        cuts = set()
        for tree in model.trees:
            cuts |= split_thresholds(tree, feature)
        bounds = [lower, *sorted(cut for cut in cuts if lower < cut < upper), upper]
        points = {lower, upper}
        points.update((left + right) / 2 for left, right in zip(bounds, bounds[1:]))
        axes.append(sorted(points))
    return max(model.predict(vector) for vector in product(*axes))


def model_to_dict(model: QuantileModel) -> dict:
    return {"constant": model.constant, "rate": model.rate,
            "trees": [tree_to_dict(tree) for tree in model.trees]}


def model_from_dict(data: dict) -> QuantileModel:
    return QuantileModel(data["constant"], data["rate"],
                         tuple(tree_from_dict(tree) for tree in data["trees"]))


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
    boxes = feature_boxes(row)
    raw_p50 = model_max(models[0], boxes)
    raw_p90 = model_max(models[1], boxes)
    p50 = round(max(0.0, raw_p50), 6)
    p90 = round(max(0.0, raw_p50, raw_p90), 6)
    return {**row, "p50_minutes": p50, "p90_minutes": p90, "crossing_adjusted": raw_p90 < raw_p50}
