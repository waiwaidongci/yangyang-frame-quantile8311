"""Fit shallow CART regions to pinball pseudo-residuals."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Vector = tuple[int | float, ...]


@dataclass(frozen=True)
class Leaf:
    correction: float

    def predict(self, features: Vector) -> float:
        return self.correction


@dataclass(frozen=True)
class Branch:
    feature: int
    threshold: float
    left: "Tree"
    right: "Tree"

    def predict(self, features: Vector) -> float:
        node = self.left if features[self.feature] <= self.threshold else self.right
        return node.predict(features)


Tree = Leaf | Branch


def tree_to_dict(tree: Tree) -> dict:
    if isinstance(tree, Leaf):
        return {"leaf": tree.correction}
    return {"feature": tree.feature, "threshold": tree.threshold,
            "left": tree_to_dict(tree.left), "right": tree_to_dict(tree.right)}


def tree_from_dict(data: dict) -> Tree:
    if "leaf" in data:
        return Leaf(data["leaf"])
    return Branch(data["feature"], data["threshold"],
                  tree_from_dict(data["left"]), tree_from_dict(data["right"]))


def best_partition(features: Sequence[Vector], gradients: Sequence[float],
                   members: list[int], min_leaf: int) -> tuple[int, float, list[int], list[int]] | None:
    """Find the maximal reduction in gradient squared error via prefix sums."""
    size = len(members)
    if size < 2 * min_leaf:
        return None
    total = sum(gradients[index] for index in members)
    best_improvement = 1e-12
    best = None
    for feature in range(len(features[0])):
        ordered = sorted(members, key=lambda index: features[index][feature])
        prefix = 0.0
        for rank, member in enumerate(ordered[:-min_leaf], start=1):
            prefix += gradients[member]
            if rank < min_leaf:
                continue
            lower, upper = features[member][feature], features[ordered[rank]][feature]
            if lower == upper:
                continue
            improvement = prefix**2 / rank + (total - prefix)**2 / (size - rank) - total**2 / size
            if improvement > best_improvement + 1e-12:
                best_improvement = improvement
                midpoint = (lower + upper) / 2
                threshold = midpoint if midpoint < upper else lower
                best = (feature, threshold, ordered, rank)
    if best is None:
        return None
    feature, threshold, ordered, rank = best
    return feature, threshold, ordered[:rank], ordered[rank:]


def fit_tree(features: Sequence[Vector], gradients: Sequence[float],
             leaf_value: Callable[[list[int]], float], max_depth: int, min_leaf: int) -> Tree:
    def grow(members: list[int], depth: int) -> Tree:
        split = best_partition(features, gradients, members, min_leaf) if depth else None
        if split is None:
            return Leaf(leaf_value(members))
        feature, threshold, left, right = split
        return Branch(feature, threshold, grow(left, depth - 1), grow(right, depth - 1))

    return grow(list(range(len(features))), max_depth)
