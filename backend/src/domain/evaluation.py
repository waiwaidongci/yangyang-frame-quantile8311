"""Compare model and frozen constant quantiles on independent scenes."""

from collections import defaultdict
from statistics import fmean

from .boosting import QUANTILES


def pinball(actual: float, predicted: float, percentile: int) -> float:
    residual = actual - predicted
    alpha = percentile / 100
    return max(alpha * residual, (alpha - 1) * residual)


def holdout_comparison(predictions: list[dict], baselines: dict[int, float]) -> list[dict]:
    scenes = defaultdict(list)
    for shot in predictions:
        scenes[shot["scene_id"]].append(shot)
    groups = [(None, predictions), *sorted(scenes.items())]
    comparison = []
    for scene, shots in groups:
        actual = [shot["actual_minutes"] for shot in shots]
        line = {"scope": "all" if scene is None else "scene", "scene_id": scene, "shot_count": len(shots)}
        for percentile in QUANTILES:
            name = f"p{percentile}"
            estimate = [shot[f"{name}_minutes"] for shot in shots]
            base = baselines[percentile]
            score = fmean(pinball(y, q, percentile) for y, q in zip(actual, estimate))
            baseline_score = fmean(pinball(y, base, percentile) for y in actual)
            line.update({
                f"{name}_pinball": round(score, 6),
                f"{name}_baseline_pinball": round(baseline_score, 6),
                f"{name}_coverage": round(sum(y <= q for y, q in zip(actual, estimate)) / len(actual), 6),
                f"{name}_baseline_coverage": round(sum(y <= base for y in actual) / len(actual), 6),
            })
        comparison.append(line)
    return comparison
