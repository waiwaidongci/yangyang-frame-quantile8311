"""Application entry points for the shot ledger and the background worker."""

from .boosting import QUANTILES, fit_quantile, predict_shot, shot_features
from .evaluation import holdout_comparison
from .fixtures import demo_rows
from .records import normalize_rows
from .reports import holdout_shots, model_profile, pinball_table, scene_manifest, shift_queue
from .settings import SPEC, parse_settings

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict) -> dict:
    options = parse_settings(settings)
    records = sorted(normalize_rows(rows), key=lambda row: (row["scene_id"], row["id"]))
    train = [row for row in records if row["split"] == "train"]
    features = [shot_features(row) for row in train]
    actual = [row["actual_minutes"] for row in train]
    p50, p90 = (fit_quantile(features, actual, quantile, options) for quantile in QUANTILES)
    models = (p50, p90)
    baselines = {50: p50.constant, 90: p90.constant}
    holdout = [predict_shot(row, models) for row in records if row["split"] == "holdout"]
    pending = [predict_shot(row, models) for row in records if row["split"] == "pending"]
    comparison = holdout_comparison(holdout, baselines)
    queue, scheduled, reserved = shift_queue(pending, options["shift_minutes"])
    crossings = sum(row["crossing_adjusted"] for row in holdout + pending)
    warnings = []
    if crossings:
        warnings.append(f"{crossings} 个镜头出现分位交叉，已保留 P50 并将 P90 提至同值。")
    if len(holdout) < 20:
        warnings.append("留出镜头少于 20 个，覆盖率容易受个别镜头影响。")
    if len(train) < 2 * options["min_leaf"]:
        warnings.append("历史镜头不足以分裂回归树，本次模型输出历史常量分位。")
    if not pending:
        warnings.append("本次台账没有待排镜头，交接工时单为空。")
    overall = comparison[0]
    train_scenes = len({row["scene_id"] for row in train})
    deferred = len(pending) - scheduled
    return {
        "summary": (
            f"用 {train_scenes} 个历史场次的 {len(train)} 个镜头训练 P50 和 P90。"
            f"在 {len(comparison) - 1} 个独立留出场次的 {len(holdout)} 个镜头上对照常量分位。"
            f"本班可排 {scheduled} 个镜头，共预留 {reserved} 分钟，{deferred} 个需顺延复核。"
        ),
        "metrics": {
            "P50 留出 Pinball": overall["p50_pinball"],
            "P90 留出 Pinball": overall["p90_pinball"],
            "P50 常量 Pinball": overall["p50_baseline_pinball"],
            "P90 常量 Pinball": overall["p90_baseline_pinball"],
            "P50 留出覆盖率": overall["p50_coverage"],
            "P90 留出覆盖率": overall["p90_coverage"],
            "已分配镜头数": scheduled,
            "待顺延镜头数": deferred,
            "本班已预留分钟": reserved,
        },
        "tables": [queue, pinball_table(comparison), holdout_shots(holdout), scene_manifest(records),
                   model_profile(options, baselines)],
        "warnings": warnings,
    }
