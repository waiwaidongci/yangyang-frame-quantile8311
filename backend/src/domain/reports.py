"""Build scalar export tables and the single-station handoff queue."""

from collections import Counter
from math import ceil

from .boosting import QUANTILES
from .settings import FEATURES, FIELDS, SETTINGS

PREDICTION_LABELS = {
    **{field: FIELDS[field] for field in ("id", "scene_id", *FEATURES)},
    "p50_minutes": "P50 工时（分钟）",
    "p90_minutes": "P90 工时（分钟）",
    "crossing_adjusted": "分位交叉已修正",
}


def table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def project_row(source: dict, columns: dict) -> dict:
    return {key: source[key] for key in columns}


def shift_queue(shots: list[dict], shift_minutes: int) -> tuple[dict, int, int]:
    ordered = sorted(shots, key=lambda row: (row["due_minutes"], -row["p90_minutes"], row["id"]))
    elapsed = scheduled = 0
    rows = []
    for position, shot in enumerate(ordered, start=1):
        reserved = ceil(shot["p90_minutes"])
        finish = elapsed + reserved
        latest = min(shift_minutes, shot["due_minutes"])
        if finish <= latest:
            start, end = elapsed, finish
            elapsed = finish
            scheduled += 1
            action, reason = "本班可排", "P90 预留在交接时限及下班前结束"
        else:
            start = end = None
            action = "顺延复核"
            reason = "预计超出交接时限" if shot["due_minutes"] < shift_minutes else "预计超出本班工位容量"
        rows.append({
            "position": position, **project_row(shot, PREDICTION_LABELS), "due_minutes": shot["due_minutes"],
            "reserve_minutes": reserved, "start_minute": start, "finish_minute": end,
            "action": action, "reason": reason,
        })
    labels = {
        "position": "交接复核顺序", **PREDICTION_LABELS, "due_minutes": FIELDS["due_minutes"],
        "reserve_minutes": "P90 向上取整预留分钟", "start_minute": "开班后开始分钟",
        "finish_minute": "开班后完成分钟", "action": "排班建议", "reason": "建议依据",
    }
    return table("shift_queue", "擦线班交接工时单", labels, rows), scheduled, elapsed


def pinball_table(comparison: list[dict]) -> dict:
    labels = {"scope": "统计范围", "scene_id": FIELDS["scene_id"], "shot_count": "留出镜头数"}
    for percentile in QUANTILES:
        key = f"p{percentile}"
        labels.update({
            f"{key}_pinball": f"P{percentile} 模型 Pinball",
            f"{key}_baseline_pinball": f"P{percentile} 常量 Pinball",
            f"{key}_coverage": f"P{percentile} 模型覆盖率",
            f"{key}_baseline_coverage": f"P{percentile} 常量覆盖率",
        })
    return table("holdout_metrics", "整场留出 Pinball 与覆盖率", labels, comparison)


def holdout_shots(shots: list[dict]) -> dict:
    labels = {**PREDICTION_LABELS, "actual_minutes": FIELDS["actual_minutes"]}
    return table("holdout_shots", "留出镜头预测及实测工时", labels, [project_row(shot, labels) for shot in shots])


def scene_manifest(shots: list[dict]) -> dict:
    counts = Counter((shot["split"], shot["scene_id"]) for shot in shots)
    rows = [
        {"split": split, "scene_id": scene, "shot_count": counts[split, scene]}
        for split in ("train", "holdout", "pending")
        for cohort, scene in sorted(counts)
        if cohort == split
    ]
    labels = {"split": FIELDS["split"], "scene_id": FIELDS["scene_id"], "shot_count": "镜头数"}
    return table("scene_manifest", "数据分区与场次清单", labels, rows)


def model_profile(settings: dict, baselines: dict[int, float]) -> dict:
    rows = [{"parameter": item["key"], "description": item["label"], "value": settings[item["key"]]} for item in SETTINGS]
    rows.extend({"parameter": f"baseline_p{q}", "description": f"历史常量 P{q}（分钟）", "value": value}
                for q, value in baselines.items())
    return table("model_profile", "冻结模型与工位参数", {"parameter": "设置项", "description": "口径", "value": "取值"}, rows)
