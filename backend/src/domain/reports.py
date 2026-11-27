"""Build scalar export tables and the single-station handoff queue."""

from collections import Counter

from .boosting import QUANTILES
from .scheduling import arrange
from .settings import FEATURES, FIELDS, SETTINGS

PREDICTION_LABELS = {
    **{field: FIELDS[field] for field in ("id", "scene_id", *FEATURES)},
    "p50_minutes": "P50 工时（分钟）",
    "p90_minutes": "P90 工时（分钟）",
    "crossing_adjusted": "分位交叉已修正",
}

QUEUE_FEATURES = {
    "id": FIELDS["id"],
    "scene_id": FIELDS["scene_id"],
    "frames": FIELDS["frames"],
    "occlusion_pct": FIELDS["occlusion_pct"],
    "occlusion_pct_high": FIELDS["occlusion_pct_high"],
    "overlapping_characters": FIELDS["overlapping_characters"],
    "rig_marks": FIELDS["rig_marks"],
    "rig_marks_high": FIELDS["rig_marks_high"],
    "p50_minutes": "P50 工时（分钟）",
    "p90_minutes": "P90 工时（分钟）",
    "crossing_adjusted": "分位交叉已修正",
}

ACTION_LABELS = {"locked": "已锁定", "scheduled": "本班可排", "deferred": "顺延复核"}


def table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def project_row(source: dict, columns: dict) -> dict:
    return {key: source[key] for key in columns}


def priority_order(shots: list[dict]) -> list[dict]:
    return sorted(shots, key=lambda row: (row["due_minutes"], -row["p90_minutes"], row["id"]))


def defer_reason(shot: dict, shift_minutes: int) -> str:
    return "预计超出交接时限" if shot["due_minutes"] < shift_minutes else "预计超出本班工位容量"


def shift_queue(shots: list[dict], shift_minutes: int) -> tuple[dict, int, int]:
    placements = arrange(shots, shift_minutes, [])
    rows = []
    for position, shot in enumerate(priority_order(shots), start=1):
        placement = placements[shot["id"]]
        scheduled = placement["action"] == "scheduled"
        rows.append({
            "position": position, **project_row(shot, QUEUE_FEATURES), "due_minutes": shot["due_minutes"],
            "reserve_minutes": placement["reserve_minutes"],
            "start_minute": placement["start_minute"], "finish_minute": placement["finish_minute"],
            "action": ACTION_LABELS[placement["action"]],
            "reason": "P90 预留在交接时限及下班前结束" if scheduled else defer_reason(shot, shift_minutes),
        })
    labels = {
        "position": "交接复核顺序", **QUEUE_FEATURES, "due_minutes": FIELDS["due_minutes"],
        "reserve_minutes": "P90 向上取整预留分钟", "start_minute": "开班后开始分钟",
        "finish_minute": "开班后完成分钟", "action": "排班建议", "reason": "建议依据",
    }
    done = [placement["reserve_minutes"] for placement in placements.values() if placement["action"] == "scheduled"]
    return table("shift_queue", "擦线班交接工时单", labels, rows), len(done), sum(done)


def live_schedule(shots: list[dict], shift_minutes: int, locks: list[dict], versions: dict) -> dict:
    """Current handoff sheet: locked slots as-is, unlocked shots re-arranged."""
    placements = arrange(shots, shift_minutes, locks)
    priority = {shot["id"]: rank for rank, shot in enumerate(priority_order(shots))}
    placed = [shot for shot in shots if placements[shot["id"]]["start_minute"] is not None]
    placed.sort(key=lambda shot: (placements[shot["id"]]["start_minute"], shot["id"]))
    deferred = [shot for shot in shots if placements[shot["id"]]["start_minute"] is None]
    deferred.sort(key=lambda shot: priority[shot["id"]])
    rows = []
    for position, shot in enumerate([*placed, *deferred], start=1):
        placement = placements[shot["id"]]
        action = placement["action"]
        if action == "locked":
            reason = "已锁定时段原样交接"
        elif action == "scheduled":
            reason = "P90 预留在交接时限及下班前结束"
        else:
            reason = defer_reason(shot, shift_minutes)
        rows.append({
            "position": position, **project_row(shot, QUEUE_FEATURES), "due_minutes": shot["due_minutes"],
            "confirmed_version": versions.get(shot["id"]),
            "reserve_minutes": placement["reserve_minutes"],
            "start_minute": placement["start_minute"], "finish_minute": placement["finish_minute"],
            "action": ACTION_LABELS[action], "reason": reason,
        })
    labels = {
        "position": "交接顺序", **QUEUE_FEATURES, "due_minutes": FIELDS["due_minutes"],
        "confirmed_version": "生效复核版本", "reserve_minutes": "预留分钟",
        "start_minute": "开班后开始分钟", "finish_minute": "开班后完成分钟",
        "action": "排班建议", "reason": "建议依据",
    }
    return table("live_schedule", "交接工时单（按最新复核重排）", labels, rows)


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
