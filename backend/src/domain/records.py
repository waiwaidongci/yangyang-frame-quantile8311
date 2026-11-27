"""Validate shot ledgers and isolate entire scenes across cohorts."""

import math
import re
from collections import Counter

from .settings import COLUMNS, FIELDS

CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SPAN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*")
SPLITS = {"train": "历史拟合", "holdout": "留出评估", "pending": "待排镜头"}
RANGE_FIELDS = ("occlusion_pct", "rig_marks")
HIGH_KEYS = {f"{field}_high" for field in RANGE_FIELDS}


def parse_code(value, key: str, line: int) -> str:
    if isinstance(value, str) and CODE.fullmatch(value.strip()):
        return value.strip()
    raise ValueError(f"第 {line} 条的{FIELDS[key]}须为 1 至 64 位编号，以英文字母或数字开头。")


def parse_number(value, key: str, line: int, minimum: float, maximum: float, *, integer=False):
    suffix = "整数" if integer else "数值"
    message = f"第 {line} 条的{FIELDS[key]}须为 {minimum} 至 {maximum} 之间的{suffix}。"
    if type(value) not in (int, float, str):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(message)
    if integer:
        if not parsed.is_integer():
            raise ValueError(message)
        return int(parsed)
    return parsed


def blank(value) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def parse_measurement(raw: dict, key: str, line: int, minimum: float, maximum: float,
                      allow_range: bool) -> tuple[float, float]:
    """Parse a ledger measurement into (lower, upper); exact values collapse to a point."""
    value = raw[key]
    if allow_range and isinstance(value, str):
        span = SPAN.fullmatch(value)
        if span:
            low = parse_number(span.group(1), key, line, minimum, maximum)
            high = parse_number(span.group(2), key, line, minimum, maximum)
            if low > high:
                raise ValueError(f"第 {line} 条的{FIELDS[key]}范围下限不能超过上限。")
            return low, high
    low = parse_number(value, key, line, minimum, maximum)
    stored = raw.get(f"{key}_high")
    high = low if stored is None else parse_number(stored, key, line, minimum, maximum)
    if low > high:
        raise ValueError(f"第 {line} 条的{FIELDS[key]}范围下限不能超过上限。")
    return low, high


def normalize_rows(rows: list[dict]) -> list[dict]:
    ids = set()
    scene_splits = {}
    counts = Counter()
    normalized = []
    shapes = (set(COLUMNS), set(COLUMNS) | HIGH_KEYS)
    for line, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict) or raw.keys() not in shapes:
            raise ValueError(f"第 {line} 条的列须与模板一致，包含 {', '.join(COLUMNS)}。")
        shot = parse_code(raw["id"], "id", line)
        if shot in ids:
            raise ValueError(f"第 {line} 条的镜头编号 {shot} 重复。")
        ids.add(shot)
        scene = parse_code(raw["scene_id"], "scene_id", line)
        split = raw["split"].strip() if isinstance(raw["split"], str) else None
        if split not in SPLITS:
            raise ValueError(f"第 {line} 条的样本用途须为 train、holdout 或 pending。")
        if scene_splits.setdefault(scene, split) != split:
            raise ValueError(f"场次 {scene} 同时出现在不同样本用途，请按整场划分。")
        due, actual = raw["due_minutes"], raw["actual_minutes"]
        if split == "pending":
            if not blank(actual):
                raise ValueError(f"第 {line} 条待排镜头尚未完工，请将实测工时留空。")
            due = parse_number(due, "due_minutes", line, 1, 1440, integer=True)
            actual = None
        else:
            if not blank(due):
                raise ValueError(f"第 {line} 条已完工镜头的交接剩余时限须留空。")
            due = None
            actual = parse_number(actual, "actual_minutes", line, 0.01, 1440)
        pending = split == "pending"
        occlusion = parse_measurement(raw, "occlusion_pct", line, 0, 100, allow_range=pending)
        rig = parse_measurement(raw, "rig_marks", line, 0, 30, allow_range=pending)
        normalized.append({
            "id": shot, "scene_id": scene, "split": split,
            "frames": parse_number(raw["frames"], "frames", line, 1, 1800, integer=True),
            "occlusion_pct": occlusion[0], "occlusion_pct_high": occlusion[1],
            "overlapping_characters": parse_number(raw["overlapping_characters"], "overlapping_characters", line, 0, 12, integer=True),
            "rig_marks": rig[0], "rig_marks_high": rig[1],
            "due_minutes": due, "actual_minutes": actual,
        })
        counts[split] += 1

    if counts["train"] < 30 or sum(split == "train" for split in scene_splits.values()) < 3:
        raise ValueError("历史拟合至少需要 30 个已完工镜头，且覆盖至少 3 个场次。")
    if counts["holdout"] < 4:
        raise ValueError("留出评估至少需要一个独立场次中的 4 个已完工镜头，或跨独立场次合计 4 个。")
    return normalized
