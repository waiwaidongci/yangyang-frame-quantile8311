"""The shot ledger, model inputs, and settings exposed by the API."""

FEATURES = ("frames", "occlusion_pct", "overlapping_characters", "rig_marks")
COLUMNS = ("id", "scene_id", "split", *FEATURES, "due_minutes", "actual_minutes")
FIELDS = {
    "id": "镜头编号",
    "scene_id": "场次编号",
    "split": "样本用途",
    "frames": "需擦线帧数",
    "occlusion_pct": "道具与角色遮挡比例（%）",
    "occlusion_pct_high": "道具与角色遮挡比例上限（%）",
    "overlapping_characters": "同帧最多交叠角色数",
    "rig_marks": "平均每帧支架与吊线痕迹数",
    "rig_marks_high": "平均每帧支架与吊线痕迹数上限",
    "due_minutes": "开班至审片交接的分钟数",
    "actual_minutes": "实测擦线工时（分钟）",
}
SETTINGS = [
    {"key": "estimators", "label": "每分位提升轮数", "default": 60, "min": 8, "max": 100, "step": 1},
    {"key": "learning_rate", "label": "提升学习率", "default": 0.12, "min": 0.02, "max": 0.30, "step": 0.01},
    {"key": "max_depth", "label": "回归树最大深度", "default": 2, "min": 1, "max": 3, "step": 1},
    {"key": "min_leaf", "label": "叶节点最少镜头数", "default": 8, "min": 2, "max": 32, "step": 1},
    {"key": "shift_minutes", "label": "单工位本班可用分钟数", "default": 240, "min": 15, "max": 720, "step": 1},
]
SPEC = {
    "columns": list(COLUMNS),
    "fields": FIELDS,
    "settings": SETTINGS,
    "modules": {
        "datasets": "镜头擦线台账",
        "runs": "分位回归运行",
        "results": "场次留出与工时单",
        "reviews": "审片交接复核",
        "schedule": "范围复核与工位锁定重排",
    },
    "limitation": "覆盖率来自本次留出场次。交接工时单按单工位、无切换损耗执行预排。",
    "range_note": "待排镜头的遮挡比例与支架痕迹可填 下限-上限 范围；复核只收窄范围，工时单按范围内最高 P90 预留。",
    "dataset_name": "20261125-擦线镜头交班.csv",
    "recommendation": "核对场次留出的 P90 覆盖率，再按交接工时单复核本班可排与顺延镜头。",
}


def parse_settings(settings: dict) -> dict[str, int | float]:
    if not isinstance(settings, dict) or settings.keys() - {item["key"] for item in SETTINGS}:
        raise ValueError("存在本项目未定义的运行参数。")
    parsed = {}
    for item in SETTINGS:
        key = item["key"]
        value = settings.get(key, item["default"])
        if type(value) not in (int, float) or not item["min"] <= value <= item["max"]:
            raise ValueError(f'{item["label"]}须为 {item["min"]} 至 {item["max"]} 之间的数值。')
        if item["step"] == 1:
            if value != int(value):
                raise ValueError(f'{item["label"]}须为整数。')
            value = int(value)
        parsed[key] = value
    return parsed
