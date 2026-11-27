"""Re-arrange unlocked shots around locked slots on the single workstation."""

from bisect import insort
from math import ceil


def range_within(inner: tuple[float, float], outer: tuple[float, float]) -> bool:
    """A review may only narrow the recorded range, never widen it."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def latest_confirmations(records: list[dict]) -> dict[str, dict]:
    """Out-of-order confirmations resolve to the highest version per shot."""
    latest = {}
    for record in records:
        current = latest.get(record["shot_id"])
        if current is None or record["version"] > current["version"]:
            latest[record["shot_id"]] = record
    return latest


def apply_confirmation(row: dict, confirmation: dict | None) -> dict:
    if confirmation is None:
        return row
    return {
        **row,
        "occlusion_pct": confirmation["occlusion_low"],
        "occlusion_pct_high": confirmation["occlusion_high"],
        "rig_marks": confirmation["rig_low"],
        "rig_marks_high": confirmation["rig_high"],
    }


def overlaps(start: int, finish: int, slots: list[dict]) -> bool:
    """One workstation slot takes a single shot at a time."""
    return any(start < slot["finish_minute"] and finish > slot["start_minute"] for slot in slots)


def earliest_start(busy: list[tuple[int, int]], reserve: int, latest: int) -> int | None:
    """Earliest station start that finishes by latest without touching busy slots."""
    cursor = 0
    for start, finish in busy:
        if cursor + reserve <= min(start, latest):
            return cursor
        cursor = max(cursor, finish)
    return cursor if cursor + reserve <= latest else None


def arrange(shots: list[dict], shift_minutes: int, locks: list[dict]) -> dict[str, dict]:
    """Place unlocked shots in priority order around locked slots.

    Locked slots are handed over as-is. Unlocked shots are considered by due
    time, then falling P90, then shot id, and take the earliest free slot that
    still meets their own handover and the end of the shift. The rest defer.
    Returns one placement per shot id.
    """
    locked = {lock["shot_id"]: lock for lock in locks}
    busy = sorted((lock["start_minute"], lock["finish_minute"]) for lock in locks)
    placements = {}
    for lock in locks:
        placements[lock["shot_id"]] = {
            "action": "locked",
            "reserve_minutes": lock["finish_minute"] - lock["start_minute"],
            "start_minute": lock["start_minute"],
            "finish_minute": lock["finish_minute"],
        }
    unlocked = (shot for shot in shots if shot["id"] not in locked)
    ordered = sorted(unlocked, key=lambda row: (row["due_minutes"], -row["p90_minutes"], row["id"]))
    for shot in ordered:
        reserve = ceil(shot["p90_minutes"])
        latest = min(shift_minutes, shot["due_minutes"])
        start = earliest_start(busy, reserve, latest)
        if start is None:
            placements[shot["id"]] = {"action": "deferred", "reserve_minutes": reserve,
                                      "start_minute": None, "finish_minute": None}
            continue
        finish = start + reserve
        insort(busy, (start, finish))
        placements[shot["id"]] = {"action": "scheduled", "reserve_minutes": reserve,
                                  "start_minute": start, "finish_minute": finish}
    return placements
