"""Create a reproducible shot ledger with nonlinear work and asymmetric delays."""

from random import Random

from .settings import COLUMNS


def demo_rows() -> list[dict]:
    rng = Random(20261125)
    rows = []
    for split, prefix, scenes, shots in (("train", "H", 18, 14), ("holdout", "V", 5, 14), ("pending", "P", 2, 6)):
        for scene_index in range(1, scenes + 1):
            scene = f"RA-{prefix}{scene_index:02d}"
            scene_shift = rng.uniform(-3, 3)
            for number in range(1, shots + 1):
                frames = rng.randint(20, 195)
                occlusion = rng.randint(3, 86)
                overlap = rng.randint(0, 4)
                marks = rng.randint(1, 7)
                due = actual = None
                if split == "pending":
                    due = (60, 95, 130, 170, 210, 260)[(number + scene_index - 2) % 6]
                else:
                    delay = rng.expovariate(1 / (2 + occlusion / 16))
                    intersection = 0.0028 * frames * occlusion if occlusion >= 40 else 0
                    total = 8 + 0.29 * frames + 2 * marks + 3 * overlap**2 + intersection + scene_shift + delay
                    actual = round(total, 2)
                values = (f"{scene}-{number:02d}", scene, split, frames, occlusion, overlap, marks, due, actual)
                rows.append(dict(zip(COLUMNS, values)))
    return rows
