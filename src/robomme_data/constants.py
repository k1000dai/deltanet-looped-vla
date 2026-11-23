"""Canonical RoboMME task, suite, and LeRobot Parquet block-mapping constants."""

from __future__ import annotations

ROBOMME_SUITES: dict[str, tuple[str, ...]] = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "Permanence": ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}

ROBOMME_TASKS: tuple[str, ...] = tuple(
    task for suite_tasks in ROBOMME_SUITES.values() for task in suite_tasks
)

TASK_TO_SUITE: dict[str, str] = {
    task: suite for suite, suite_tasks in ROBOMME_SUITES.items() for task in suite_tasks
}

# Verified 16 x 100-episode contiguous block order in `Yinpei/robomme_data_lerobot` (`epis_idx` 0..1599).
LEROBOT_BLOCK_TO_TASK: tuple[str, ...] = (
    "PatternLock",       # Block  0: epis_idx    0..  99
    "ButtonUnmaskSwap",  # Block  1: epis_idx  100.. 199
    "ButtonUnmask",      # Block  2: epis_idx  200.. 299
    "VideoPlaceButton",  # Block  3: epis_idx  300.. 399
    "VideoUnmaskSwap",   # Block  4: epis_idx  400.. 499
    "PickXtimes",        # Block  5: epis_idx  500.. 599
    "StopCube",          # Block  6: epis_idx  600.. 699
    "SwingXtimes",       # Block  7: epis_idx  700.. 799
    "PickHighlight",     # Block  8: epis_idx  800.. 899
    "MoveCube",          # Block  9: epis_idx  900.. 999
    "InsertPeg",         # Block 10: epis_idx 1000..1099
    "RouteStick",        # Block 11: epis_idx 1100..1199
    "BinFill",           # Block 12: epis_idx 1200..1299
    "VideoPlaceOrder",   # Block 13: epis_idx 1300..1399
    "VideoRepick",       # Block 14: epis_idx 1400..1499
    "VideoUnmask",       # Block 15: epis_idx 1500..1599
)

TASK_TO_LEROBOT_BLOCK: dict[str, int] = {
    task: blk for blk, task in enumerate(LEROBOT_BLOCK_TO_TASK)
}

TASKS_WITH_VIDEO_DEMO: tuple[str, ...] = (
    "VideoUnmask",
    "VideoUnmaskSwap",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "VideoRepick",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)

DELTA_JOINT_MASK: tuple[bool, ...] = (True, True, True, True, True, True, True, False)


def epis_idx_to_task(epis_idx: int) -> str:
    """Map global `Yinpei/robomme_data_lerobot` `epis_idx` (`0..1599`) to its `EnvID`."""
    blk = int(epis_idx) // 100
    if not (0 <= blk < len(LEROBOT_BLOCK_TO_TASK)):
        raise ValueError(f"epis_idx {epis_idx} out of expected [0, 1599] range")
    return LEROBOT_BLOCK_TO_TASK[blk]


def task_to_epis_range(task_name: str) -> tuple[int, int]:
    """Return `(start_inclusive, end_exclusive)` episode index range in `robomme_data_lerobot`."""
    if task_name not in TASK_TO_LEROBOT_BLOCK:
        raise ValueError(f"Unknown RoboMME task {task_name!r}")
    blk = TASK_TO_LEROBOT_BLOCK[task_name]
    return blk * 100, (blk + 1) * 100
