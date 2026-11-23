"""Reusable headless ManiSkill3/SAPIEN evaluation package for stateful & looped RoboMME policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .interface import (
    ActionStateNormalizer,
    ChallengePolicyAdapter,
    DELTA_JOINT_MASK,
    DummyStatefulLoopedPolicy,
    MME_VLA_NORM_STATS,
    PolicyChunkOutput,
    ROBOMME_SUITES,
    ROBOMME_TASKS,
    StatefulPolicy,
    TASKS_WITH_VIDEO_DEMO,
    TASK_TO_SUITE,
    pack_robomme_state,
    to_numpy,
    to_uint8_rgb,
)
from .metrics import (
    EpisodeResult,
    TaskEvalSummary,
    apply_holm_bonferroni,
    bootstrap_ci,
    compare_did_policy_runs,
    compare_policy_runs,
    paired_bootstrap_ci,
    summarize_episodes,
)

if TYPE_CHECKING:
    from .dp_baseline import DP_MINMAX_NORM_STATS, PretrainedDPPolicy
    from .runner import evaluate_episode, evaluate_policy

__all__ = [
    "ActionStateNormalizer",
    "ChallengePolicyAdapter",
    "DELTA_JOINT_MASK",
    "DP_MINMAX_NORM_STATS",
    "DummyStatefulLoopedPolicy",
    "EpisodeResult",
    "MME_VLA_NORM_STATS",
    "PolicyChunkOutput",
    "PretrainedDPPolicy",
    "ROBOMME_SUITES",
    "ROBOMME_TASKS",
    "StatefulPolicy",
    "TASKS_WITH_VIDEO_DEMO",
    "TASK_TO_SUITE",
    "TaskEvalSummary",
    "bootstrap_ci",
    "compare_policy_runs",
    "evaluate_episode",
    "evaluate_policy",
    "paired_bootstrap_ci",
    "pack_robomme_state",
    "summarize_episodes",
    "to_numpy",
    "to_uint8_rgb",
]


def __getattr__(name: str) -> Any:
    if name in ("evaluate_episode", "evaluate_policy"):
        from . import runner

        return getattr(runner, name)
    if name in ("PretrainedDPPolicy", "DP_MINMAX_NORM_STATS"):
        from . import dp_baseline

        return getattr(dp_baseline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
