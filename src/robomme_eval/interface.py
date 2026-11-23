"""Stateful policy interface, observation/action packing, and normalization utilities for RoboMME.

Matches the exact conventions verified across:
- `RoboMME/robomme_benchmark` (`BenchmarkEnvBuilder`, `challenge_interface/policy.py`)
- `RoboMME/DP` (`minmax` normalization on absolute `joint_angle` `(8,)` actions)
- `RoboMME/robomme_policy_learning` (`MME-VLA-Suite`: `q01`/`q99` quantile normalization +
  delta arm joint actions `mask=(True,)*7 + (False,)` with absolute gripper in `[-1, 1]`)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

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

# Tasks that include conditioning video demo frames in `obs["front_rgb_list"][:-1]` on `env.reset()`
# (`exec_start_idx = len(obs["front_rgb_list"]) - 1 > 0`).
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

# Official 1,600-episode RoboMME norm stats from `RoboMME/robomme_policy_learning/assets/norm_stats.json`.
# Note: `actions` stats are computed AFTER `DeltaActions(mask=DELTA_JOINT_MASK)` (`actions[:, :7] -= state[:7]`).
MME_VLA_NORM_STATS: dict[str, dict[str, tuple[float, ...]]] = {
    "state": {
        "mean": (
            0.007005180232226849,
            0.29467275738716125,
            -0.00331046711653471,
            -2.178870677947998,
            0.00200783577747643,
            2.4726014137268066,
            0.8105767965316772,
            0.023638639599084854,
        ),
        "std": (
            0.1478731632232666,
            0.2897662818431854,
            0.10731205344200134,
            0.4198242127895355,
            0.0689874067902565,
            0.3204159438610077,
            0.495117723941803,
            0.016432875767350197,
        ),
        "q01": (
            -0.34438448441028596,
            -0.22709672201871872,
            -0.28122754330635075,
            -2.860268020105362,
            -0.20585966855287552,
            1.5707963705062866,
            -0.4366058077812196,
            0.0,
        ),
        "q99": (
            0.3873456451892854,
            0.9852914290308954,
            0.27367080111503594,
            -1.161573127269745,
            0.20890909433364868,
            3.0531936925649643,
            2.1071231347084045,
            0.03999199910610914,
        ),
    },
    "actions": {
        "mean": (
            0.000614401011262089,
            0.010110578499734402,
            -0.0003670427540782839,
            -0.01857929676771164,
            0.000019355442418600433,
            0.028272755444049835,
            0.0007321431767195463,
            -0.10509656369686127,
        ),
        "std": (
            0.0553252138197422,
            0.13423579931259155,
            0.03725232183933258,
            0.13932852447032928,
            0.03795165568590164,
            0.14365772902965546,
            0.11429385095834732,
            0.9944620132446289,
        ),
        "q01": (
            -0.18040079298019407,
            -0.3191569724082947,
            -0.13117732753753664,
            -0.5344759382843971,
            -0.12701646758317942,
            -0.2923886525869369,
            -0.4027193263053894,
            -1.0,
        ),
        "q99": (
            0.19587624197006226,
            0.3774391655921935,
            0.11749427890777586,
            0.4144394028544425,
            0.12551680266857157,
            0.5218227294564248,
            0.4166772034645081,
            0.9996,
        ),
    },
}


def to_numpy(x: Any) -> np.ndarray:
    """Convert a PyTorch/JAX tensor or array-like object to a CPU NumPy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_uint8_rgb(img: Any) -> np.ndarray:
    """Validate and convert an RGB frame to `(H, W, 3)` `uint8` NumPy array."""
    arr = to_numpy(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels on last axis, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 + 1e-5 and arr.min() >= -1e-5:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def pack_robomme_state(joint_state: Any, gripper_state: Any) -> np.ndarray:
    """Pack raw simulator `joint_state` `(7,)` + `gripper_state` `(2,)` into `(8,)` `float32`.

    Uses `gripper_state[:1]` (first Panda finger width in `[0, 0.04]` m), matching both
    `Yinpei/robomme_data_lerobot` (`state` column) and `MME-VLA-Suite` (`env_runner.pack_state`).
    """
    j = to_numpy(joint_state).reshape(-1).astype(np.float32)
    g = to_numpy(gripper_state).reshape(-1).astype(np.float32)
    if j.shape[0] != 7:
        raise ValueError(f"Expected 7 joint angles in joint_state, got shape {j.shape}")
    if g.shape[0] < 1:
        raise ValueError(f"Expected >=1 gripper width in gripper_state, got shape {g.shape}")
    return np.concatenate([j, g[:1]], axis=0, dtype=np.float32)


class ActionStateNormalizer:
    """Exact state/action normalizer supporting both `RoboMME/DP` and `MME-VLA-Suite` conventions.

    Modes:
    - `"quantile_delta"` (`MME-VLA-Suite` / `pi05` convention):
      * `state` `(..., 8)` normalized via `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`.
      * `actions` `(..., T, 8)`: arm joints `0..6` converted to delta relative to current raw `state[:7]`
        (`actions[..., :7] - state[..., None, :7]`), gripper `7` kept absolute in `[-1, 1]`, then
        normalized via `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`.
    - `"minmax_abs"` (`RoboMME/DP` convention):
      * Both `state` and absolute `actions` normalized via `(x - low) / (high - low + 1e-6) * 2 - 1`.
    """

    def __init__(
        self,
        mode: Literal["quantile_delta", "minmax_abs"] = "quantile_delta",
        stats: dict[str, dict[str, Sequence[float]]] | None = None,
        delta_mask: Sequence[bool] = DELTA_JOINT_MASK,
        binarize_gripper_on_decode: bool = False,
    ) -> None:
        self.mode = mode
        raw_stats = MME_VLA_NORM_STATS if stats is None else stats
        self.delta_mask = np.asarray(delta_mask, dtype=bool)
        self.binarize_gripper_on_decode = binarize_gripper_on_decode

        low_key = "q01" if mode == "quantile_delta" else "min"
        high_key = "q99" if mode == "quantile_delta" else "max"
        if low_key not in raw_stats["state"] or high_key not in raw_stats["state"]:
            raise ValueError(f"Missing {low_key}/{high_key} in state stats for mode={mode!r}")

        self.state_low = np.asarray(raw_stats["state"][low_key], dtype=np.float32)
        self.state_high = np.asarray(raw_stats["state"][high_key], dtype=np.float32)
        self.action_low = np.asarray(raw_stats["actions"][low_key], dtype=np.float32)
        self.action_high = np.asarray(raw_stats["actions"][high_key], dtype=np.float32)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32)
        return (x - self.state_low) / (self.state_high - self.state_low + 1e-6) * 2.0 - 1.0

    def unnormalize_state(self, norm_state: np.ndarray) -> np.ndarray:
        x = np.asarray(norm_state, dtype=np.float32)
        return (x + 1.0) / 2.0 * (self.state_high - self.state_low + 1e-6) + self.state_low

    def encode_actions(self, raw_actions: np.ndarray, raw_state: np.ndarray | None = None) -> np.ndarray:
        act = np.asarray(raw_actions, dtype=np.float32).copy()
        if self.mode == "quantile_delta":
            if raw_state is None:
                raise ValueError("raw_state is required when encoding delta actions")
            st = np.asarray(raw_state, dtype=np.float32)
            offset = np.where(self.delta_mask, st[..., : self.delta_mask.shape[0]], 0.0)
            if act.ndim == st.ndim + 1:
                offset = np.expand_dims(offset, axis=-2)
            act[..., : self.delta_mask.shape[0]] -= offset
        return (act - self.action_low) / (self.action_high - self.action_low + 1e-6) * 2.0 - 1.0

    def decode_actions(self, norm_actions: np.ndarray, raw_state: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(norm_actions, dtype=np.float32)
        act = (x + 1.0) / 2.0 * (self.action_high - self.action_low + 1e-6) + self.action_low
        if self.mode == "quantile_delta":
            if raw_state is None:
                raise ValueError("raw_state is required when decoding delta actions")
            st = np.asarray(raw_state, dtype=np.float32)
            offset = np.where(self.delta_mask, st[..., : self.delta_mask.shape[0]], 0.0)
            if act.ndim == st.ndim + 1:
                offset = np.expand_dims(offset, axis=-2)
            act[..., : self.delta_mask.shape[0]] += offset
        if self.binarize_gripper_on_decode:
            act[..., 7] = np.where(act[..., 7] >= 0.0, 1.0, -1.0)
        return act.astype(np.float32)


@dataclass
class PolicyChunkOutput:
    """Output of a single action-chunk prediction step."""

    actions: np.ndarray  # float32 array of shape (chunk_horizon, 8) in simulator joint_angle space
    loop_depth: float | int = 1  # effective loop depth L_t used for this chunk (supports adaptive halting)
    aux_info: dict[str, Any] = field(default_factory=dict)


class StatefulPolicy(abc.ABC):
    """Base interface for recurrent / memory-augmented visuomotor policies on RoboMME."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all per-episode recurrent memory state at the start of a new episode."""

    def set_seed(self, seed: int) -> None:
        """Optional hook to re-seed stochastic sampling (e.g. Flow Matching / DDPM initial noise)."""

    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        """Ingest conditioning video demonstration frames `obs[*][:-1]` at episode start (`step 0`).

        For non-video tasks (`len == 0`), this is a no-op by default.
        For video-conditioned tasks (`len > 0`, e.g. `VideoUnmaskSwap`), subclasses update their
        recurrent memory sequentially over all demonstration frames before action execution begins.
        """

    def ingest_step_history(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        """Optional hook receiving intermediate execution frames collected during the previous chunk.

        Called right before `predict_chunk` on chunks `k >= 1` with the `exec_horizon - 1` frames
        observed between the previous chunk start and the current observation (excluding the current
        frame passed into `predict_chunk`, so no frame is double-counted).
        """

    @abc.abstractmethod
    def predict_chunk(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        task_goal: str,
        *,
        step_idx: int = 0,
        exec_start_idx: int = 0,
    ) -> PolicyChunkOutput:
        """Predict an absolute `joint_angle` action chunk `(pred_horizon, 8)` from current observation."""


class DummyStatefulLoopedPolicy(StatefulPolicy):
    """Lightweight stateful test policy for end-to-end evaluator smoke testing & timing benchmarks.

    Verifies:
    - Episode `reset()` clears recurrent state (`demo_frames_seen`, `exec_frames_seen`, `hidden_state`).
    - `ingest_demo_frames` updates recurrent memory across all `obs[*][:-1]` video demo frames.
    - `ingest_step_history` + `predict_chunk` track every execution step without dropping or duplicating frames.
    - Emits valid `(pred_horizon, 8)` `joint_angle` chunks and simulated adaptive loop depth `L_t` in `[1, max_loops]`.
    """

    def __init__(
        self,
        pred_horizon: int = 16,
        max_loops: int = 4,
        noise_std: float = 0.005,
        hold_current_joints: bool = True,
        seed: int = 7,
    ) -> None:
        self.pred_horizon = pred_horizon
        self.max_loops = max_loops
        self.noise_std = noise_std
        self.hold_current_joints = hold_current_joints
        self.rng = np.random.default_rng(seed)
        self.home_action = np.array(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
            dtype=np.float32,
        )
        self.hidden_state = np.zeros(16, dtype=np.float32)
        self.demo_frames_seen = 0
        self.exec_frames_seen = 0
        self.chunks_predicted = 0

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.hidden_state = np.zeros(16, dtype=np.float32)
        self.demo_frames_seen = 0
        self.exec_frames_seen = 0
        self.chunks_predicted = 0

    def _update_memory(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray) -> None:
        f_rgb = to_uint8_rgb(front_rgb)
        w_rgb = to_uint8_rgb(wrist_rgb)
        st = np.asarray(state, dtype=np.float32)
        if f_rgb.shape != (256, 256, 3) or w_rgb.shape != (256, 256, 3):
            raise ValueError(f"Unexpected RGB shapes: front={f_rgb.shape}, wrist={w_rgb.shape}")
        if st.shape != (8,):
            raise ValueError(f"Unexpected packed state shape: {st.shape}")

        feat = np.zeros(16, dtype=np.float32)
        feat[:8] = st
        feat[8:11] = f_rgb.mean(axis=(0, 1)) / 255.0
        feat[11:14] = w_rgb.mean(axis=(0, 1)) / 255.0
        feat[14] = float(f_rgb.std()) / 255.0
        feat[15] = 1.0
        # Gated exponential moving update mimicking recurrent memory retention
        self.hidden_state = 0.9 * self.hidden_state + 0.1 * feat

    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        for f_rgb, w_rgb, st in zip(front_rgb_list, wrist_rgb_list, state_list):
            self._update_memory(f_rgb, w_rgb, st)
            self.demo_frames_seen += 1

    def ingest_step_history(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        for f_rgb, w_rgb, st in zip(front_rgb_list, wrist_rgb_list, state_list):
            self._update_memory(f_rgb, w_rgb, st)
            self.exec_frames_seen += 1

    def predict_chunk(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        task_goal: str,
        *,
        step_idx: int = 0,
        exec_start_idx: int = 0,
    ) -> PolicyChunkOutput:
        self._update_memory(front_rgb, wrist_rgb, state)
        self.exec_frames_seen += 1
        self.chunks_predicted += 1

        base = self.home_action.copy()
        if self.hold_current_joints:
            base[:7] = np.asarray(state[:7], dtype=np.float32)
            base[7] = 1.0

        noise = self.rng.normal(0.0, self.noise_std, size=(self.pred_horizon, 8)).astype(np.float32)
        noise[:, 7] = 0.0
        actions = base[None, :] + noise

        # Deterministic-plus-context adaptive loop depth in [1, max_loops]
        mem_energy = float(np.linalg.norm(self.hidden_state[:8]))
        bonus_loop = 1 if self.demo_frames_seen > 0 else 0
        loop_depth = min(
            self.max_loops,
            max(1, 1 + bonus_loop + int((self.chunks_predicted + int(mem_energy * 10)) % max(1, self.max_loops - 1))),
        )
        return PolicyChunkOutput(
            actions=actions,
            loop_depth=loop_depth,
            aux_info={
                "demo_frames_seen": self.demo_frames_seen,
                "exec_frames_seen": self.exec_frames_seen,
                "mem_norm": round(float(np.linalg.norm(self.hidden_state)), 4),
            },
        )


class ChallengePolicyAdapter:
    """Wraps any `StatefulPolicy` into RoboMME's `challenge_interface/policy.py` (`infer` / `reset`) API."""

    def __init__(self, policy: StatefulPolicy, exec_horizon: int = 16) -> None:
        self.policy = policy
        self.exec_horizon = exec_horizon
        self.exec_start_idx = 0
        self.step_cursor = 0

    def reset(self) -> None:
        self.policy.reset()
        self.exec_start_idx = 0
        self.step_cursor = 0

    def infer(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        task_goals = inputs["task_goal"]
        task_goal = task_goals[0] if isinstance(task_goals, (list, tuple)) else str(task_goals)

        front_list = [to_uint8_rgb(x) for x in inputs["front_rgb_list"]]
        wrist_list = [to_uint8_rgb(x) for x in inputs["wrist_rgb_list"]]
        state_list = [
            pack_robomme_state(j, g)
            for j, g in zip(inputs["joint_state_list"], inputs["gripper_state_list"])
        ]

        if inputs.get("is_first_step", False):
            self.exec_start_idx = len(front_list) - 1
            if self.exec_start_idx > 0:
                self.policy.ingest_demo_frames(
                    front_list[:-1], wrist_list[:-1], state_list[:-1], task_goal
                )
            self.step_cursor = self.exec_start_idx
        else:
            if len(front_list) > 1:
                self.policy.ingest_step_history(
                    front_list[:-1], wrist_list[:-1], state_list[:-1], task_goal
                )
            self.step_cursor += len(front_list)

        out = self.policy.predict_chunk(
            front_list[-1],
            wrist_list[-1],
            state_list[-1],
            task_goal,
            step_idx=self.step_cursor,
            exec_start_idx=self.exec_start_idx,
        )
        return {"actions": np.asarray(out.actions[: self.exec_horizon], dtype=np.float32)}
