"""Memory-mapped `robomme_pi05_cache` dataset loaders (`< 0.5s` startup across all 1,600 episodes).

Provides both:
  - `Pi05EpisodeDataset`: full-episode sequence loader (`T <= 1,411`, 100% demo + execution retention,
    dual-camera `front` `8x8` + `wrist` `4x4` history + exact unpooled `16x16 = 256` tokens at execution
    chunk steps, A3 causal readout cutoffs, dense grounded subgoal targets, and optional `k in {1..3}`
    cross-session distractor episode prefixing for boundary-reset gate training).
  - `Pi05StepDataset`: flat step-level loader over cached `256`-token execution chunk steps (`120k` samples).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from robomme_data.cache_dataset import build_padded_action_chunk
from robomme_data.constants import ROBOMME_TASKS, TASK_TO_SUITE
from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS

from .vision_tower import project_siglip_1152_to_2048

DEFAULT_MIYABI_PI05_CACHE_DIR = "/work/gj26/b20098/datasets/robomme_pi05_cache"


def load_pi05_normalizer(
    cache_dir: str | Path | None = None,
    *,
    stats_variant: Literal["mme_vla", "pi05_baseline", "framesamp_modul"] = "mme_vla",
) -> ActionStateNormalizer:
    """Load exact `quantile_delta` (`q01`/`q99`) `ActionStateNormalizer` for `π0.5`."""
    if cache_dir is not None:
        stats_path = Path(cache_dir) / "norm_stats.json"
        if stats_path.exists():
            payload = json.loads(stats_path.read_text())
            key_map = {
                "mme_vla": "norm_stats",
                "pi05_baseline": "pi05_baseline_norm_stats",
                "framesamp_modul": "framesamp_modul_norm_stats",
            }
            chosen = payload.get(key_map[stats_variant]) or payload.get("norm_stats")
            if chosen is not None:
                return ActionStateNormalizer(
                    mode="quantile_delta",
                    stats={"state": chosen["state"], "actions": chosen["actions"]},
                )
    return ActionStateNormalizer(mode="quantile_delta", stats=MME_VLA_NORM_STATS)


def load_pi05_shards(
    cache_dir: str | Path = DEFAULT_MIYABI_PI05_CACHE_DIR,
    tasks: Sequence[str] | None = None,
    *,
    mmap: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load `{TaskName}.pt` shards via `torch.load(..., mmap=True)` in `< 0.5s`."""
    root = Path(cache_dir)
    target_tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
    shards: dict[str, dict[str, Any]] = {}
    for task in target_tasks:
        p = root / f"{task}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing pi05 cache shard: {p}")
        shards[task] = torch.load(p, map_location="cpu", weights_only=False, mmap=mmap)
    return shards


def compute_episode_exec_grid_steps(exec_start: int, ep_len: int, exec_stride: int = 4) -> np.ndarray:
    """Deterministic execution step indices cached with unpooled `256`-token images inside each episode."""
    if ep_len <= 0:
        return np.zeros((0,), dtype=np.int64)
    start = min(int(exec_start), ep_len - 1)
    steps = np.arange(start, ep_len, max(1, int(exec_stride)), dtype=np.int64)
    if steps.size == 0 or steps[-1] != ep_len - 1:
        steps = np.append(steps, ep_len - 1)
    return steps


class Pi05EpisodeDataset(Dataset):
    """Full-episode `robomme_pi05_cache` dataset supporting Stage A/B/C & cross-session distractor training."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_MIYABI_PI05_CACHE_DIR,
        *,
        tasks: Sequence[str] | None = None,
        episode_indices_per_task: Sequence[int] | None = None,
        stats_variant: Literal["mme_vla", "pi05_baseline", "framesamp_modul"] = "mme_vla",
        action_horizon: int = 20,
        exec_horizon: int = 16,
        exclusion_window: int | None = None,
        random_chunk_phase: bool = True,
        project_history_to_2048: bool = True,
        project_exec_256_to_2048: bool = True,
        load_exec_256: bool = True,
        pad_actions_to_32: bool = False,
        distractor_prob: float = 0.0,
        max_distractors: int = 3,
        max_distractor_frames_each: int = 160,
        feature_dtype: torch.dtype = torch.bfloat16,
        mmap: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
        self.action_horizon = int(action_horizon)
        self.exec_horizon = int(exec_horizon)
        self.exclusion_window = self.exec_horizon if exclusion_window is None else int(exclusion_window)
        self.random_chunk_phase = bool(random_chunk_phase)
        self.project_history_to_2048 = bool(project_history_to_2048)
        self.project_exec_256_to_2048 = bool(project_exec_256_to_2048)
        self.load_exec_256 = bool(load_exec_256)
        self.pad_actions_to_32 = bool(pad_actions_to_32)
        self.distractor_prob = float(distractor_prob)
        self.max_distractors = int(max_distractors)
        self.max_distractor_frames_each = int(max_distractor_frames_each)
        self.feature_dtype = feature_dtype
        self.rng = np.random.default_rng(seed)

        self.normalizer = load_pi05_normalizer(self.cache_dir, stats_variant=stats_variant)
        self.shards = load_pi05_shards(self.cache_dir, tasks=self.tasks, mmap=mmap)

        first_sh = next(iter(self.shards.values()))
        self.head_weight = first_sh["siglip_head_weight"].detach().clone()
        self.head_bias = first_sh["siglip_head_bias"].detach().clone()
        self.exec_stride = int(first_sh.get("exec_256_stride", 4))

        self.index: list[tuple[str, int]] = []
        for task in self.tasks:
            n_eps = int(self.shards[task]["epis_idx"].shape[0])
            ep_ids = (
                range(n_eps)
                if episode_indices_per_task is None
                else [i for i in episode_indices_per_task if 0 <= i < n_eps]
            )
            for ep_local in ep_ids:
                self.index.append((task, int(ep_local)))

    def __len__(self) -> int:
        return len(self.index)

    def _maybe_project(self, x: torch.Tensor, do_proj: bool) -> torch.Tensor:
        out = x.to(dtype=self.feature_dtype)
        if do_proj and out.shape[-1] == 1152:
            out = project_siglip_1152_to_2048(out, self.head_weight, self.head_bias)
        return out

    def _load_single_episode_history(self, task: str, ep_local: int, max_frames: int | None = None) -> dict[str, Any]:
        sh = self.shards[task]
        s = int(sh["ep_offsets"][ep_local].item())
        e = int(sh["ep_offsets"][ep_local + 1].item())
        if max_frames is not None and (e - s) > max_frames:
            s = e - max_frames
        front = self._maybe_project(sh["front_emb"][s:e], self.project_history_to_2048)
        wrist = self._maybe_project(sh["wrist_emb"][s:e], self.project_history_to_2048)
        raw_st = sh["state"][s:e].to(dtype=torch.float32)
        st_norm = torch.from_numpy(self.normalizer.normalize_state(raw_st.numpy()))
        return {
            "front": front,
            "wrist": wrist,
            "state_norm": st_norm,
            "gripper_event": sh["gripper_event"][s:e].to(dtype=torch.bool),
            "proprio_events": sh["proprio_events"][s:e].to(dtype=torch.float32),
            "length": e - s,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, ep_local = self.index[idx]
        sh = self.shards[task]

        s = int(sh["ep_offsets"][ep_local].item())
        e = int(sh["ep_offsets"][ep_local + 1].item())
        ep_len = e - s
        exec_start = int(sh["exec_start_idx"][ep_local].item())
        epis_idx = int(sh["epis_idx"][ep_local].item())

        front_emb = self._maybe_project(sh["front_emb"][s:e], self.project_history_to_2048)
        wrist_emb = self._maybe_project(sh["wrist_emb"][s:e], self.project_history_to_2048)
        raw_state_t = sh["state"][s:e].to(dtype=torch.float32)
        raw_actions_t = sh["actions"][s:e].to(dtype=torch.float32)
        is_demo = sh["is_demo"][s:e].to(dtype=torch.bool)
        raw_state_np = raw_state_t.numpy()
        raw_actions_np = raw_actions_t.numpy()
        state_norm = torch.from_numpy(self.normalizer.normalize_state(raw_state_np))

        # Sample execution chunk steps aligned with `exec_stride` so cached `256`-token VLM views match 100%
        exec_s = int(sh["exec_ep_offsets"][ep_local].item())
        exec_e = int(sh["exec_ep_offsets"][ep_local + 1].item())
        cached_rel_steps = sh["exec_rel_step_idx"][exec_s:exec_e].numpy().astype(np.int64)

        grid_step_per_chunk = max(1, self.exec_horizon // self.exec_stride)
        phase_grid = 0
        if self.random_chunk_phase and cached_rel_steps.size > 1 and grid_step_per_chunk > 1:
            max_p = min(grid_step_per_chunk, cached_rel_steps.size) - 1
            if max_p > 0:
                phase_grid = int(self.rng.integers(0, max_p + 1))

        picked_slots = np.arange(phase_grid, cached_rel_steps.size, grid_step_per_chunk, dtype=np.int64)
        if picked_slots.size == 0:
            picked_slots = np.array([0], dtype=np.int64)
        chunk_steps = cached_rel_steps[picked_slots]

        target_chunks = np.stack(
            [
                build_padded_action_chunk(
                    raw_actions_np,
                    raw_state_np[int(t_step)],
                    int(t_step),
                    self.action_horizon,
                    self.normalizer,
                )
                for t_step in chunk_steps
            ],
            axis=0,
        )
        if self.pad_actions_to_32:
            padded_32 = np.zeros((target_chunks.shape[0], self.action_horizon, 32), dtype=np.float32)
            padded_32[:, :, :8] = target_chunks
            target_chunks_t = torch.from_numpy(padded_32)
        else:
            target_chunks_t = torch.from_numpy(target_chunks)

        raw_cutoff = chunk_steps - self.exclusion_window + 1
        cutoff_excl = np.maximum(exec_start, np.maximum(0, raw_cutoff))

        abs_exec_rows = torch.from_numpy(exec_s + picked_slots)
        if self.load_exec_256:
            exec_front_256 = self._maybe_project(
                sh["exec_front_256"][abs_exec_rows], self.project_exec_256_to_2048
            )
            exec_wrist_256 = self._maybe_project(
                sh["exec_wrist_256"][abs_exec_rows], self.project_exec_256_to_2048
            )
        else:
            exec_front_256 = None
            exec_wrist_256 = None

        # Optional cross-session distractor prefixing (`k in {1..max_distractors}`) for boundary decay gate training
        session_boundary_mask = torch.zeros((ep_len,), dtype=torch.bool)
        prefix_offset = 0
        if self.distractor_prob > 0.0 and self.max_distractors >= 1 and len(self.index) > 1:
            if float(self.rng.random()) < self.distractor_prob:
                k_dist = int(self.rng.integers(1, self.max_distractors + 1))
                dist_fronts, dist_wrists, dist_states, dist_gevt, dist_prop, dist_bounds = [], [], [], [], [], []
                for _ in range(k_dist):
                    d_idx = int(self.rng.integers(0, len(self.index)))
                    d_task, d_ep = self.index[d_idx]
                    d_hist = self._load_single_episode_history(
                        d_task, d_ep, max_frames=self.max_distractor_frames_each
                    )
                    d_len = d_hist["length"]
                    dist_fronts.append(d_hist["front"])
                    dist_wrists.append(d_hist["wrist"])
                    dist_states.append(d_hist["state_norm"])
                    dist_gevt.append(d_hist["gripper_event"])
                    dist_prop.append(d_hist["proprio_events"])
                    b_mask = torch.zeros((d_len,), dtype=torch.bool)
                    if prefix_offset > 0 and d_len > 0:
                        b_mask[0] = True
                    dist_bounds.append(b_mask)
                    prefix_offset += d_len

                if prefix_offset > 0:
                    session_boundary_mask[0] = True
                    front_emb = torch.cat([*dist_fronts, front_emb], dim=0)
                    wrist_emb = torch.cat([*dist_wrists, wrist_emb], dim=0)
                    state_norm = torch.cat([*dist_states, state_norm], dim=0)
                    is_demo = torch.cat(
                        [torch.ones((prefix_offset,), dtype=torch.bool), is_demo], dim=0
                    )
                    session_boundary_mask = torch.cat([*dist_bounds, session_boundary_mask], dim=0)
                    chunk_steps = chunk_steps + prefix_offset
                    cutoff_excl = cutoff_excl + prefix_offset
                    exec_start = exec_start + prefix_offset

        abs_chunk_rows = torch.from_numpy(s + (chunk_steps - prefix_offset))
        out: dict[str, Any] = {
            "task_name": task,
            "suite": sh.get("suite", TASK_TO_SUITE.get(task, "Unknown")),
            "epis_idx": epis_idx,
            "exec_start_idx": exec_start,
            "distractor_prefix_frames": prefix_offset,
            "total_steps": int(front_emb.shape[0]),
            "prompt": sh["prompts"][ep_local],
            "lang_tokens": sh["lang_tokens"][ep_local].to(dtype=torch.long),
            "lang_masks": sh["lang_masks"][ep_local].to(dtype=torch.bool),
            "front_emb": front_emb,                                      # (T_tot, 64, 2048 or 1152) bf16
            "wrist_emb": wrist_emb,                                      # (T_tot, 16, 2048 or 1152) bf16
            "state_norm": state_norm,                                    # (T_tot, 8) float32
            "raw_state": raw_state_t,                                    # (T_ep, 8) float32
            "raw_actions": raw_actions_t,                                # (T_ep, 8) float32
            "is_demo": is_demo,                                          # (T_tot,) bool
            "session_boundary": session_boundary_mask,                   # (T_tot,) bool
            "chunk_step_indices": torch.from_numpy(chunk_steps),         # (M,) int64 (into T_tot)
            "readout_cutoff_excl": torch.from_numpy(cutoff_excl),        # (M,) int64 (into T_tot)
            "target_action_chunks": target_chunks_t,                     # (M, 20, 8 or 32) float32
            # Dense auxiliary targets sliced at `chunk_step_indices` `(M, ...)`
            "chunk_target_xy": sh["target_xy"][abs_chunk_rows].to(dtype=torch.float32),
            "chunk_target_xy_m11": sh["target_xy_m11"][abs_chunk_rows].to(dtype=torch.float32),
            "chunk_has_xy": sh["has_xy"][abs_chunk_rows].to(dtype=torch.bool),
            "chunk_secondary_xy": sh["secondary_xy"][abs_chunk_rows].to(dtype=torch.float32),
            "chunk_has_secondary_xy": sh["has_secondary_xy"][abs_chunk_rows].to(dtype=torch.bool),
            "chunk_count_state": sh["count_state"][abs_chunk_rows].to(dtype=torch.long),
            "chunk_has_count": sh["has_count"][abs_chunk_rows].to(dtype=torch.bool),
            "chunk_phase_id": sh["phase_id"][abs_chunk_rows].to(dtype=torch.long),
            "chunk_color_id": sh["color_id"][abs_chunk_rows].to(dtype=torch.long),
            "chunk_ordinal_id": sh["ordinal_id"][abs_chunk_rows].to(dtype=torch.long),
            "chunk_subgoal_tokens": sh["subgoal_tokens"][abs_chunk_rows].to(dtype=torch.long),
            "chunk_subgoal_masks": sh["subgoal_masks"][abs_chunk_rows].to(dtype=torch.bool),
            # Episode-wide dense frame targets `(T_ep, ...)`
            "ep_target_xy": sh["target_xy"][s:e].to(dtype=torch.float32),
            "ep_has_xy": sh["has_xy"][s:e].to(dtype=torch.bool),
            "ep_count_state": sh["count_state"][s:e].to(dtype=torch.long),
            "ep_has_count": sh["has_count"][s:e].to(dtype=torch.bool),
            "ep_phase_id": sh["phase_id"][s:e].to(dtype=torch.long),
            "ep_gripper_event": sh["gripper_event"][s:e].to(dtype=torch.bool),
            "ep_proprio_events": sh["proprio_events"][s:e].to(dtype=torch.float32),
            "ep_subgoal_boundary": sh["subgoal_boundary"][s:e].to(dtype=torch.bool),
        }
        if exec_front_256 is not None:
            out["exec_front_256"] = exec_front_256                       # (M, 256, 2048 or 1152) bf16
            out["exec_wrist_256"] = exec_wrist_256                       # (M, 256, 2048 or 1152) bf16
        return out


class Pi05StepDataset(Dataset):
    """Flat step-indexed dataset over all cached `256`-token execution steps (`~120,016` steps across 16 tasks)."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_MIYABI_PI05_CACHE_DIR,
        *,
        tasks: Sequence[str] | None = None,
        episode_indices_per_task: Sequence[int] | None = None,
        stats_variant: Literal["mme_vla", "pi05_baseline", "framesamp_modul"] = "mme_vla",
        action_horizon: int = 20,
        exec_horizon: int = 16,
        exclusion_window: int | None = None,
        project_to_2048: bool = True,
        feature_dtype: torch.dtype = torch.bfloat16,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
        self.action_horizon = int(action_horizon)
        self.exec_horizon = int(exec_horizon)
        self.exclusion_window = self.exec_horizon if exclusion_window is None else int(exclusion_window)
        self.project_to_2048 = bool(project_to_2048)
        self.feature_dtype = feature_dtype

        self.normalizer = load_pi05_normalizer(self.cache_dir, stats_variant=stats_variant)
        self.shards = load_pi05_shards(self.cache_dir, tasks=self.tasks, mmap=mmap)

        first_sh = next(iter(self.shards.values()))
        self.head_weight = first_sh["siglip_head_weight"].detach().clone()
        self.head_bias = first_sh["siglip_head_bias"].detach().clone()

        # Flat index `(task, ep_local, exec_row_abs, rel_step_in_ep)`
        self.samples: list[tuple[str, int, int, int]] = []
        for task in self.tasks:
            sh = self.shards[task]
            n_eps = int(sh["epis_idx"].shape[0])
            ep_ids = (
                range(n_eps)
                if episode_indices_per_task is None
                else [i for i in episode_indices_per_task if 0 <= i < n_eps]
            )
            for ep_local in ep_ids:
                ex_s = int(sh["exec_ep_offsets"][ep_local].item())
                ex_e = int(sh["exec_ep_offsets"][ep_local + 1].item())
                rel_steps = sh["exec_rel_step_idx"][ex_s:ex_e].tolist()
                for slot_i, rel_t in enumerate(rel_steps):
                    self.samples.append((task, int(ep_local), ex_s + slot_i, int(rel_t)))

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        out = x.to(dtype=self.feature_dtype)
        if self.project_to_2048 and out.shape[-1] == 1152:
            out = project_siglip_1152_to_2048(out, self.head_weight, self.head_bias)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, ep_local, exec_row, step_in_ep = self.samples[idx]
        sh = self.shards[task]
        s = int(sh["ep_offsets"][ep_local].item())
        e = int(sh["ep_offsets"][ep_local + 1].item())
        exec_start = int(sh["exec_start_idx"][ep_local].item())
        abs_frame = s + step_in_ep

        front_256 = self._maybe_project(sh["exec_front_256"][exec_row])
        wrist_256 = self._maybe_project(sh["exec_wrist_256"][exec_row])
        target_chunk = sh["exec_target_action_chunks"][exec_row].to(dtype=torch.float32)

        cutoff_excl = max(exec_start, max(0, step_in_ep - self.exclusion_window + 1))

        return {
            "task_name": task,
            "epis_idx": int(sh["epis_idx"][ep_local].item()),
            "ep_local": ep_local,
            "step_idx": step_in_ep,
            "exec_start_idx": exec_start,
            "readout_cutoff_excl": cutoff_excl,
            "lang_tokens": sh["lang_tokens"][ep_local].to(dtype=torch.long),
            "lang_masks": sh["lang_masks"][ep_local].to(dtype=torch.bool),
            "front_256": front_256,                                      # (256, 2048 or 1152) bf16
            "wrist_256": wrist_256,                                      # (256, 2048 or 1152) bf16
            "state_norm": sh["state_norm"][abs_frame].to(dtype=torch.float32),
            "raw_state": sh["state"][abs_frame].to(dtype=torch.float32),
            "target_actions": target_chunk,                              # (20, 8) float32
            "target_xy": sh["target_xy"][abs_frame].to(dtype=torch.float32),
            "target_xy_m11": sh["target_xy_m11"][abs_frame].to(dtype=torch.float32),
            "has_xy": sh["has_xy"][abs_frame].to(dtype=torch.bool),
            "count_state": sh["count_state"][abs_frame].to(dtype=torch.long),
            "has_count": sh["has_count"][abs_frame].to(dtype=torch.bool),
            "phase_id": sh["phase_id"][abs_frame].to(dtype=torch.long),
            "color_id": sh["color_id"][abs_frame].to(dtype=torch.long),
            "subgoal_tokens": sh["subgoal_tokens"][abs_frame].to(dtype=torch.long),
            "subgoal_masks": sh["subgoal_masks"][abs_frame].to(dtype=torch.bool),
            "gripper_event": sh["gripper_event"][abs_frame].to(dtype=torch.bool),
        }
