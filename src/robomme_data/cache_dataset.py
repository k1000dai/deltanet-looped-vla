"""Memory-mapped RoboMME SigLIP spatial feature cache loaders (episode-level & chunk-level).

Supports both official RoboMME action conventions:
- `action_mode="quantile_delta"` (`MME-VLA-Suite` / `RESEARCH_PLAN.md`: `H_act=20, H_exec=16`,
  delta arm joints `mask=(True,)*7 + (False,)` + quantile `q01`/`q99` scaling to `[-1, 1]`)
- `action_mode="minmax_abs"` (`RoboMME/DP`: `H_act=16, H_exec=8`, absolute joint angles +
  min-max scaling to `[-1, 1]`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS

from .constants import ROBOMME_TASKS, TASK_TO_SUITE


def load_cache_normalizer(
    cache_dir: str | Path | None,
    action_mode: Literal["quantile_delta", "minmax_abs"] = "quantile_delta",
) -> ActionStateNormalizer:
    """Load `ActionStateNormalizer` from `cache_dir/norm_stats.json` (falling back to `MME_VLA_NORM_STATS`)."""
    if cache_dir is not None:
        stats_file = Path(cache_dir) / "norm_stats.json"
        if stats_file.exists():
            payload = json.loads(stats_file.read_text())
            if action_mode in payload:
                return ActionStateNormalizer(mode=action_mode, stats=payload[action_mode])
            if "norm_stats" in payload:
                ns = payload["norm_stats"]
                mapped = {"state": ns["state"], "actions": ns.get("actions", ns.get("action"))}
                return ActionStateNormalizer(mode=action_mode, stats=mapped)
    if action_mode == "quantile_delta":
        return ActionStateNormalizer(mode="quantile_delta", stats=MME_VLA_NORM_STATS)
    raise FileNotFoundError(
        f"minmax_abs stats require norm_stats.json in cache_dir={cache_dir!r}"
    )


def build_padded_action_chunk(
    raw_actions_ep: np.ndarray,
    raw_state_at_step: np.ndarray,
    step_in_ep: int,
    action_horizon: int,
    normalizer: ActionStateNormalizer,
) -> np.ndarray:
    """Slice `[step_in_ep : step_in_ep + action_horizon]`, repeat final action at episode end, and normalize."""
    ep_len = raw_actions_ep.shape[0]
    if not (0 <= step_in_ep < ep_len):
        raise IndexError(f"step_in_ep={step_in_ep} out of bounds for episode length {ep_len}")

    end = min(ep_len, step_in_ep + action_horizon)
    chunk = raw_actions_ep[step_in_ep:end]
    if chunk.shape[0] < action_horizon:
        pad = np.repeat(raw_actions_ep[-1:], action_horizon - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return normalizer.encode_actions(chunk, raw_state_at_step)


def load_task_shards(
    cache_dir: str | Path,
    tasks: Sequence[str] | None = None,
    *,
    mmap: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load `{TaskName}.pt` shards from `cache_dir` using `torch.load(..., mmap=True)`."""
    root = Path(cache_dir)
    target_tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
    shards: dict[str, dict[str, Any]] = {}
    for task in target_tasks:
        shard_path = root / f"{task}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing SigLIP cache shard: {shard_path}")
        shards[task] = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=mmap)
    return shards


class RoboMMEEpisodeDataset(Dataset):
    """Episode-level sequence dataset for full-episode / segment-sequential TBPTT training.

    Each item returns an entire episode's pre-extracted multi-camera SigLIP tokens, normalized states,
    chunk boundary indices (`chunk_step_indices`), normalized target action chunks `(M, H_act, 8)`,
    and A3 exclusion-window readout cutoffs (`chunk_step_indices - exclusion_window`).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        tasks: Sequence[str] | None = None,
        episode_indices_per_task: Sequence[int] | None = None,
        action_mode: Literal["quantile_delta", "minmax_abs"] = "quantile_delta",
        action_horizon: int | None = None,
        exec_horizon: int | None = None,
        exclusion_window: int | None = None,
        random_chunk_phase: bool = False,
        feature_dtype: torch.dtype = torch.bfloat16,
        mmap: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
        self.action_mode = action_mode
        self.action_horizon = (
            (20 if action_mode == "quantile_delta" else 16)
            if action_horizon is None
            else int(action_horizon)
        )
        self.exec_horizon = (
            (16 if action_mode == "quantile_delta" else 8)
            if exec_horizon is None
            else int(exec_horizon)
        )
        self.exclusion_window = (
            self.exec_horizon if exclusion_window is None else int(exclusion_window)
        )
        self.random_chunk_phase = random_chunk_phase
        self.feature_dtype = feature_dtype
        self.rng = np.random.default_rng(seed)

        self.normalizer = load_cache_normalizer(self.cache_dir, action_mode=action_mode)
        self.shards = load_task_shards(self.cache_dir, tasks=self.tasks, mmap=mmap)

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, ep_local = self.index[idx]
        sh = self.shards[task]

        s = int(sh["ep_offsets"][ep_local].item())
        e = int(sh["ep_offsets"][ep_local + 1].item())
        ep_len = e - s
        exec_start = int(sh["exec_start_idx"][ep_local].item())
        epis_idx = int(sh["epis_idx"][ep_local].item())

        front_emb = sh["front_emb"][s:e].to(dtype=self.feature_dtype)
        wrist_emb = sh["wrist_emb"][s:e].to(dtype=self.feature_dtype)
        raw_state_t = sh["state"][s:e].to(dtype=torch.float32)
        raw_actions_t = sh["actions"][s:e].to(dtype=torch.float32)
        is_demo = sh["is_demo"][s:e].to(dtype=torch.bool)

        raw_state_np = raw_state_t.numpy()
        raw_actions_np = raw_actions_t.numpy()
        state_norm = torch.from_numpy(self.normalizer.normalize_state(raw_state_np))

        exec_len = max(1, ep_len - exec_start)
        phase = 0
        if self.random_chunk_phase and exec_len > 1:
            max_phase = min(self.exec_horizon, exec_len) - 1
            if max_phase > 0:
                phase = int(self.rng.integers(0, max_phase + 1))

        chunk_steps = np.arange(exec_start + phase, ep_len, self.exec_horizon, dtype=np.int64)
        if chunk_steps.size == 0:
            chunk_steps = np.array([min(exec_start, ep_len - 1)], dtype=np.int64)

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
        # Exclusive upper bound index `cutoff_excl` for history stream `[0 : cutoff_excl]` under A3:
        # at step `t_step`, frames `[0 .. t_step - exclusion_window]` are eligible (`cutoff_excl = max(0, t_step - excl + 1)`).
        # Note: all demo frames `[0 .. exec_start - 1]` are always eligible before execution starts.
        raw_cutoff = chunk_steps - self.exclusion_window + 1
        cutoff_excl = np.maximum(exec_start, np.maximum(0, raw_cutoff))

        simple_sgs_ep = sh["simple_subgoals"][ep_local]
        grounded_sgs_ep = sh["grounded_subgoals"][ep_local]

        return {
            "task_name": task,
            "suite": sh.get("suite", TASK_TO_SUITE.get(task, "Unknown")),
            "epis_idx": epis_idx,
            "exec_start_idx": exec_start,
            "total_steps": ep_len,
            "prompt": sh["prompts"][ep_local],
            "front_emb": front_emb,                                      # (T, 64, 768)
            "wrist_emb": wrist_emb,                                      # (T, 16, 768)
            "state_norm": state_norm,                                    # (T, 8) float32
            "raw_state": raw_state_t,                                    # (T, 8) float32
            "raw_actions": raw_actions_t,                                # (T, 8) float32
            "is_demo": is_demo,                                          # (T,) bool
            "text_token_emb": sh["text_token_emb"][ep_local].to(self.feature_dtype),  # (64, 768)
            "text_pool_emb": sh["text_pool_emb"][ep_local].to(self.feature_dtype),    # (768,)
            "text_attn_mask": sh["text_attn_mask"][ep_local].to(torch.bool),          # (64,)
            "chunk_step_indices": torch.from_numpy(chunk_steps),         # (M,) int64
            "readout_cutoff_excl": torch.from_numpy(cutoff_excl),        # (M,) int64
            "target_action_chunks": torch.from_numpy(target_chunks),     # (M, H_act, 8) float32
            "simple_subgoals_at_chunks": [simple_sgs_ep[int(t)] for t in chunk_steps],
            "grounded_subgoals_at_chunks": [grounded_sgs_ep[int(t)] for t in chunk_steps],
        }


class RoboMMEChunkDataset(Dataset):
    """Step/chunk-indexed dataset over execution frames with history windowing & A3 exclusion support."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        tasks: Sequence[str] | None = None,
        episode_indices_per_task: Sequence[int] | None = None,
        action_mode: Literal["quantile_delta", "minmax_abs"] = "quantile_delta",
        action_horizon: int | None = None,
        exec_horizon: int | None = None,
        sample_stride: int = 1,
        history_stride: int = 4,
        max_history_frames: int = 64,
        exclusion_window: int | None = None,
        feature_dtype: torch.dtype = torch.bfloat16,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
        self.action_mode = action_mode
        self.action_horizon = (
            (20 if action_mode == "quantile_delta" else 16)
            if action_horizon is None
            else int(action_horizon)
        )
        self.exec_horizon = (
            (16 if action_mode == "quantile_delta" else 8)
            if exec_horizon is None
            else int(exec_horizon)
        )
        self.sample_stride = max(1, int(sample_stride))
        self.history_stride = max(1, int(history_stride))
        self.max_history_frames = max(1, int(max_history_frames))
        self.exclusion_window = (
            self.exec_horizon if exclusion_window is None else int(exclusion_window)
        )
        self.feature_dtype = feature_dtype

        self.normalizer = load_cache_normalizer(self.cache_dir, action_mode=action_mode)
        self.shards = load_task_shards(self.cache_dir, tasks=self.tasks, mmap=mmap)

        # Flat index of (task, ep_local, step_in_ep)
        self.samples: list[tuple[str, int, int]] = []
        for task in self.tasks:
            sh = self.shards[task]
            n_eps = int(sh["epis_idx"].shape[0])
            ep_ids = (
                range(n_eps)
                if episode_indices_per_task is None
                else [i for i in episode_indices_per_task if 0 <= i < n_eps]
            )
            for ep_local in ep_ids:
                s = int(sh["ep_offsets"][ep_local].item())
                e = int(sh["ep_offsets"][ep_local + 1].item())
                ep_len = e - s
                exec_start = int(sh["exec_start_idx"][ep_local].item())
                for step_in_ep in range(exec_start, ep_len, self.sample_stride):
                    self.samples.append((task, int(ep_local), int(step_in_ep)))

    def __len__(self) -> int:
        return len(self.samples)

    def _select_history_indices(self, step_in_ep: int, exec_start: int) -> np.ndarray:
        """Select up to `max_history_frames` eligible history frames respecting A3 exclusion window."""
        cutoff_excl = max(exec_start, max(0, step_in_ep - self.exclusion_window + 1))
        if cutoff_excl <= 0:
            return np.zeros((0,), dtype=np.int64)

        cand = np.arange(0, cutoff_excl, self.history_stride, dtype=np.int64)
        if cand.size > self.max_history_frames:
            # Preserve evenly spaced demo frames if present, plus most recent execution history
            if exec_start > 0:
                demo_budget = min(self.max_history_frames // 2, max(1, exec_start // self.history_stride))
                demo_idx = np.linspace(0, exec_start - 1, demo_budget, dtype=np.int64)
                exec_cand = cand[cand >= exec_start]
                rem_budget = self.max_history_frames - demo_idx.size
                cand = np.concatenate([demo_idx, exec_cand[-rem_budget:]], axis=0)
            else:
                cand = cand[-self.max_history_frames :]
        return cand

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, ep_local, step_in_ep = self.samples[idx]
        sh = self.shards[task]
        s = int(sh["ep_offsets"][ep_local].item())
        e = int(sh["ep_offsets"][ep_local + 1].item())

        exec_start = int(sh["exec_start_idx"][ep_local].item())
        abs_row = s + step_in_ep

        curr_front = sh["front_emb"][abs_row].to(self.feature_dtype)     # (64, 768)
        curr_wrist = sh["wrist_emb"][abs_row].to(self.feature_dtype)     # (16, 768)

        raw_state_ep = sh["state"][s:e].numpy()
        raw_actions_ep = sh["actions"][s:e].numpy()
        curr_raw_state = raw_state_ep[step_in_ep]
        curr_state_norm = torch.from_numpy(self.normalizer.normalize_state(curr_raw_state))

        target_chunk = torch.from_numpy(
            build_padded_action_chunk(
                raw_actions_ep,
                curr_raw_state,
                step_in_ep,
                self.action_horizon,
                self.normalizer,
            )
        )

        hist_rel = self._select_history_indices(step_in_ep, exec_start)
        h_len = hist_rel.size
        hist_front = torch.zeros((self.max_history_frames, 64, 768), dtype=self.feature_dtype)
        hist_wrist = torch.zeros((self.max_history_frames, 16, 768), dtype=self.feature_dtype)
        hist_state = torch.zeros((self.max_history_frames, 8), dtype=torch.float32)
        hist_step_idx = torch.zeros((self.max_history_frames,), dtype=torch.int64)
        hist_is_demo = torch.zeros((self.max_history_frames,), dtype=torch.bool)
        hist_mask = torch.zeros((self.max_history_frames,), dtype=torch.bool)

        if h_len > 0:
            abs_hist = torch.from_numpy(s + hist_rel)
            # Left-align or right-align: we right-pad (`[:h_len]`) in chronological order `[0 .. h_len-1]`
            hist_front[:h_len] = sh["front_emb"][abs_hist].to(self.feature_dtype)
            hist_wrist[:h_len] = sh["wrist_emb"][abs_hist].to(self.feature_dtype)
            hist_state[:h_len] = torch.from_numpy(
                self.normalizer.normalize_state(raw_state_ep[hist_rel])
            )
            hist_step_idx[:h_len] = torch.from_numpy(hist_rel)
            hist_is_demo[:h_len] = torch.from_numpy(hist_rel < exec_start)
            hist_mask[:h_len] = True

        return {
            "task_name": task,
            "epis_idx": int(sh["epis_idx"][ep_local].item()),
            "step_idx": step_in_ep,
            "exec_start_idx": exec_start,
            "front_emb": curr_front,                                                   # (64, 768)
            "wrist_emb": curr_wrist,                                                   # (16, 768)
            "state_norm": curr_state_norm,                                             # (8,)
            "raw_state": torch.from_numpy(curr_raw_state.copy()),                      # (8,)
            "target_actions": target_chunk,                                            # (H_act, 8)
            "text_token_emb": sh["text_token_emb"][ep_local].to(self.feature_dtype),   # (64, 768)
            "text_pool_emb": sh["text_pool_emb"][ep_local].to(self.feature_dtype),     # (768,)
            "text_attn_mask": sh["text_attn_mask"][ep_local].to(torch.bool),           # (64,)
            "hist_front_emb": hist_front,                                              # (H_mem, 64, 768)
            "hist_wrist_emb": hist_wrist,                                              # (H_mem, 16, 768)
            "hist_state_norm": hist_state,                                             # (H_mem, 8)
            "hist_step_idx": hist_step_idx,                                            # (H_mem,)
            "hist_is_demo": hist_is_demo,                                              # (H_mem,)
            "hist_mask": hist_mask,                                                    # (H_mem,)
            "simple_subgoal": sh["simple_subgoals"][ep_local][step_in_ep],
        }
