"""Dataset Wrapper & Realistic Cross-Session Distractor Builder (`src/pi05_mem/data.py`).

Wraps `src.pi05_data.dataset.Pi05EpisodeDataset` (`mmap=True`, `< 0.5s` load) and adds:
  1. **True-Phase Cross-Session Distractor Prefixing (`p_distractor = 0.25`, `k in {1..3}`):**
     Preserves each distractor episode's TRUE `is_demo` (`False` on distractor execution steps!)
     and populates `prompt_cos_jump` (`1.0` at cross-task/cross-prompt episode transitions) so
     `EpisodeBoundaryForgetGate` trains on the exact sensor/instruction transitions seen in live
     `eval_pi05.py --interference-k 4` closed-loop rollouts!
  2. **`SyntheticPi05EpisodeDataset`:**
     Deterministic in-memory synthetic dataset matching `Pi05EpisodeDataset` tensor keys & shapes
     for instant CPU/MIG self-tests (`0` disk dependency).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from analysis.subgoal_grounding import resolve_episode_subgoals
from pi05_data.dataset import DEFAULT_MIYABI_PI05_CACHE_DIR, Pi05EpisodeDataset
from robomme_data.cache_dataset import build_padded_action_chunk
from robomme_data.constants import ROBOMME_TASKS, TASK_TO_SUITE

from .config import Pi05MemConfig

_AT_COORD_RE = re.compile(r"at\s*<\s*(\d+)\s*,\s*(\d+)\s*>")


def _jitter_subgoal_coords(sg: str, max_px: int, rng: np.random.Generator) -> str:
    if not sg or max_px <= 0:
        return sg

    def _repl(m: re.Match[str]) -> str:
        ry = int(m.group(1))
        cx = int(m.group(2))
        std = max_px / 2.5
        dy = int(np.clip(round(float(rng.normal(0.0, std))), -max_px, max_px))
        dx = int(np.clip(round(float(rng.normal(0.0, std))), -max_px, max_px))
        ny = int(np.clip(ry + dy, 0, 255))
        nx = int(np.clip(cx + dx, 0, 255))
        return f"at <{ny}, {nx}>"

    return _AT_COORD_RE.sub(_repl, sg)


def collate_episode_list(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Variable-length episode list collator (keeps episodes unpadded for exact `O(W)` GDN scan)."""
    return batch


class RealisticCrossSessionEpisodeDataset(Dataset):
    """Wraps `Pi05EpisodeDataset` with realistic `k in {1..3}` un-reset distractor episode chains.

    Systems optimizations for Miyabi GH200 (`interact-g` `< 100 GB` host RAM cgroup):
      - `defer_siglip_head_to_device=True` keeps `1152-D` `bf16` tokens unprojected on CPU (`0` CPU
        `bf16` GEMM overhead, `44%` smaller IPC/NVLink transfers); `DeltaMemPi05Policy` projects
        `1152 -> 2048` on GPU via frozen `vision_tower.head` in `0.02 ms` (`0.0` bit-diff!).
      - `max_chunks_per_episode` slices ONLY the sampled `<= M` execution rows from `exec_front_256` /
        `exec_wrist_256` (`4x-10x` less Lustre `mmap` I/O) and `.clone()`s compact episode tensors so
        multiprocess DataLoader IPC never retains whole-file `UntypedStorage` handles.
    """

    _CHUNK_KEYS = (
        "chunk_step_indices",
        "readout_cutoff_excl",
        "target_action_chunks",
        "chunk_target_xy",
        "chunk_target_xy_m11",
        "chunk_has_xy",
        "chunk_secondary_xy",
        "chunk_has_secondary_xy",
        "chunk_count_state",
        "chunk_has_count",
        "chunk_phase_id",
        "chunk_color_id",
        "chunk_ordinal_id",
        "chunk_subgoal_tokens",
        "chunk_subgoal_masks",
    )

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_MIYABI_PI05_CACHE_DIR,
        *,
        tasks: Sequence[str] | None = None,
        episode_indices_per_task: Sequence[int] | None = None,
        stats_variant: Literal["mme_vla", "pi05_baseline", "framesamp_modul"] = "pi05_baseline",
        action_horizon: int = 20,
        exec_horizon: int = 16,
        exclusion_window: int = 16,
        random_chunk_phase: bool = True,
        max_chunks_per_episode: int | None = None,
        defer_siglip_head_to_device: bool = True,
        distractor_prob: float = 0.25,
        max_distractors: int = 3,
        max_distractor_frames_each: int = 180,
        pred_subgoals_by_task: Optional[Dict[str, List[Dict[int, str]]]] = None,
        vlm_max_token_len: int = 64,
        subgoal_dg_prob: float = 0.0,
        subgoal_oracle_prob: float = 0.0,
        subgoal_jitter_px: int = 6,
        cont_vlm_cache_dir: Optional[str | Path] = None,
        reactive_precision_sampling: bool = False,
        reactive_stopcube_excl: int = 1,
        tokenizer_path: Optional[str | Path] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.defer_siglip_head_to_device = bool(defer_siglip_head_to_device)
        self.max_chunks_per_episode = (
            None if max_chunks_per_episode is None else max(1, int(max_chunks_per_episode))
        )
        self.pred_subgoals_by_task = pred_subgoals_by_task
        self.vlm_max_token_len = int(vlm_max_token_len)
        self.subgoal_dg_prob = float(subgoal_dg_prob)
        self.subgoal_oracle_prob = float(subgoal_oracle_prob)
        self.subgoal_jitter_px = int(subgoal_jitter_px)
        self.reactive_precision_sampling = bool(reactive_precision_sampling)
        self.reactive_stopcube_excl = max(0, int(reactive_stopcube_excl))
        self.tokenizer_path = tokenizer_path
        self._vlm_tokenizer: Any = None
        self.use_vlm_subgoals = (self.subgoal_dg_prob > 0.0) or (self.subgoal_oracle_prob > 0.0)
        self.cont_vlm_shards: Dict[str, Any] = {}
        if cont_vlm_cache_dir is not None and Path(cont_vlm_cache_dir).exists():
            cv_root = Path(cont_vlm_cache_dir)
            req_tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
            for t_cv in req_tasks:
                p_cv = cv_root / f"{t_cv}.pt"
                if p_cv.exists():
                    self.cont_vlm_shards[t_cv] = torch.load(p_cv, map_location="cpu", mmap=True, weights_only=False)
        self.base = Pi05EpisodeDataset(
            cache_dir=cache_dir,
            tasks=tasks,
            episode_indices_per_task=episode_indices_per_task,
            stats_variant=stats_variant,
            action_horizon=action_horizon,
            exec_horizon=exec_horizon,
            exclusion_window=exclusion_window,
            random_chunk_phase=random_chunk_phase,
            project_history_to_2048=not self.defer_siglip_head_to_device,
            project_exec_256_to_2048=not self.defer_siglip_head_to_device,
            load_exec_256=False,  # Sliced selectively below after `max_chunks_per_episode` subsampling!
            distractor_prob=0.0,  # Handled here with true `is_demo` & `prompt_cos_jump`!
            mmap=True,
            seed=seed,
        )
        self.distractor_prob = float(distractor_prob)
        self.max_distractors = int(max_distractors)
        self.max_distractor_frames_each = int(max_distractor_frames_each)
        self.rng = np.random.default_rng(seed + 997)

        # Pre-pin all tiny (<95 MB total across 16 shards!) scalar/state/action/aux metadata in RAM
        # so `__getitem__` performs ONLY 4 contiguous Lustre reads instead of 24 random Lustre seeks!
        self._meta: Dict[str, Dict[str, Any]] = {}
        for t_name, sh in self.base.shards.items():
            raw_st_all = sh["state"].to(dtype=torch.float32).contiguous().clone().numpy()
            raw_act_all = sh["actions"].to(dtype=torch.float32).contiguous().clone().numpy()
            st_norm_all = torch.from_numpy(self.base.normalizer.normalize_state(raw_st_all))
            self._meta[t_name] = {
                "suite": sh.get("suite", TASK_TO_SUITE.get(t_name, "Unknown")),
                "ep_offsets": sh["ep_offsets"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_start_idx": sh["exec_start_idx"].to(dtype=torch.long).contiguous().clone().numpy(),
                "epis_idx": sh["epis_idx"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_ep_offsets": sh["exec_ep_offsets"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_rel_step_idx": sh["exec_rel_step_idx"].to(dtype=torch.long).contiguous().clone().numpy(),
                "prompts": list(sh["prompts"]),
                "lang_tokens": sh["lang_tokens"].to(dtype=torch.long).contiguous().clone(),
                "lang_masks": sh["lang_masks"].to(dtype=torch.bool).contiguous().clone(),
                "raw_state_np": raw_st_all,
                "raw_actions_np": raw_act_all,
                "state_norm": st_norm_all,
                "is_demo": sh["is_demo"].to(dtype=torch.bool).contiguous().clone(),
                "target_xy_m11": sh["target_xy_m11"].to(dtype=torch.float32).contiguous().clone(),
                "has_xy": sh["has_xy"].to(dtype=torch.bool).contiguous().clone(),
                "secondary_xy": sh["secondary_xy"].to(dtype=torch.float32).contiguous().clone(),
                "has_secondary_xy": sh["has_secondary_xy"].to(dtype=torch.bool).contiguous().clone(),
                "count_state": sh["count_state"].to(dtype=torch.long).contiguous().clone(),
                "has_count": sh["has_count"].to(dtype=torch.bool).contiguous().clone(),
                "phase_id": sh["phase_id"].to(dtype=torch.long).contiguous().clone(),
                "color_id": sh["color_id"].to(dtype=torch.long).contiguous().clone(),
                "ordinal_id": sh["ordinal_id"].to(dtype=torch.long).contiguous().clone(),
                "gs_on": (
                    [resolve_episode_subgoals(seq) for seq in (sh.get("grounded_subgoals") or sh["grounded_subgoals_online"])]
                    if self.use_vlm_subgoals
                    else None
                ),
            }

    def _get_vlm_tokenizer(self) -> Any:
        if self._vlm_tokenizer is None:
            from pi05.policy import Pi05Tokenizer

            self._vlm_tokenizer = Pi05Tokenizer(
                max_len=self.vlm_max_token_len,
                discrete_state_input=False,
                tokenizer_path=self.tokenizer_path,
            )
        return self._vlm_tokenizer

    def __len__(self) -> int:
        return len(self.base)

    def _slice_true_distractor(self, task: str, ep_local: int) -> Dict[str, Any]:
        sh = self.base.shards[task]
        mt = self._meta[task]
        s = int(mt["ep_offsets"][ep_local])
        e = int(mt["ep_offsets"][ep_local + 1])
        if (e - s) > self.max_distractor_frames_each:
            # Keep the tail of the distractor episode so it ends in realistic execution (`is_demo=False`)!
            s = e - self.max_distractor_frames_each
        front = self.base._maybe_project(sh["front_emb"][s:e], self.base.project_history_to_2048).contiguous().clone()
        wrist = self.base._maybe_project(sh["wrist_emb"][s:e], self.base.project_history_to_2048).contiguous().clone()
        return {
            "front": front,
            "wrist": wrist,
            "state_norm": mt["state_norm"][s:e].clone(),
            "is_demo": mt["is_demo"][s:e].clone(),
            "prompt": str(mt["prompts"][ep_local]),
            "length": int(e - s),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task, ep_local = self.base.index[idx]
        sh = self.base.shards[task]
        mt = self._meta[task]

        s = int(mt["ep_offsets"][ep_local])
        e = int(mt["ep_offsets"][ep_local + 1])
        ep_len = e - s
        exec_start = int(mt["exec_start_idx"][ep_local])
        epis_idx = int(mt["epis_idx"][ep_local])

        exec_s = int(mt["exec_ep_offsets"][ep_local])
        exec_e = int(mt["exec_ep_offsets"][ep_local + 1])
        cached_rel_steps = mt["exec_rel_step_idx"][exec_s:exec_e]
        raw_state_ep = mt["raw_state_np"][s:e]
        raw_actions_ep = mt["raw_actions_np"][s:e]
        prompt_str = str(mt["prompts"][ep_local])

        is_reac_stop = False
        is_prec_peg = False
        if self.reactive_precision_sampling:
            from .deltaground import should_use_precision_insertpeg, should_use_reactive_stopcube

            is_reac_stop = should_use_reactive_stopcube(prompt_str)
            is_prec_peg = should_use_precision_insertpeg(prompt_str)

        n_avail = int(cached_rel_steps.size)
        k_want = self.max_chunks_per_episode or 8
        if (is_reac_stop or is_prec_peg) and n_avail > 1:
            exec_steps_arr = cached_rel_steps.astype(np.int64)
            if is_reac_stop:
                # Detect exact pass-N button-strike onset where demonstrator arm breaks static wait (`||q_des - q||_2 > 0.018`)!
                arm_delta = np.linalg.norm(
                    raw_actions_ep[exec_steps_arr, :7] - raw_state_ep[exec_steps_arr, :7], axis=-1
                )
                active_hits = np.flatnonzero(arm_delta > 0.018)
                onset_slot = int(active_hits[0]) if active_hits.size > 0 else int(np.argmax(arm_delta))
                trig_lo = max(0, onset_slot - 4)
                trig_hi = min(n_avail - 1, onset_slot + 18)
                trig_pool = np.arange(trig_lo, trig_hi + 1, dtype=np.int64)
                hard_neg_pool = np.arange(max(0, onset_slot - 52), max(0, onset_slot - 5), dtype=np.int64)
                early_pool = np.arange(0, max(1, onset_slot - 5), dtype=np.int64)

                n_trig = max(1, int(round(k_want * 0.50)))
                n_hneg = max(1, int(round(k_want * 0.25)))
                n_early = max(1, k_want - n_trig - n_hneg)

                def _draw_pool(pool_arr: np.ndarray, cnt: int) -> np.ndarray:
                    if pool_arr.size == 0 or cnt <= 0:
                        return np.empty((0,), dtype=np.int64)
                    return self.rng.choice(pool_arr, size=min(cnt, pool_arr.size), replace=False)

                cand_p = np.unique(
                    np.concatenate(
                        [
                            _draw_pool(trig_pool, n_trig),
                            _draw_pool(hard_neg_pool, n_hneg),
                            _draw_pool(early_pool, n_early),
                        ]
                    )
                )
            else:
                # `InsertPeg`: 3x oversample (a) peg-end grasp contact (`|Delta g| > 0.12` +- 14f) & (b) box-hole alignment/insertion!
                g_act = raw_actions_ep[exec_steps_arr, 7]
                g_jump = np.abs(np.diff(g_act, prepend=g_act[0]))
                g_hits = np.flatnonzero(g_jump > 0.10)
                grasp_center = int(g_hits[0]) if g_hits.size > 0 else max(0, n_avail // 4)
                grasp_pool = np.arange(max(0, grasp_center - 14), min(n_avail, grasp_center + 16), dtype=np.int64)
                insert_pool = np.arange(int(n_avail * 0.52), n_avail, dtype=np.int64)
                all_pool = np.arange(0, n_avail, dtype=np.int64)

                n_gr = max(1, int(round(k_want * 0.35)))
                n_in = max(1, int(round(k_want * 0.40)))
                n_ap = max(1, k_want - n_gr - n_in)

                def _draw_p2(pool_arr: np.ndarray, cnt: int) -> np.ndarray:
                    if pool_arr.size == 0 or cnt <= 0:
                        return np.empty((0,), dtype=np.int64)
                    return self.rng.choice(pool_arr, size=min(cnt, pool_arr.size), replace=False)

                cand_p = np.unique(
                    np.concatenate(
                        [
                            _draw_p2(grasp_pool, n_gr),
                            _draw_p2(insert_pool, n_in),
                            _draw_p2(all_pool, n_ap),
                        ]
                    )
                )
            if cand_p.size < min(k_want, n_avail):
                rem_all = np.setdiff1d(np.arange(n_avail, dtype=np.int64), cand_p, assume_unique=True)
                extra = self.rng.choice(rem_all, size=min(k_want - cand_p.size, rem_all.size), replace=False)
                cand_p = np.unique(np.concatenate([cand_p, extra]))
            if cand_p.size > k_want:
                cand_p = np.sort(self.rng.choice(cand_p, size=k_want, replace=False))
            picked_slots = np.sort(cand_p) if cand_p.size > 0 else np.array([0], dtype=np.int64)
        else:
            grid_step_per_chunk = max(1, self.base.exec_horizon // self.base.exec_stride)
            phase_grid = 0
            if self.base.random_chunk_phase and cached_rel_steps.size > 1 and grid_step_per_chunk > 1:
                max_p = min(grid_step_per_chunk, cached_rel_steps.size) - 1
                if max_p > 0:
                    phase_grid = int(self.rng.integers(0, max_p + 1))

            picked_slots = np.arange(phase_grid, cached_rel_steps.size, grid_step_per_chunk, dtype=np.int64)
            if picked_slots.size == 0:
                picked_slots = np.array([0], dtype=np.int64)

            if self.max_chunks_per_episode is not None and picked_slots.size > self.max_chunks_per_episode:
                sel_np = np.sort(self.rng.choice(picked_slots.size, size=self.max_chunks_per_episode, replace=False))
                picked_slots = picked_slots[sel_np]

        chunk_steps = cached_rel_steps[picked_slots]
        target_chunks = np.stack(
            [
                build_padded_action_chunk(
                    raw_actions_ep,
                    raw_state_ep[int(t_step)],
                    int(t_step),
                    self.base.action_horizon,
                    self.base.normalizer,
                )
                for t_step in chunk_steps
            ],
            axis=0,
        )
        eff_excl = self.reactive_stopcube_excl if is_reac_stop else self.base.exclusion_window
        raw_cutoff = chunk_steps - eff_excl + 1
        cutoff_excl = np.maximum(exec_start, np.maximum(0, raw_cutoff))

        # ONLY 4 Lustre mmap reads per episode (instead of 24)!
        front_emb = self.base._maybe_project(sh["front_emb"][s:e], self.base.project_history_to_2048).contiguous().clone()
        wrist_emb = self.base._maybe_project(sh["wrist_emb"][s:e], self.base.project_history_to_2048).contiguous().clone()
        abs_exec_rows = torch.from_numpy(exec_s + picked_slots).long()
        exec_front_256 = (
            self.base._maybe_project(sh["exec_front_256"][abs_exec_rows], self.base.project_exec_256_to_2048)
            .contiguous()
            .clone()
        )
        exec_wrist_256 = (
            self.base._maybe_project(sh["exec_wrist_256"][abs_exec_rows], self.base.project_exec_256_to_2048)
            .contiguous()
            .clone()
        )

        abs_chunk_rows = torch.from_numpy(s + chunk_steps).long()
        ep: Dict[str, Any] = {
            "task_name": task,
            "suite": mt["suite"],
            "epis_idx": epis_idx,
            "exec_start_idx": exec_start,
            "distractor_prefix_frames": 0,
            "total_steps": ep_len,
            "prompt": mt["prompts"][ep_local],
            "lang_tokens": mt["lang_tokens"][ep_local].clone(),
            "lang_masks": mt["lang_masks"][ep_local].clone(),
            "front_emb": front_emb,
            "wrist_emb": wrist_emb,
            "state_norm": mt["state_norm"][s:e].clone(),
            "is_demo": mt["is_demo"][s:e].clone(),
            "chunk_step_indices": torch.from_numpy(chunk_steps).long(),
            "readout_cutoff_excl": torch.from_numpy(cutoff_excl).long(),
            "target_action_chunks": torch.from_numpy(target_chunks),
            "exec_front_256": exec_front_256,
            "exec_wrist_256": exec_wrist_256,
            "chunk_target_xy_m11": mt["target_xy_m11"][abs_chunk_rows].clone(),
            "chunk_has_xy": mt["has_xy"][abs_chunk_rows].clone(),
            "chunk_secondary_xy": mt["secondary_xy"][abs_chunk_rows].clone(),
            "chunk_has_secondary_xy": mt["has_secondary_xy"][abs_chunk_rows].clone(),
            "chunk_count_state": mt["count_state"][abs_chunk_rows].clone(),
            "chunk_has_count": mt["has_count"][abs_chunk_rows].clone(),
            "chunk_phase_id": mt["phase_id"][abs_chunk_rows].clone(),
            "chunk_color_id": mt["color_id"][abs_chunk_rows].clone(),
            "chunk_ordinal_id": mt["ordinal_id"][abs_chunk_rows].clone(),
        }
        if is_reac_stop:
            ep["write_stride_exec_override"] = 1
        if is_prec_peg:
            ep["use_hop2_l2_override"] = True
            ep["precision_insertpeg_climax"] = True
        if task in self.cont_vlm_shards:
            cv_sh = self.cont_vlm_shards[task]
            ep["chunk_cont_vlm_ground"] = {
                "front_vlm_8x8": cv_sh["front_vlm_8x8"][abs_exec_rows].contiguous().clone(),
                "hm_maps": cv_sh["hm_maps"][abs_exec_rows].to(torch.float32).contiguous().clone(),
                "roi_vecs": cv_sh["roi_vecs"][abs_exec_rows].contiguous().clone(),
                "sem_probs": cv_sh["sem_probs"][abs_exec_rows].to(torch.float32).contiguous().clone(),
                "xy_coords": cv_sh["xy_coords"][abs_exec_rows].to(torch.float32).contiguous().clone(),
                "active_mask": torch.ones((int(abs_exec_rows.numel()),), dtype=torch.bool),
            }

        if self.use_vlm_subgoals:
            tok = self._get_vlm_tokenizer()
            raw_prompt = str(mt["prompts"][ep_local])
            dg_ep_map: Dict[int, str] = {}
            if self.pred_subgoals_by_task is not None and task in self.pred_subgoals_by_task:
                t_list = self.pred_subgoals_by_task[task]
                if 0 <= ep_local < len(t_list):
                    dg_ep_map = t_list[ep_local]
            gs_on_ep = mt["gs_on"][ep_local] if mt.get("gs_on") is not None else []

            vlm_tok_rows: List[np.ndarray] = []
            vlm_mask_rows: List[np.ndarray] = []
            for t_step in chunk_steps.tolist():
                u_draw = float(self.rng.random())
                sg_use: Optional[str] = None
                if u_draw < self.subgoal_dg_prob and dg_ep_map:
                    t_int = int(t_step)
                    if t_int in dg_ep_map:
                        sg_use = dg_ep_map[t_int]
                    else:
                        le_keys = [k for k in dg_ep_map.keys() if k <= t_int]
                        if le_keys:
                            sg_use = dg_ep_map[max(le_keys)]
                elif u_draw < (self.subgoal_dg_prob + self.subgoal_oracle_prob) and gs_on_ep:
                    raw_oracle = gs_on_ep[min(int(t_step), len(gs_on_ep) - 1)]
                    sg_use = _jitter_subgoal_coords(raw_oracle, self.subgoal_jitter_px, self.rng)
                if sg_use is not None and not sg_use.strip():
                    sg_use = None
                t_np, m_np = tok.tokenize(raw_prompt, norm_state=None, subgoal=sg_use)
                vlm_tok_rows.append(t_np)
                vlm_mask_rows.append(m_np)
            ep["chunk_vlm_tokens"] = torch.from_numpy(np.stack(vlm_tok_rows, axis=0)).long()
            ep["chunk_vlm_masks"] = torch.from_numpy(np.stack(vlm_mask_rows, axis=0)).bool()

        t_ep = int(ep["front_emb"].shape[0])
        prompt_jump = torch.zeros((t_ep,), dtype=torch.float32)

        if (
            self.distractor_prob <= 0.0
            or self.max_distractors < 1
            or len(self.base.index) <= 1
            or float(self.rng.random()) >= self.distractor_prob
        ):
            ep["prompt_cos_jump"] = prompt_jump
            return ep

        k_dist = int(self.rng.integers(1, self.max_distractors + 1))
        fronts: List[torch.Tensor] = []
        wrists: List[torch.Tensor] = []
        states: List[torch.Tensor] = []
        demos: List[torch.Tensor] = []
        jumps: List[torch.Tensor] = []
        prev_prompt: Optional[str] = None
        prefix_offset = 0

        for _ in range(k_dist):
            d_i = int(self.rng.integers(0, len(self.base.index)))
            d_task, d_ep = self.base.index[d_i]
            d_seg = self._slice_true_distractor(d_task, d_ep)
            d_len = d_seg["length"]
            if d_len <= 0:
                continue
            j_vec = torch.zeros((d_len,), dtype=torch.float32)
            if prev_prompt is not None and d_seg["prompt"] != prev_prompt:
                j_vec[0] = 1.0
            prev_prompt = d_seg["prompt"]
            fronts.append(d_seg["front"])
            wrists.append(d_seg["wrist"])
            states.append(d_seg["state_norm"])
            demos.append(d_seg["is_demo"])
            jumps.append(j_vec)
            prefix_offset += d_len

        if prefix_offset > 0:
            if prev_prompt is not None and str(ep["prompt"]) != prev_prompt:
                prompt_jump[0] = 1.0
            ep["front_emb"] = torch.cat([*fronts, ep["front_emb"]], dim=0)
            ep["wrist_emb"] = torch.cat([*wrists, ep["wrist_emb"]], dim=0)
            ep["state_norm"] = torch.cat([*states, ep["state_norm"]], dim=0)
            ep["is_demo"] = torch.cat([*demos, ep["is_demo"]], dim=0)
            ep["prompt_cos_jump"] = torch.cat([*jumps, prompt_jump], dim=0)
            ep["chunk_step_indices"] = ep["chunk_step_indices"] + prefix_offset
            ep["readout_cutoff_excl"] = ep["readout_cutoff_excl"] + prefix_offset
            ep["exec_start_idx"] = int(ep["exec_start_idx"]) + prefix_offset
            ep["distractor_prefix_frames"] = prefix_offset
            ep["total_steps"] = int(ep["front_emb"].shape[0])
        else:
            ep["prompt_cos_jump"] = prompt_jump
        return ep


class SyntheticPi05EpisodeDataset(Dataset):
    """Self-contained synthetic dataset exercising video-demo, non-demo, & `k=3` boundary episodes."""

    def __init__(self, cfg: Pi05MemConfig, num_episodes: int = 8, seed: int = 123) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_episodes = int(num_episodes)
        g = torch.Generator().manual_seed(seed)
        self.episodes: List[Dict[str, Any]] = []

        for i in range(self.num_episodes):
            has_demo = (i % 2) == 0
            demo_len = 24 if has_demo else 0
            exec_len = 48
            has_distractor = (i % 3) == 0
            dist_len = 20 if has_distractor else 0

            tot = dist_len + demo_len + exec_len
            exec_start = dist_len + demo_len

            # Temporally smooth scene features (`cos_sim ~ 0.99` within session, true scene cut at `dist_len`!)
            scene_main_f = torch.randn((1, cfg.front_tokens_hist, cfg.vis_dim), generator=g)
            scene_main_w = torch.randn((1, cfg.wrist_tokens_hist, cfg.vis_dim), generator=g)
            front_hist = scene_main_f + 0.12 * torch.randn((tot, cfg.front_tokens_hist, cfg.vis_dim), generator=g)
            wrist_hist = scene_main_w + 0.12 * torch.randn((tot, cfg.wrist_tokens_hist, cfg.vis_dim), generator=g)
            state_norm = torch.randn((tot, cfg.state_dim), generator=g) * 0.08
            if dist_len > 0:
                scene_dist_f = torch.randn((1, cfg.front_tokens_hist, cfg.vis_dim), generator=g)
                front_hist[:dist_len] = scene_dist_f + 0.12 * front_hist[:dist_len]
                # Inject realistic execution teleport & gripper spikes at cross-session boundary
                state_norm[dist_len, :7] += 1.6
            state_norm[exec_start + 12, 7] = -0.9

            is_demo = torch.zeros((tot,), dtype=torch.bool)
            if demo_len > 0:
                is_demo[dist_len : dist_len + demo_len] = True

            p_jump = torch.zeros((tot,), dtype=torch.float32)
            if dist_len > 0:
                p_jump[dist_len] = 0.85

            c_steps = torch.arange(exec_start, tot, cfg.exec_horizon, dtype=torch.long)
            raw_cut = c_steps - cfg.exclusion_window + 1
            cutoffs = torch.maximum(
                torch.full_like(c_steps, exec_start), torch.clamp(raw_cut, min=dist_len)
            )
            m = int(c_steps.numel())

            f_256 = torch.randn((m, cfg.front_tokens_curr, cfg.vis_dim), generator=g)
            w_256 = torch.randn((m, cfg.wrist_tokens_curr, cfg.vis_dim), generator=g)
            actions = torch.randn((m, cfg.action_horizon, cfg.raw_action_dim), generator=g) * 0.3

            lang_tok = torch.randint(1, max(10, cfg.vocab_size - 1), (cfg.max_token_len,), generator=g)
            lang_mask = torch.ones((cfg.max_token_len,), dtype=torch.bool)
            lang_mask[-6:] = False
            chunk_vlm_tok = torch.randint(1, max(10, cfg.vocab_size - 1), (m, cfg.max_token_len), generator=g)
            chunk_vlm_mask = torch.ones((m, cfg.max_token_len), dtype=torch.bool)
            for mi in range(m):
                if (mi % 3) == 2:
                    chunk_vlm_mask[mi, cfg.max_token_len // 2 :] = False
                else:
                    chunk_vlm_mask[mi, -4:] = False

            xy_m11 = (torch.rand((m, 2), generator=g) * 1.4) - 0.7
            sec_xy = torch.rand((m, 2), generator=g)
            cnt_state = torch.zeros((m, 3), dtype=torch.long)
            cnt_state[:, 0] = torch.arange(m) % 4
            cnt_state[:, 1] = 3 - (torch.arange(m) % 4)
            cnt_state[:, 2] = 3

            t_name = ROBOMME_TASKS[i % len(ROBOMME_TASKS)]
            self.episodes.append(
                {
                    "task_name": t_name,
                    "suite": TASK_TO_SUITE[t_name],
                    "epis_idx": i,
                    "exec_start_idx": exec_start,
                    "distractor_prefix_frames": dist_len,
                    "total_steps": tot,
                    "prompt": f"synthetic task {i % 4}",
                    "lang_tokens": lang_tok,
                    "lang_masks": lang_mask,
                    "front_emb": front_hist,
                    "wrist_emb": wrist_hist,
                    "state_norm": state_norm,
                    "is_demo": is_demo,
                    "prompt_cos_jump": p_jump,
                    "chunk_step_indices": c_steps,
                    "readout_cutoff_excl": cutoffs,
                    "target_action_chunks": actions,
                    "exec_front_256": f_256,
                    "exec_wrist_256": w_256,
                    "chunk_target_xy_m11": xy_m11,
                    "chunk_has_xy": torch.ones((m,), dtype=torch.bool),
                    "chunk_secondary_xy": sec_xy,
                    "chunk_has_secondary_xy": torch.tensor([j % 2 == 0 for j in range(m)], dtype=torch.bool),
                    "chunk_count_state": cnt_state,
                    "chunk_has_count": torch.ones((m,), dtype=torch.bool),
                    "chunk_phase_id": torch.randint(0, cfg.num_stage_classes, (m,), generator=g),
                    "chunk_color_id": torch.randint(0, cfg.num_color_classes, (m,), generator=g),
                    "chunk_ordinal_id": torch.randint(0, 5, (m,), generator=g),
                    **(
                        {"chunk_vlm_tokens": chunk_vlm_tok, "chunk_vlm_masks": chunk_vlm_mask}
                        if cfg.arch == "p2_m7_grounded_hybrid"
                        else {}
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]


class Stratified16TaskBatchSampler(Sampler[List[int]]):
    """Stratified multi-task `batch_sampler` guaranteeing ALL tasks appear in every macro-step (`#248`).

    When `N_tasks == 16` (`ROBOMME_TASKS`), `batch_size == 4`, and `grad_accum_steps == 4`:
    every 4 consecutive micro-batches (`1` optimizer step = `16` episodes) yields a random permutation
    of all `16` RoboMME tasks (`1` episode per task, `0` task omission variance!).
    """

    def __init__(self, dataset: Any, batch_size: int = 4, seed: int = 42) -> None:
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self._epoch = 0

        buckets: Dict[str, List[int]] = {}
        if hasattr(dataset, "base") and hasattr(dataset.base, "index"):
            for idx, (task_name, _) in enumerate(dataset.base.index):
                buckets.setdefault(str(task_name), []).append(idx)
        elif hasattr(dataset, "episodes"):
            for idx, ep in enumerate(dataset.episodes):
                buckets.setdefault(str(ep.get("task_name", f"t_{idx}")), []).append(idx)
        else:
            for idx in range(len(dataset)):
                buckets.setdefault(f"b_{idx % 16}", []).append(idx)

        self.task_names = sorted(buckets.keys())
        self.buckets = buckets
        self.max_per_task = max(len(v) for v in buckets.values()) if buckets else 0
        self.num_batches = max(1, (len(self.task_names) * self.max_per_task) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + 1009 * self._epoch)
        self._epoch += 1

        pools: Dict[str, List[int]] = {}
        cursors: Dict[str, int] = {}
        for t in self.task_names:
            arr = list(self.buckets[t])
            rng.shuffle(arr)
            pools[t] = arr
            cursors[t] = 0

        flat: List[int] = []
        n_rounds = max(1, self.max_per_task)
        for _ in range(n_rounds):
            t_order = list(self.task_names)
            rng.shuffle(t_order)
            for t in t_order:
                c = cursors[t]
                if c >= len(pools[t]):
                    arr = list(self.buckets[t])
                    rng.shuffle(arr)
                    pools[t] = arr
                    c = 0
                flat.append(pools[t][c])
                cursors[t] = c + 1

        for s in range(0, len(flat) - self.batch_size + 1, self.batch_size):
            yield flat[s : s + self.batch_size]
