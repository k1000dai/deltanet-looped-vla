"""Online `Pi05MemoryAdapter` & `eval_pi05.py` Factory (`src/pi05_mem/adapter.py`).

Implements `@pi05-bridge`'s `src.pi05.policy.Pi05MemoryAdapter` protocol with:
  - Incremental `O(1)` per-chunk GDN / `T²Mem` state advancement (`committed_u -> u_end`),
  - Zero-oracle `EpisodeBoundaryForgetGate` cross-session reset (driven strictly by task instruction
    cosine jump, `exec -> demo` transition, & arm-teleport / SigLIP scene-cut — NEVER reading
    `step_idx == 0` as an oracle reset under `--interference-k`!),
  - Exact Invariant A3 demo-preserving causal cutoff (`cutoff_excl`),
  - Automatic loading of EMA delta weights (`mem_core` + `18L` hooks + optional Stage-B LoRA/Expert)
    onto `Pi05Model` via `build_eval_adapter(ckpt_path, pi05_model, device)`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from pi05 import MemModMode, Pi05Model, Pi05Tokenizer, resolve_tokenizer_path

from .config import Pi05MemConfig
from .memory_modules import pool_spatial_tokens_to_square
from .model import DeltaMemCore, DeltaMemPi05Policy


class DeltaMemEvalAdapter:
    """Stateful online memory adapter plugging directly into `Pi05RoboMMEPolicy` (`eval_pi05.py`)."""

    def __init__(
        self,
        cfg: Pi05MemConfig,
        mem_core: DeltaMemCore,
        pi05_model: Pi05Model,
        *,
        device: str | torch.device = "cuda",
        hop_mode: str = "auto",  # "l1", "l2", or "auto" (uses Stage-C CRN gate when present)
    ) -> None:
        self.cfg = cfg
        self.mem_core = mem_core.eval()
        self.pi05 = pi05_model
        self.device = torch.device(device)
        self.hop_mode = hop_mode
        self.reactive_hop2_refiner: Optional[Any] = None
        self.switch_reactive_lora: bool = False
        self.non_reactive_lora_mode: str = "base_policy"
        self.dynamic_policy_lora_on_subgoal: bool = False
        if getattr(pi05_model, "vision_tower", None) is not None:
            pi05_model.vision_tower._host_perceptual_pi05 = weakref.ref(pi05_model)  # type: ignore[attr-defined]

        try:
            self._tokenizer: Optional[Pi05Tokenizer] = Pi05Tokenizer(
                max_len=cfg.max_token_len,
                discrete_state_input=cfg.discrete_state_input,
                tokenizer_path=resolve_tokenizer_path(None),
            )
        except FileNotFoundError:
            self._tokenizer = None

        self._prompt_cache: Dict[str, Tensor] = {}
        self.reset()

    @torch.no_grad()
    def _encode_prompt_pool(self, task_goal: str) -> Tensor:
        cached = self._prompt_cache.get(task_goal)
        if cached is not None:
            return cached
        if self._tokenizer is not None:
            tok_np, mask_np = self._tokenizer.tokenize(task_goal)
            lt = torch.from_numpy(tok_np).unsqueeze(0).to(self.device)
            lm = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        else:
            # Deterministic byte-hash fallback for offline unit tests without SentencePiece model file
            ids = [((ord(c) * 131 + i) % max(2, self.cfg.vocab_size - 2)) + 1 for i, c in enumerate(task_goal[:48])]
            if not ids:
                ids = [1]
            pad = max(0, self.cfg.max_token_len - len(ids))
            lt = torch.tensor([ids + [0] * pad], device=self.device, dtype=torch.long)
            lm = torch.tensor([[True] * len(ids) + [False] * pad], device=self.device, dtype=torch.bool)

        v_max = int(self.pi05.paligemma_with_expert.embed_tokens.num_embeddings) - 1
        tok_ids = (lt.long() % max(2, v_max)) if v_max < 4096 else lt.long().clamp(0, v_max)
        t_emb = self.pi05.paligemma_with_expert.embed_language(tok_ids)
        lm_f = lm.to(t_emb.dtype).unsqueeze(-1)
        pool = ((t_emb * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)).detach()
        self._prompt_cache[task_goal] = pool
        return pool

    def reset(self) -> None:
        self.stream_total_frames: int = 0
        self.demo_frames_in_session: int = 0
        self.session_start_frame: int = 0
        self.session_exec_start_frame: int = 0
        self._last_prompt_pool: Optional[Tensor] = None
        self._pending_prompt_jump: float = 0.0
        self._prepared_write_batch: Optional[Tuple[List[int], List[float], bool]] = None

        # Kinematics & CausalConv1D continuity across written frames
        self._prev_w_front: Optional[Tensor] = None
        self._prev_w_state: Optional[Tensor] = None
        self._prev_w_demo: Optional[Tensor] = None
        self._conv_hist: Optional[Tensor] = None
        self._surprise_ema: float = 0.0
        self._pure_vis_ema: float = 0.0

        # Incremental written-frame buffers (`1` entry per strided written frame)
        self._w_steps: List[int] = []
        self._w_is_demo: List[bool] = []
        self._w_front_hist: List[Tensor] = []
        self._w_gdn_slots: List[Tensor] = []
        self._w_lm_slots: List[Tensor] = []
        self._w_vis_rms: List[float] = []
        self._w_grip_spike: List[float] = []
        self._w_b_reset: List[float] = []
        self._w_salience: List[float] = []
        self._w_pure_vis_salience: List[float] = []

        # Incremental committed pointer & states (`O(1)` update per chunk!)
        self._committed_u: int = 0
        self._gdn_states = (
            self.mem_core.gdn.init_empty_states(1, self.device)
            if self.mem_core.gdn is not None
            else None
        )
        self._surp_l1_state = (
            torch.zeros(
                (1, self.cfg.gdn_heads, self.cfg.gdn_d_v, self.cfg.gdn_d_k),
                device=self.device,
                dtype=torch.float32,
            )
            if self.mem_core.gdn is not None
            else None
        )
        self._ttt_state = (
            self.mem_core.t2mem.init_state(1, self.device)
            if self.mem_core.t2mem is not None
            else None
        )
        if self.switch_reactive_lora:
            from .deltaground_vlm import set_vlm_lora_mode

            set_vlm_lora_mode(self.pi05, self.non_reactive_lora_mode)  # type: ignore[arg-type]

    @torch.no_grad()
    def predict_subgoal(
        self,
        *,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        del front_tokens_256, wrist_tokens_256, raw_state, norm_state, step_idx, exec_start_idx
        if self.switch_reactive_lora:
            from .deltaground import should_route_to_reactive_precision
            from .deltaground_vlm import set_vlm_lora_mode

            target_m = (
                "reactive_policy"
                if should_route_to_reactive_precision(task_goal)
                else self.non_reactive_lora_mode
            )
            set_vlm_lora_mode(self.pi05, target_m)  # type: ignore[arg-type]
        return None

    def _should_write_frame(
        self,
        local_session_idx: int,
        is_demo: bool,
        is_batch_end_demo: bool,
        *,
        write_stride_exec_override: Optional[int] = None,
    ) -> bool:
        if local_session_idx < int(self.cfg.dense_prefix_frames):
            return True
        if is_demo:
            stride = (
                int(self.cfg.long_demo_stride)
                if local_session_idx >= int(self.cfg.adaptive_demo_threshold)
                else max(1, int(self.cfg.write_stride_demo))
            )
            return (local_session_idx % stride == 0) or is_batch_end_demo
        exec_rel = max(0, local_session_idx - (self.session_exec_start_frame - self.session_start_frame))
        eff_es = self.cfg.write_stride_exec if write_stride_exec_override is None else write_stride_exec_override
        return (exec_rel % max(1, int(eff_es))) == 0

    @torch.no_grad()
    def select_write_indices(
        self,
        num_frames: int,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> List[int]:
        """Pre-select which frame indices in `0..num_frames-1` will be written BEFORE `encode_images` (`#232`).

        Advances `stream_total_frames` & prompt-jump tracking identically to `ingest_frames`, stashing
        the prepared `(sel_stream_steps, sel_p_jumps, is_demo)` tuple for `ingest_selected_frames(...)`
        when `len(sel_local) > 0` (`0.0` bit-diff vs `ingest_frames`, `~2x` fewer SigLIP ViT passes!).
        """
        t_in = int(num_frames)
        if t_in <= 0:
            self._prepared_write_batch = None
            return []
        if self.cfg.arch == "p2_b0_pi05_base":
            self.stream_total_frames += t_in
            self._prepared_write_batch = None
            return []

        cur_p_pool = self._encode_prompt_pool(task_goal)
        if self._last_prompt_pool is not None:
            cos = F.cosine_similarity(cur_p_pool.float(), self._last_prompt_pool.float(), dim=-1).item()
            p_jump = max(0.0, 1.0 - float(cos))
            self._pending_prompt_jump = max(self._pending_prompt_jump, p_jump)
        self._last_prompt_pool = cur_p_pool

        sel_local: List[int] = []
        sel_stream_steps: List[int] = []
        sel_p_jumps: List[float] = []

        for i in range(t_in):
            g_step = self.stream_total_frames + i
            # Detect `exec -> demo` new-episode start before deciding stride!
            if is_demo and len(self._w_is_demo) > 0 and (not self._w_is_demo[-1]) and self.demo_frames_in_session == 0:
                self.session_start_frame = g_step
                self.session_exec_start_frame = g_step
            if self._pending_prompt_jump > 0.02:
                self.demo_frames_in_session = 0
            local_idx = g_step - self.session_start_frame
            if is_demo:
                self.demo_frames_in_session += 1
                is_end_demo = (
                    (self.demo_frames_in_session == int(total_demo_len))
                    if total_demo_len is not None
                    else (i == t_in - 1)
                )
            else:
                self.demo_frames_in_session = 0
                is_end_demo = False
            w_stride_ov: Optional[int] = None
            if (not is_demo) and bool(self.cfg.use_reactive_precision_mode):
                from .deltaground import should_use_reactive_stopcube

                if should_use_reactive_stopcube(task_goal):
                    w_stride_ov = 1
            if self._should_write_frame(
                local_idx, is_demo=is_demo, is_batch_end_demo=is_end_demo, write_stride_exec_override=w_stride_ov
            ) or (self._pending_prompt_jump > 0.02):
                sel_local.append(i)
                sel_stream_steps.append(g_step)
                sel_p_jumps.append(self._pending_prompt_jump)
                self._pending_prompt_jump = 0.0

        self.stream_total_frames += t_in
        if is_demo:
            self.session_exec_start_frame = self.stream_total_frames

        if sel_local:
            self._prepared_write_batch = (sel_stream_steps, sel_p_jumps, bool(is_demo))
        else:
            self._prepared_write_batch = None
        return sel_local

    @torch.no_grad()
    def ingest_selected_frames(
        self,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        """Ingest ONLY the pre-selected `len(sel_local)` frames returned by `select_write_indices`."""
        del task_goal
        prep = self._prepared_write_batch
        self._prepared_write_batch = None
        if prep is None:
            return
        sel_stream_steps, sel_p_jumps, prep_demo = prep
        is_demo = bool(prep_demo)
        w_cnt = len(sel_stream_steps)
        if w_cnt == 0 or int(front_tokens_256.shape[0]) == 0:
            return
        if int(front_tokens_256.shape[0]) != w_cnt:
            raise ValueError(
                f"ingest_selected_frames expected {w_cnt} selected frames from select_write_indices, "
                f"got {int(front_tokens_256.shape[0])}"
            )

        assert self.mem_core.writer is not None
        f_w = pool_spatial_tokens_to_square(front_tokens_256.to(self.device), self.cfg.front_tokens_hist)
        if wrist_tokens_256 is not None:
            w_w = pool_spatial_tokens_to_square(wrist_tokens_256.to(self.device), self.cfg.wrist_tokens_hist)
        else:
            w_w = torch.zeros(
                (w_cnt, self.cfg.wrist_tokens_hist, f_w.shape[-1]), device=self.device, dtype=f_w.dtype
            )
        s_w = torch.from_numpy(np.asarray(norm_states, dtype=np.float32)).to(self.device)
        d_w = torch.full((w_cnt,), bool(is_demo), device=self.device, dtype=torch.bool)
        # Local session step indices keep positional embeddings invariant to cross-session stream offset!
        local_step_t = torch.tensor(
            [st_s - self.session_start_frame for st_s in sel_stream_steps],
            device=self.device,
            dtype=torch.long,
        )

        kin = self.mem_core.writer.compute_frame_kinematics(
            f_w,
            s_w,
            d_w,
            prev_front=self._prev_w_front,
            prev_state=self._prev_w_state,
            prev_is_demo=self._prev_w_demo,
        )
        self._prev_w_front = f_w[-1].detach().clone()
        self._prev_w_state = s_w[-1].detach().clone()
        self._prev_w_demo = d_w[-1].detach().clone()

        raw_gdn, lm_slots = self.mem_core.writer.encode_frames_raw(f_w, w_w, s_w, local_step_t, d_w, kin)
        gdn_slots, self._conv_hist = self.mem_core.writer.apply_slot_conv(raw_gdn, conv_history=self._conv_hist)

        pj_t = torch.tensor(sel_p_jumps, device=self.device, dtype=torch.float32)
        b_gate = None
        if self.mem_core.gdn is not None and self.mem_core.gdn.boundary_gate is not None:
            b_gate = self.mem_core.gdn.boundary_gate
        elif self.mem_core.t2mem is not None and self.mem_core.t2mem.boundary_gate is not None:
            b_gate = self.mem_core.t2mem.boundary_gate

        if b_gate is not None:
            br_t = b_gate(gdn_slots[:, 0, :], kin, prompt_cos_jump=pj_t)
        else:
            br_t = torch.zeros((w_cnt,), device=self.device, dtype=torch.float32)

        # Update session anchors if a boundary reset fired on an execution-start transition (`k=3` un-reset!)
        for bi in range(w_cnt):
            if float(br_t[bi].item()) > float(self.cfg.boundary_flush_threshold):
                self.session_start_frame = sel_stream_steps[bi]
                if not is_demo:
                    self.session_exec_start_frame = sel_stream_steps[bi]
                # Reset conv history cleanly at detected session boundary
                if self._conv_hist is not None:
                    self._conv_hist.zero_()

        # Compute online causal GDN write-surprise on newly written frames if landmark bank or hybrid bridge is active
        if self.mem_core.gdn is not None and (
            self.mem_core.landmark_bank is not None or self.mem_core.hybrid_bridge is not None
        ):
            assert self._surp_l1_state is not None
            k1, v1, g1, b1, _ = self.mem_core.gdn.layer1.project_slots(
                gdn_slots.unsqueeze(0),
                vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
                grip_spike=kin["grip_spike"].unsqueeze(0),
                boundary_reset=br_t.unsqueeze(0),
            )
            raw_surp = self.mem_core.gdn.layer1.compute_exact_write_surprise(
                k1,
                v1,
                g1,
                b1,
                num_write_slots=self.cfg.num_write_slots,
                initial_state_fp32=self._surp_l1_state,
            ).squeeze(0)
            # Advance `_surp_l1_state`
            from models.deltaloop_vla import run_dual_backend_gdn

            _, self._surp_l1_state, _ = run_dual_backend_gdn(
                k=k1, v=v1, g_fp32=g1, beta_fp32=b1, q=None, initial_state_fp32=self._surp_l1_state, track_occ=False
            )
            if (not is_demo) and len(self._w_is_demo) > 0 and self._w_is_demo[-1]:
                self._surprise_ema = 0.0
                self._pure_vis_ema = 0.0
            sal_t, self._surprise_ema = self.mem_core.smooth_surprise_ema(
                raw_surp,
                kin["grip_spike"],
                torch.zeros_like(d_w),  # Apply +1.0 first/last demo boost once across session at readout!
                br_t,
                ema_beta=self.cfg.surprise_ema_beta,
                init_ema=self._surprise_ema,
            )
            pvis_t, self._pure_vis_ema = self.mem_core.smooth_surprise_ema(
                raw_surp,
                torch.zeros_like(kin["grip_spike"]),
                torch.zeros_like(d_w),
                br_t,
                ema_beta=self.cfg.surprise_ema_beta,
                init_ema=self._pure_vis_ema,
            )
        else:
            sal_t = torch.zeros((w_cnt,), device=self.device, dtype=torch.float32)
            pvis_t = torch.zeros((w_cnt,), device=self.device, dtype=torch.float32)

        for wi in range(w_cnt):
            self._w_steps.append(sel_stream_steps[wi])
            self._w_is_demo.append(bool(is_demo))
            if self.mem_core.hybrid_bridge is not None or self.mem_core.vlm_surprise_selector is not None:
                self._w_front_hist.append(f_w[wi].detach())
            self._w_gdn_slots.append(gdn_slots[wi])
            self._w_lm_slots.append(lm_slots[wi])
            self._w_vis_rms.append(float(kin["vis_motion_rms"][wi].item()))
            self._w_grip_spike.append(float(kin["grip_spike"][wi].item()))
            self._w_b_reset.append(float(br_t[wi].item()))
            self._w_salience.append(float(sal_t[wi].item()))
            self._w_pure_vis_salience.append(float(pvis_t[wi].item()))

    @torch.no_grad()
    def ingest_frames(
        self,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> None:
        t_in = int(front_tokens_256.shape[0])
        if t_in == 0:
            return
        sel_local = self.select_write_indices(
            t_in, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        if not sel_local:
            return
        idx_t = torch.tensor(sel_local, device=front_tokens_256.device, dtype=torch.long)
        f_sel = front_tokens_256[idx_t]
        w_sel = wrist_tokens_256[idx_t.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
        raw_np = np.asarray(raw_states)[sel_local]
        norm_np = np.asarray(norm_states)[sel_local]
        self.ingest_selected_frames(
            f_sel, w_sel, raw_np, norm_np, task_goal=task_goal, is_demo=is_demo
        )

    @torch.no_grad()
    def predict_vlm_prefix_tokens(
        self,
        *,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> Tuple[Tensor | None, Tensor | None]:
        """Return `(extra_prefix_embs [1, S_vlm, 2048], extra_prefix_mask [1, S_vlm])` for `P2-M8` (`#329`/`#330`)."""
        del wrist_tokens_256, raw_state, norm_state, task_goal, step_idx, exec_start_idx
        sel_mod = self.mem_core.vlm_surprise_selector
        if sel_mod is None:
            return None, None

        q_stream_step = max(0, self.stream_total_frames - 1)
        w_tot = len(self._w_steps)
        if w_tot == 0:
            empty_f = torch.empty(
                (0, self.cfg.front_tokens_hist, self.cfg.vis_dim),
                device=self.device,
                dtype=front_tokens_256.dtype,
            )
            vt, vm = sel_mod.encode_vlm_surprise_tokens(empty_f)
            return vt.unsqueeze(0).to(self.pi05.compute_dtype), vm.unsqueeze(0)

        w_steps_t = torch.tensor(self._w_steps, device=self.device, dtype=torch.long)
        d_all_v = torch.tensor(self._w_is_demo, device=self.device, dtype=torch.bool)
        br_all_v = torch.tensor(self._w_b_reset, device=self.device, dtype=torch.float32)
        sal_all_v = torch.tensor(self._w_salience, device=self.device, dtype=torch.float32)
        pvis_all_v = torch.tensor(self._w_pure_vis_salience, device=self.device, dtype=torch.float32)

        vr_v = torch.nonzero(br_all_v > float(self.cfg.boundary_flush_threshold), as_tuple=False).flatten()
        sp_v = int(vr_v[-1].item()) if vr_v.numel() > 0 else 0
        di_v = torch.nonzero(d_all_v[sp_v:w_tot], as_tuple=False).flatten()
        if di_v.numel() >= 1:
            sal_all_v[sp_v + int(di_v[0].item())] += 1.0
            sal_all_v[sp_v + int(di_v[-1].item())] += 1.0

        v_sel = sel_mod.select_vlm_surprise_indices(
            w_steps_t,
            d_all_v,
            sal_all_v,
            pvis_all_v,
            br_all_v,
            cutoff_excl_frame=q_stream_step + 1,
        )
        if int(v_sel.numel()) > 0:
            f_vlm = torch.stack([self._w_front_hist[int(vi)] for vi in v_sel.tolist()], dim=0)
        else:
            f_vlm = torch.empty(
                (0, self.cfg.front_tokens_hist, self.cfg.vis_dim),
                device=self.device,
                dtype=front_tokens_256.dtype,
            )
        vt, vm = sel_mod.encode_vlm_surprise_tokens(f_vlm)
        return vt.unsqueeze(0).to(self.pi05.compute_dtype), vm.unsqueeze(0)

    @torch.no_grad()
    def read_memory(
        self,
        *,
        prefix_hidden: Tensor | None = None,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
        cont_vlm_ground: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor | None, Tensor | None, Dict[str, Any]]:
        if self.cfg.arch == "p2_b0_pi05_base":
            return None, None, {"loop_depth": 1, "mem_active_tokens": 0}

        assert self.mem_core.obs_encoder is not None
        is_reac_stop = False
        is_prec_peg = False
        if bool(self.cfg.use_reactive_precision_mode):
            from .deltaground import should_use_precision_insertpeg, should_use_reactive_stopcube

            is_reac_stop = should_use_reactive_stopcube(task_goal)
            is_prec_peg = should_use_precision_insertpeg(task_goal)
        eff_excl = int(self.cfg.reactive_stopcube_excl) if is_reac_stop else int(self.cfg.exclusion_window)
        # `predict_chunk` just ingested the current frame, so `query_stream_step = stream_total_frames - 1`
        q_stream_step = max(0, self.stream_total_frames - 1)
        cutoff_stream = max(
            self.session_exec_start_frame,
            max(self.session_start_frame, q_stream_step - eff_excl + 1),
        )

        p_pool = self._encode_prompt_pool(task_goal)
        st_cur = torch.from_numpy(np.asarray(norm_state, dtype=np.float32)).unsqueeze(0).to(self.device)
        ctx_tokens, front_spatial_ctx = self.mem_core.obs_encoder(
            front_tokens_256.to(self.device),
            wrist_tokens_256.to(self.device),
            st_cur,
            prompt_pool=p_pool,
        )

        w_tot = len(self._w_steps)
        w_steps_t = (
            torch.tensor(self._w_steps, device=self.device, dtype=torch.long)
            if w_tot > 0
            else torch.empty((0,), device=self.device, dtype=torch.long)
        )
        u_end = int(torch.searchsorted(w_steps_t, cutoff_stream, right=False).item()) if w_tot > 0 else 0
        u_query = int(torch.searchsorted(w_steps_t, q_stream_step, right=True).item()) if w_tot > 0 else 0

        loop_depth = 1
        b_reset_max = max(self._w_b_reset[:u_query]) if u_query > 0 else 0.0
        pending_r = max(self._w_b_reset[u_end:u_query]) if u_query > u_end else 0.0
        flush_now = pending_r > float(self.cfg.boundary_flush_threshold)

        # 1. GDN / DeltaMem family
        if self.mem_core.gdn is not None:
            assert self._gdn_states is not None
            if u_end > self._committed_u:
                seg_slots = torch.stack(self._w_gdn_slots[self._committed_u : u_end], dim=0).unsqueeze(0)
                seg_kin = {
                    "vis_motion_rms": torch.tensor(
                        self._w_vis_rms[self._committed_u : u_end], device=self.device, dtype=torch.float32
                    ),
                    "grip_spike": torch.tensor(
                        self._w_grip_spike[self._committed_u : u_end], device=self.device, dtype=torch.float32
                    ),
                }
                seg_br = torch.tensor(
                    self._w_b_reset[self._committed_u : u_end], device=self.device, dtype=torch.float32
                ).unsqueeze(0)
                seg_out = self.mem_core.gdn.forward_segment(
                    seg_slots, seg_kin, seg_br, init_states=self._gdn_states, compute_surprise=False
                )
                self._gdn_states = (seg_out["S1"], seg_out["occ1"], seg_out["S2"], seg_out["occ2"])
                self._committed_u = u_end

            S1_p, o1_p, S2_p, o2_p = self._gdn_states
            has_hist_cur = (u_end > 0) and (not flush_now)
            if flush_now:
                df = math.exp(-float(self.cfg.boundary_reset_strength) * pending_r)
                S1_p, o1_p, S2_p, o2_p = S1_p * df, o1_p * df, S2_p * df, o2_p * df

            gdn_tok_l1, cov_stats = self.mem_core.gdn.readout(ctx_tokens, S1_p, o1_p, S2_p, o2_p)
            n_gdn_tot = int(gdn_tok_l1.shape[1])
            gdn_mask = torch.full((1, n_gdn_tot), has_hist_cur, device=self.device, dtype=torch.bool)
            if self.cfg.num_persistent_tokens > 0:
                gdn_mask[:, : self.cfg.num_persistent_tokens] = True

            tok_parts = [gdn_tok_l1]
            mask_parts = [gdn_mask]
            if self.mem_core.landmark_bank is not None:
                if has_hist_cur:
                    lm_slots_all = torch.stack(self._w_lm_slots[:u_end], dim=0)
                    d_all = torch.tensor(self._w_is_demo[:u_end], device=self.device, dtype=torch.bool)
                    sal_all = torch.tensor(self._w_salience[:u_end], device=self.device, dtype=torch.float32)
                    br_all = torch.tensor(self._w_b_reset[:u_end], device=self.device, dtype=torch.float32)
                    vr_lm = torch.nonzero(br_all > float(self.cfg.boundary_flush_threshold), as_tuple=False).flatten()
                    sp_lm = int(vr_lm[-1].item()) if vr_lm.numel() > 0 else 0
                    di_lm = torch.nonzero(d_all[sp_lm:u_end], as_tuple=False).flatten()
                    if di_lm.numel() >= 1:
                        sal_all[sp_lm + int(di_lm[0].item())] += 1.0
                        sal_all[sp_lm + int(di_lm[-1].item())] += 1.0
                    w_sub = w_steps_t[:u_end]
                    sel_pos = self.mem_core.landmark_bank.select_landmark_indices(
                        w_sub, d_all, sal_all, br_all, cutoff_excl_frame=cutoff_stream
                    )
                    lt, lm = self.mem_core.landmark_bank.gather_padded_landmarks(
                        lm_slots_all, w_sub, d_all, sal_all, sel_pos, query_step_idx=q_stream_step
                    )
                else:
                    empty_lm = torch.empty((0, 16, self.cfg.mem_token_dim), device=self.device, dtype=gdn_tok_l1.dtype)
                    empty_idx = torch.empty((0,), device=self.device, dtype=torch.long)
                    empty_b = torch.empty((0,), device=self.device, dtype=torch.bool)
                    empty_f = torch.empty((0,), device=self.device, dtype=torch.float32)
                    lt, lm = self.mem_core.landmark_bank.gather_padded_landmarks(
                        empty_lm, empty_idx, empty_b, empty_f, empty_idx, query_step_idx=q_stream_step
                    )
                tok_parts.append(lt.unsqueeze(0).to(gdn_tok_l1.dtype))
                mask_parts.append(lm.unsqueeze(0))

            mem_tok = torch.cat(tok_parts, dim=1)
            mem_mask = torch.cat(mask_parts, dim=1)

            aux_preds = (
                self.mem_core.aux_suite(front_spatial_ctx, ctx_tokens, mem_tok, mem_mask)
                if self.mem_core.aux_suite is not None
                else None
            )
            if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
                sg_tok = aux_preds["self_subgoal_tokens"].to(mem_tok.dtype)
                sg_mask = torch.ones((1, sg_tok.shape[1]), device=self.device, dtype=torch.bool)
                mem_tok = torch.cat([mem_tok, sg_tok], dim=1)
                mem_mask = torch.cat([mem_mask, sg_mask], dim=1)
            mem_tok = torch.where(mem_mask.unsqueeze(-1), mem_tok, torch.zeros_like(mem_tok))

            if self.mem_core.hybrid_bridge is not None:
                if self.pi05.mem_encoder is None:
                    raise ValueError("`p2_m5_hybrid_surprise_gdn` requires `pi05.mem_encoder`")
                d_all_h = (
                    torch.tensor(self._w_is_demo[:w_tot], device=self.device, dtype=torch.bool)
                    if w_tot > 0
                    else torch.empty((0,), device=self.device, dtype=torch.bool)
                )
                sal_all_h = (
                    torch.tensor(self._w_salience[:w_tot], device=self.device, dtype=torch.float32)
                    if w_tot > 0
                    else torch.empty((0,), device=self.device, dtype=torch.float32)
                )
                br_all_h = (
                    torch.tensor(self._w_b_reset[:w_tot], device=self.device, dtype=torch.float32)
                    if w_tot > 0
                    else torch.empty((0,), device=self.device, dtype=torch.float32)
                )
                if w_tot > 0:
                    vr_h = torch.nonzero(br_all_h > float(self.cfg.boundary_flush_threshold), as_tuple=False).flatten()
                    sp_h = int(vr_h[-1].item()) if vr_h.numel() > 0 else 0
                    di_h = torch.nonzero(d_all_h[sp_h:w_tot], as_tuple=False).flatten()
                    if di_h.numel() >= 1:
                        sal_all_h[sp_h + int(di_h[0].item())] += 1.0
                        sal_all_h[sp_h + int(di_h[-1].item())] += 1.0
                        if is_prec_peg:
                            d_abs_h = sp_h + di_h
                            grip_h = torch.tensor(self._w_grip_spike[:w_tot], device=self.device, dtype=torch.float32)
                            gr_m = grip_h[d_abs_h] > 0.20
                            if bool(gr_m.any()):
                                sal_all_h[d_abs_h[gr_m]] += 3.0
                            sal_all_h[d_abs_h[-min(4, int(d_abs_h.numel())) :]] += 3.0
                anc_ov_h = min(float(self.cfg.hybrid_anchor_ratio), 0.75) if is_prec_peg else None
                # `#259` Bug #2 Fix: Perceptual canvas uses `excl=0` (`q_stream_step + 1`), keeping `excl=16` on GDN!
                sel_u, loc_pos, sess_s_h = self.mem_core.hybrid_bridge.select_hybrid_keyframes(
                    w_steps_t,
                    d_all_h,
                    sal_all_h,
                    br_all_h,
                    cutoff_excl_frame=q_stream_step + 1,
                    query_step_idx=q_stream_step,
                    anchor_ratio_override=anc_ov_h,
                )
                if int(sel_u.numel()) > 0:
                    f_sel = torch.stack([self._w_front_hist[int(ui)] for ui in sel_u.tolist()], dim=0)
                else:
                    f_sel = torch.empty(
                        (0, 64, self.cfg.vis_dim), device=self.device, dtype=front_tokens_256.dtype
                    )
                c_tok, c_m = self.mem_core.hybrid_bridge.encode_perceptual_canvas(
                    f_sel,
                    loc_pos,
                    self.pi05.mem_encoder,
                    fallback_curr_front=front_tokens_256[0].to(self.device),
                    query_local_step=max(0, q_stream_step - sess_s_h),
                )
                mem_tok = self.mem_core.hybrid_bridge(
                    c_tok.unsqueeze(0),
                    c_m.unsqueeze(0),
                    mem_tok,
                    mem_mask,
                    ctx_tokens,
                )
                mem_mask = c_m.unsqueeze(0)
                if self.mem_core.continuous_vlm_bridge is not None and cont_vlm_ground is not None:
                    mem_tok = self.mem_core.continuous_vlm_bridge(mem_tok, mem_mask, cont_vlm_ground)

            if is_prec_peg and (self.reactive_hop2_refiner is not None or self.mem_core.hop2_refiner is not None):
                active_hop2 = self.reactive_hop2_refiner if self.reactive_hop2_refiner is not None else self.mem_core.hop2_refiner
                assert active_hop2 is not None
                np_off = int(self.cfg.num_persistent_tokens)
                gdn_only = gdn_tok_l1[:, np_off : np_off + self.cfg.num_gdn_readout_tokens, :]
                q_bias2 = active_hop2.compute_query_bias(gdn_only)
                gdn_hop2, _ = self.mem_core.gdn.readout(
                    ctx_tokens, S1_p, o1_p, S2_p, o2_p, extra_query_bias=q_bias2
                )
                mem_tok = active_hop2(mem_tok, mem_mask, gdn_hop2, ctx_tokens)
                loop_depth = 2
            elif self.mem_core.hop2_refiner is not None and self.mem_core.crn_gate is not None and self.hop_mode != "l1":
                from .deltaground import should_use_hop2_trajectory_imitation

                if self.hop_mode == "imitation_l2":
                    run_hop2 = should_use_hop2_trajectory_imitation(task_goal)
                elif self.hop_mode == "l2":
                    run_hop2 = True
                else:
                    crn_logit = self.mem_core.crn_gate(ctx_tokens, mem_tok, mem_mask, cov_stats)
                    run_hop2 = float(torch.sigmoid(crn_logit).item()) > 0.5
                if run_hop2:
                    np_off = int(self.cfg.num_persistent_tokens)
                    gdn_only = gdn_tok_l1[:, np_off : np_off + self.cfg.num_gdn_readout_tokens, :]
                    q_bias2 = self.mem_core.hop2_refiner.compute_query_bias(gdn_only)
                    gdn_hop2, _ = self.mem_core.gdn.readout(
                        ctx_tokens, S1_p, o1_p, S2_p, o2_p, extra_query_bias=q_bias2
                    )
                    mem_tok = self.mem_core.hop2_refiner(mem_tok, mem_mask, gdn_hop2, ctx_tokens)
                    loop_depth = 2

            info: Dict[str, Any] = {
                "loop_depth": loop_depth,
                "mem_active_tokens": int(mem_mask.sum().item()),
                "committed_written_frames": self._committed_u,
                "boundary_reset_max": b_reset_max,
            }
            if is_reac_stop or is_prec_peg:
                info["recommended_exec_horizon"] = 4
            if aux_preds is not None:
                info["pred_subgoal"] = self.mem_core.aux_suite.format_predicted_subgoal_strings(aux_preds)[0]  # type: ignore[union-attr]
            return mem_tok.to(self.pi05.compute_dtype), mem_mask, info

        # 2. `T²Mem` baseline
        if self.mem_core.t2mem is not None:
            assert self._ttt_state is not None
            if u_end > self._committed_u:
                seg_slots = torch.stack(self._w_gdn_slots[self._committed_u : u_end], dim=0).unsqueeze(0)
                seg_br = torch.tensor(
                    self._w_b_reset[self._committed_u : u_end], device=self.device, dtype=torch.float32
                ).unsqueeze(0)
                self._ttt_state = self.mem_core.t2mem.update_segment(seg_slots, seg_br, state=self._ttt_state)
                self._committed_u = u_end
            mem_tok = self.mem_core.t2mem.query_readout(ctx_tokens, self._ttt_state)
            mem_mask = torch.full((1, mem_tok.shape[1]), u_end > 0, device=self.device, dtype=torch.bool)
            aux_preds = (
                self.mem_core.aux_suite(front_spatial_ctx, ctx_tokens, mem_tok, mem_mask)
                if self.mem_core.aux_suite is not None
                else None
            )
            if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
                sg_tok = aux_preds["self_subgoal_tokens"].to(mem_tok.dtype)
                sg_mask = torch.ones((1, sg_tok.shape[1]), device=self.device, dtype=torch.bool)
                mem_tok = torch.cat([mem_tok, sg_tok], dim=1)
                mem_mask = torch.cat([mem_mask, sg_mask], dim=1)
            return mem_tok.to(self.pi05.compute_dtype), mem_mask, {
                "loop_depth": 1,
                "mem_active_tokens": int(mem_mask.sum().item()),
            }

        # 3. `FrameSamp` baseline
        assert self.mem_core.framesamp is not None
        if u_end > 0:
            slots_all = torch.stack(self._w_gdn_slots[:u_end], dim=0)
            ft, fm = self.mem_core.framesamp.gather_frames(
                slots_all, w_steps_t[:u_end], cutoff_excl_frame=cutoff_stream, query_step_idx=q_stream_step
            )
        else:
            empty_s = torch.empty((0, self.cfg.num_write_slots, self.cfg.d_mem), device=self.device)
            ft, fm = self.mem_core.framesamp.gather_frames(
                empty_s, w_steps_t[:0], cutoff_excl_frame=cutoff_stream, query_step_idx=q_stream_step
            )
        mem_tok = ft.unsqueeze(0)
        mem_mask = fm.unsqueeze(0)
        aux_preds = (
            self.mem_core.aux_suite(front_spatial_ctx, ctx_tokens, mem_tok, mem_mask)
            if self.mem_core.aux_suite is not None
            else None
        )
        if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
            sg_tok = aux_preds["self_subgoal_tokens"].to(mem_tok.dtype)
            sg_mask = torch.ones((1, sg_tok.shape[1]), device=self.device, dtype=torch.bool)
            mem_tok = torch.cat([mem_tok, sg_tok], dim=1)
            mem_mask = torch.cat([mem_mask, sg_mask], dim=1)
        return mem_tok.to(self.pi05.compute_dtype), mem_mask, {
            "loop_depth": 1,
            "mem_active_tokens": int(mem_mask.sum().item()),
        }


def build_eval_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> DeltaMemEvalAdapter:
    """Default `K=32 / 512t` `TimeUniform` factory (`#265`+`#266`: upgrades 256t Stage-A ckpts seamlessly!)."""
    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_eval_adapter`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    if cfg.arch == "p2_m7_grounded_hybrid" and isinstance(raw, dict) and ("deltaground" in raw or "ema_deltaground" in raw):
        return build_grounded_hybrid_adapter(ckpt_path, pi05_model, device=device, hop_mode=hop_mode)  # type: ignore[return-value]
    if cfg.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid"):
        cfg.hybrid_anchor_ratio = max(float(cfg.hybrid_anchor_ratio), 0.875)
        if int(cfg.mem_token_dim) >= 512:
            cfg.hybrid_num_keyframes = 32
            cfg.hybrid_canvas_tokens = 512
            cfg.q_rope_offset = 512

    dev = torch.device(device)
    prec = pi05_model.config.precision
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False)
    if (
        cfg.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid")
        and int(cfg.hybrid_canvas_tokens) == int(cfg.q_rope_offset or 512)
    ):
        for blk in wrapper.pi05.paligemma_with_expert.layers:
            if blk.mem_attn is not None and hasattr(blk.mem_attn, "k_rope_span_frames"):
                blk.mem_attn.k_rope_span_frames = False  # 512t == q_rope_offset, contiguous 0..511!
    wrapper.eval()
    return DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=hop_mode)


def build_eval_adapter_512(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> DeltaMemEvalAdapter:
    """Explicit alias for `K=32 / 512-token` `TimeUniform` evaluation (`#265`/`#266`)."""
    return build_eval_adapter(ckpt_path, pi05_model, device=device, hop_mode=hop_mode)


def build_eval_adapter_imitation_l2(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "imitation_l2",
) -> DeltaMemEvalAdapter:
    """Zero-oracle `#401` Stage-C Hop-2 adapter (`L=2` `65.0%` on `Imitation`, `0.0`-diff `L=1` `80.0%` on `Counting`)."""
    del hop_mode
    return build_eval_adapter(ckpt_path, pi05_model, device=device, hop_mode="imitation_l2")


def build_eval_adapter_256(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> DeltaMemEvalAdapter:
    """`K=16 / 256-token` (`2.0x` compact Pareto point) `TimeUniform` + strided-RoPE entrypoint."""
    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_eval_adapter_256`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    if cfg.arch == "p2_m5_hybrid_surprise_gdn":
        cfg.hybrid_num_keyframes = 16
        cfg.hybrid_canvas_tokens = 256
        cfg.q_rope_offset = 512
        cfg.hybrid_anchor_ratio = max(float(cfg.hybrid_anchor_ratio), 0.875)

    dev = torch.device(device)
    prec = pi05_model.config.precision
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False)
    wrapper.eval()
    return DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=hop_mode)


def build_eval_adapter_anc1(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> DeltaMemEvalAdapter:
    """`K=32 / 512t` `hybrid_anchor_ratio=1.0` (`32/32` pure `TimeUniform` RoPE + `DeltaMem` cross-attn, `#269`)."""
    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_eval_adapter_anc1`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    if cfg.arch == "p2_m5_hybrid_surprise_gdn":
        if int(cfg.mem_token_dim) >= 512:
            cfg.hybrid_num_keyframes = 32
            cfg.hybrid_canvas_tokens = 512
            cfg.q_rope_offset = 512
        cfg.hybrid_anchor_ratio = 1.0

    dev = torch.device(device)
    prec = pi05_model.config.precision
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False)
    if cfg.arch == "p2_m5_hybrid_surprise_gdn" and int(cfg.hybrid_canvas_tokens) == int(cfg.q_rope_offset or 512):
        for blk in wrapper.pi05.paligemma_with_expert.layers:
            if blk.mem_attn is not None and hasattr(blk.mem_attn, "k_rope_span_frames"):
                blk.mem_attn.k_rope_span_frames = False
    wrapper.eval()
    return DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=hop_mode)


def build_eval_adapter_anc0875(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> DeltaMemEvalAdapter:
    """`K=32 / 512t` `hybrid_anchor_ratio=0.875` (`28/32` `TimeUniform` + `4` canvas surprise peaks, `#277`)."""
    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_eval_adapter_anc0875`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    if cfg.arch == "p2_m5_hybrid_surprise_gdn":
        if int(cfg.mem_token_dim) >= 512:
            cfg.hybrid_num_keyframes = 32
            cfg.hybrid_canvas_tokens = 512
            cfg.q_rope_offset = 512
        cfg.hybrid_anchor_ratio = 0.875

    dev = torch.device(device)
    prec = pi05_model.config.precision
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False)
    if cfg.arch == "p2_m5_hybrid_surprise_gdn" and int(cfg.hybrid_canvas_tokens) == int(cfg.q_rope_offset or 512):
        for blk in wrapper.pi05.paligemma_with_expert.layers:
            if blk.mem_attn is not None and hasattr(blk.mem_attn, "k_rope_span_frames"):
                blk.mem_attn.k_rope_span_frames = False
    wrapper.eval()
    return DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=hop_mode)


def _load_deltaground_from_ckpt(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode_override: str = "l3",
):
    from .deltaground import DeltaGroundConfig, DeltaGroundEvalAdapter, DeltaGroundModel

    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_deltaground_adapter*`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    payload = raw.get("ema_deltaground") or raw.get("deltaground") or raw
    cfg = DeltaGroundConfig.from_dict(payload["deltaground_config"])
    tmpl_vocab = payload["template_vocab"]
    model = DeltaGroundModel(cfg, template_vocab=tmpl_vocab)
    if pi05_model.config.precision == "bfloat16":
        model.cast_bf16_keep_fp32_gates()
    model.load_state_dict(payload["state_dict"], strict=True)
    return DeltaGroundEvalAdapter(
        model,
        device=device,
        hop_mode=hop_mode_override,
        state_stats=payload.get("state_stats"),
    )


def build_deltaground_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
):
    """Default `L=3` looped multi-hop `DeltaGround-π0.5` adapter (`l1`/`l2` via `--hop-mode` also supported!)."""
    eff_hop = "l3" if hop_mode == "auto" else hop_mode
    return _load_deltaground_from_ckpt(ckpt_path, pi05_model, device=device, hop_mode_override=eff_hop)


def build_deltaground_adapter_l1(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
):
    return _load_deltaground_from_ckpt(ckpt_path, pi05_model, device=device, hop_mode_override="l1")


def build_deltaground_adapter_l2(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
):
    return _load_deltaground_from_ckpt(ckpt_path, pi05_model, device=device, hop_mode_override="l2")


def build_deltaground_adapter_l3(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
):
    return _load_deltaground_from_ckpt(ckpt_path, pi05_model, device=device, hop_mode_override="l3")


def build_deltaground_adapter_auto(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
):
    return _load_deltaground_from_ckpt(ckpt_path, pi05_model, device=device, hop_mode_override="auto")


class UnifiedGroundedHybridEvalAdapter:
    """Single-pass closed-loop adapter fusing `DeltaGround` (`128t` `<row_y, col_x>` VLM prefix) + `512t` `Hybrid` (`#313`).

    Shares ONE `select_write_indices` / `ingest_selected_frames` union pass (`0` duplicate SigLIP calls, `< 5 ms`
    adapter latency, `< 7.5 GiB` VRAM/worker, full `k=3` boundary-reset parity across both engines!).
    """

    def __init__(self, dg_adapter: Any, hyb_adapter: DeltaMemEvalAdapter) -> None:
        self.dg_adapter = dg_adapter
        self.hyb_adapter = hyb_adapter
        self.cfg = hyb_adapter.cfg
        self.device = hyb_adapter.device
        self.min_token_len: int = 64 if bool(hyb_adapter.cfg.use_continuous_vlm_ground) else 128
        self._pending_union_map: Optional[Tuple[List[int], List[int], List[int]]] = None
        self._last_cont_vlm_ground: Optional[Dict[str, Tensor]] = None

    def reset(self) -> None:
        self.dg_adapter.reset()
        self.hyb_adapter.reset()
        self._pending_union_map = None
        self._last_cont_vlm_ground = None

    @torch.no_grad()
    def select_write_indices(
        self,
        num_frames: int,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> List[int]:
        sel_dg = self.dg_adapter.select_write_indices(
            num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        sel_hyb = self.hyb_adapter.select_write_indices(
            num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        union = sorted(set(sel_dg) | set(sel_hyb))
        if union:
            pos = {orig_i: u_i for u_i, orig_i in enumerate(union)}
            dg_sub = [pos[i] for i in sel_dg]
            hyb_sub = [pos[i] for i in sel_hyb]
            self._pending_union_map = (union, dg_sub, hyb_sub)
        else:
            self._pending_union_map = None
        return union

    @torch.no_grad()
    def ingest_selected_frames(
        self,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        pend = self._pending_union_map
        self._pending_union_map = None
        if pend is None:
            return
        union, dg_sub, hyb_sub = pend
        if int(front_tokens_256.shape[0]) != len(union):
            raise ValueError(f"Expected {len(union)} union frames, got {int(front_tokens_256.shape[0])}")
        if dg_sub:
            di = torch.tensor(dg_sub, device=front_tokens_256.device, dtype=torch.long)
            w_dg = wrist_tokens_256[di.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
            self.dg_adapter.ingest_selected_frames(
                front_tokens_256[di],
                w_dg,
                np.asarray(raw_states)[dg_sub],
                np.asarray(norm_states)[dg_sub],
                task_goal=task_goal,
                is_demo=is_demo,
            )
        if hyb_sub:
            hi = torch.tensor(hyb_sub, device=front_tokens_256.device, dtype=torch.long)
            w_hyb = wrist_tokens_256[hi.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
            self.hyb_adapter.ingest_selected_frames(
                front_tokens_256[hi],
                w_hyb,
                np.asarray(raw_states)[hyb_sub],
                np.asarray(norm_states)[hyb_sub],
                task_goal=task_goal,
                is_demo=is_demo,
            )

    @torch.no_grad()
    def ingest_frames(
        self,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> None:
        t_in = int(front_tokens_256.shape[0])
        if t_in == 0:
            return
        union = self.select_write_indices(
            t_in, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        if not union:
            return
        ui = torch.tensor(union, device=front_tokens_256.device, dtype=torch.long)
        w_u = wrist_tokens_256[ui.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
        self.ingest_selected_frames(
            front_tokens_256[ui],
            w_u,
            np.asarray(raw_states)[union],
            np.asarray(norm_states)[union],
            task_goal=task_goal,
            is_demo=is_demo,
        )

    @torch.no_grad()
    def predict_subgoal(
        self,
        *,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        if bool(self.cfg.use_continuous_vlm_ground) and hasattr(self.dg_adapter, "predict_continuous_grounding"):
            self._last_cont_vlm_ground = self.dg_adapter.predict_continuous_grounding(
                front_tokens_256=front_tokens_256,
                wrist_tokens_256=wrist_tokens_256,
                raw_state=raw_state,
                task_goal=task_goal,
            )
            return None
        self._last_cont_vlm_ground = None
        return self.dg_adapter.predict_subgoal(
            front_tokens_256=front_tokens_256,
            wrist_tokens_256=wrist_tokens_256,
            raw_state=raw_state,
            norm_state=norm_state,
            task_goal=task_goal,
            step_idx=step_idx,
            exec_start_idx=exec_start_idx,
        )

    @torch.no_grad()
    def read_memory(
        self,
        *,
        prefix_hidden: Tensor | None = None,
        front_tokens_256: Tensor,
        wrist_tokens_256: Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
        cont_vlm_ground: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor | None, Tensor | None, Dict[str, Any]]:
        eff_cont = cont_vlm_ground if cont_vlm_ground is not None else self._last_cont_vlm_ground
        mem_tok, mem_mask, info = self.hyb_adapter.read_memory(
            prefix_hidden=prefix_hidden,
            front_tokens_256=front_tokens_256,
            wrist_tokens_256=wrist_tokens_256,
            raw_state=raw_state,
            norm_state=norm_state,
            task_goal=task_goal,
            step_idx=step_idx,
            exec_start_idx=exec_start_idx,
            cont_vlm_ground=eff_cont,
        )
        info["pred_subgoal"] = self.dg_adapter._last_pred_subgoal
        info["loop_depth"] = max(int(info.get("loop_depth", 1)), int(self.dg_adapter._last_loop_depth))
        info["cont_vlm_ground_active"] = bool(eff_cont is not None)
        info["boundary_reset_max"] = max(
            float(info.get("boundary_reset_max", 0.0)), float(self.dg_adapter._last_b_reset_max)
        )
        return mem_tok, mem_mask, info


def build_grounded_hybrid_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
    *,
    min_template_confidence: float = 0.0,
    abstain_ordinal_counting: bool = False,
    abstain_place_order: bool = True,
    abstain_temporal_button: Optional[bool] = None,
    dynamic_policy_lora_on_subgoal: Optional[bool] = None,
) -> UnifiedGroundedHybridEvalAdapter:
    """Build `UnifiedGroundedHybridEvalAdapter` (`P2-M7A`, `P2-M7B`, `P2-M7C`, `P2-M7D`, & `P2-M7B + P2-M9`) from a single bundled `.pt` checkpoint."""
    from .deltaground import DeltaGroundConfig, DeltaGroundEvalAdapter, DeltaGroundModel

    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_grounded_hybrid_adapter`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    ckpt_args = raw.get("args") if isinstance(raw, dict) else None
    is_m7d_ckpt = bool(
        (isinstance(raw, dict) and raw.get("is_m7d_dynamic_lora"))
        or (isinstance(ckpt_args, dict) and ckpt_args.get("freeze_all_except_vlm_lora"))
    )
    eff_dyn_lora = is_m7d_ckpt if dynamic_policy_lora_on_subgoal is None else bool(dynamic_policy_lora_on_subgoal)
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    cfg.hybrid_anchor_ratio = max(float(cfg.hybrid_anchor_ratio), 0.875)
    if int(cfg.mem_token_dim) >= 512:
        cfg.hybrid_num_keyframes = 32
        cfg.hybrid_canvas_tokens = 512
        cfg.q_rope_offset = 512

    dev = torch.device(device)
    prec = pi05_model.config.precision
    if (not eff_dyn_lora) and (not bool(cfg.use_continuous_vlm_ground)):
        pi05_model.config.max_token_len = max(int(pi05_model.config.max_token_len), 128)
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False)
    if int(cfg.hybrid_canvas_tokens) == int(cfg.q_rope_offset or 512):
        for blk in wrapper.pi05.paligemma_with_expert.layers:
            if blk.mem_attn is not None and hasattr(blk.mem_attn, "k_rope_span_frames"):
                blk.mem_attn.k_rope_span_frames = False
    wrapper.eval()
    hyb_hop_mode = (
        ("imitation_l2" if hop_mode in ("auto", "imitation_l2") else hop_mode)
        if bool(cfg.use_residual_hop2)
        else "l1"
    )
    hyb_adapter = DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=hyb_hop_mode)

    dg_vlm_payload = raw.get("ema_deltaground_vlm") or raw.get("deltaground_vlm")
    if dg_vlm_payload is not None:
        from .deltaground_vlm import (
            DeltaGroundVLMConfig,
            DeltaGroundVLMEvalAdapter,
            DeltaGroundVLMSystem,
            attach_base_vlm_lora,
            attach_reactive_lora,
        )

        dgv_cfg = DeltaGroundVLMConfig.from_dict(dg_vlm_payload["deltaground_vlm_config"])
        dgv_sys = DeltaGroundVLMSystem(dgv_cfg, wrapper.pi05, template_vocab=dg_vlm_payload["template_vocab"])
        dgv_sys.load_payload(dg_vlm_payload, strict=True)
        base_vlm_lora = raw.get("base_vlm_lora") if isinstance(raw, dict) else None
        non_sg_mode = "none"
        if isinstance(base_vlm_lora, dict) and len(base_vlm_lora) > 0:
            if attach_base_vlm_lora(wrapper.pi05, base_vlm_lora) > 0:
                non_sg_mode = "base_policy"

        has_reactive_m12 = False
        reac_pay = raw.get("reactive_m12_payload") if isinstance(raw, dict) else None
        if isinstance(reac_pay, dict):
            reac_delta = reac_pay.get("ema_delta") or reac_pay.get("model_delta") or reac_pay
            if isinstance(reac_delta, dict) and isinstance(reac_delta.get("pi05_delta"), dict):
                if attach_reactive_lora(wrapper.pi05, reac_delta["pi05_delta"]) > 0:
                    has_reactive_m12 = True
                    hyb_adapter.cfg.use_reactive_precision_mode = True
                    reac_mc = reac_delta.get("mem_core", {})
                    hop2_sub = {
                        k[len("hop2_refiner.") :]: v
                        for k, v in reac_mc.items()
                        if k.startswith("hop2_refiner.")
                    }
                    if hop2_sub:
                        from .memory_modules import ResidualMultiHopRefiner

                        r_hop2 = ResidualMultiHopRefiner(cfg).to(device=dev, dtype=wrapper.pi05.compute_dtype)
                        r_hop2.load_state_dict(hop2_sub, strict=True)
                        r_hop2.gamma_hop2.data = r_hop2.gamma_hop2.data.float()
                        r_hop2.eval()
                        hyb_adapter.reactive_hop2_refiner = r_hop2

        rest_mode = "policy" if wrapper.has_vlm_lora else "none"
        eff_temp_btn = True if abstain_temporal_button is None else bool(abstain_temporal_button)
        eff_conf = 0.0 if eff_dyn_lora else float(min_template_confidence)
        dgv_adapter = DeltaGroundVLMEvalAdapter(
            dgv_sys,
            restore_lora_mode=rest_mode,  # type: ignore[arg-type]
            fold_occlusion_twins=bool(dgv_cfg.v2_mode),
            stage_entry_latch=bool(dgv_cfg.v2_mode),
            min_template_confidence=eff_conf,
            abstain_ordinal_counting=bool(abstain_ordinal_counting) and (not eff_dyn_lora),
            abstain_place_order=bool(abstain_place_order),
            abstain_temporal_button=eff_temp_btn,
            dynamic_policy_lora_on_subgoal=eff_dyn_lora,
            non_subgoal_lora_mode=non_sg_mode,  # type: ignore[arg-type]
            enable_reactive_lora_routing=has_reactive_m12,
            state_stats=dg_vlm_payload.get("state_stats"),
        )
        return UnifiedGroundedHybridEvalAdapter(dgv_adapter, hyb_adapter)  # type: ignore[arg-type]

    dg_payload = raw.get("deltaground") or raw.get("ema_deltaground")
    if dg_payload is None:
        dg_fallback = Path(ckpt_path).resolve().parents[1] / "p2_m6_deltaground_v2_on100" / "step_001000_online.pt"
        if not dg_fallback.exists():
            dg_fallback = Path(ckpt_path).resolve().parents[1] / "p2_m6_deltaground_v1" / "step_001000_online.pt"
        dg_raw = torch.load(dg_fallback, map_location="cpu", weights_only=False)
        dg_payload = dg_raw.get("deltaground") or dg_raw.get("ema_deltaground") or dg_raw

    dg_cfg = DeltaGroundConfig.from_dict(dg_payload["deltaground_config"])
    dg_model = DeltaGroundModel(dg_cfg, template_vocab=dg_payload["template_vocab"])
    if prec == "bfloat16":
        dg_model.cast_bf16_keep_fp32_gates()
    dg_model.load_state_dict(dg_payload["state_dict"], strict=True)
    eff_hop = "l3" if hop_mode == "auto" else hop_mode
    eff_temp_btn_dg = False if abstain_temporal_button is None else bool(abstain_temporal_button)
    dg_adapter = DeltaGroundEvalAdapter(
        dg_model,
        device=dev,
        hop_mode=eff_hop,
        min_template_confidence=float(min_template_confidence),
        abstain_ordinal_counting=bool(abstain_ordinal_counting),
        abstain_place_order=bool(abstain_place_order),
        abstain_temporal_button=eff_temp_btn_dg,
        state_stats=dg_payload.get("state_stats"),
    )
    return UnifiedGroundedHybridEvalAdapter(dg_adapter, hyb_adapter)


def build_grounded_hybrid_adapter_selective(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle `#338`/`#343`/`#355` Selective-Abstention Gate (`conf >= 0.60` + 6-task on `M9`/`M7C`, 4-task on `dgv2`)."""
    return build_grounded_hybrid_adapter(
        ckpt_path,
        pi05_model,
        device=device,
        hop_mode=hop_mode,
        min_template_confidence=0.60,
        abstain_ordinal_counting=True,
        abstain_place_order=True,
    )


def build_grounded_hybrid_adapter_m7c_6task(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle `#355` `P2-M7C` 6-task episode-level abstention (`conf=0.0` -> zero mid-episode 2-container `None` toggling!)."""
    return build_grounded_hybrid_adapter(
        ckpt_path,
        pi05_model,
        device=device,
        hop_mode=hop_mode,
        min_template_confidence=0.0,
        abstain_ordinal_counting=True,
        abstain_place_order=True,
        abstain_temporal_button=True,
    )


def build_grounded_hybrid_adapter_selective_keep_po(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle `#343` Selective-Abstention Gate keeping `VideoPlaceOrder` grounded (`abstain_place_order=False`)."""
    return build_grounded_hybrid_adapter(
        ckpt_path,
        pi05_model,
        device=device,
        hop_mode=hop_mode,
        min_template_confidence=0.60,
        abstain_ordinal_counting=True,
        abstain_place_order=False,
    )


def build_grounded_hybrid_adapter_ordinal_only(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle `#338`/`#341` pure ordinal-counting abstention (`conf=0.0`, keeps 100% of `Permanence` `VideoUnmask 80%`!)."""
    return build_grounded_hybrid_adapter(
        ckpt_path,
        pi05_model,
        device=device,
        hop_mode=hop_mode,
        min_template_confidence=0.0,
        abstain_ordinal_counting=True,
    )


def build_grounded_hybrid_adapter_conf65(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle confidence-gated (`max_prob >= 0.65`, else `subgoal=None` `Hybrid` fallback!) `#337` entrypoint."""
    return build_grounded_hybrid_adapter(
        ckpt_path, pi05_model, device=device, hop_mode=hop_mode, min_template_confidence=0.65
    )


def build_grounded_hybrid_adapter_conf50(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-oracle confidence-gated (`max_prob >= 0.50`, else `subgoal=None` `Hybrid` fallback!) `#337` entrypoint."""
    return build_grounded_hybrid_adapter(
        ckpt_path, pi05_model, device=device, hop_mode=hop_mode, min_template_confidence=0.50
    )


def build_deltaground_vlm_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> Any:
    """Standalone `P2-M9: DeltaGround-VLM` adapter sharing `symbolic_grounded_subgoal_pt`'s `PaliGemma-2B` (`0` extra VLM!)."""
    del hop_mode
    from .deltaground_vlm import DeltaGroundVLMConfig, DeltaGroundVLMEvalAdapter, DeltaGroundVLMSystem

    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_deltaground_vlm_adapter`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    payload = raw.get("ema_deltaground_vlm") or raw.get("deltaground_vlm") or raw
    dev = torch.device(device)
    pi05_model = pi05_model.to(dev)
    pi05_model.config.max_token_len = max(int(pi05_model.config.max_token_len), 128)
    dgv_cfg = DeltaGroundVLMConfig.from_dict(payload["deltaground_vlm_config"])
    dgv_sys = DeltaGroundVLMSystem(dgv_cfg, pi05_model, template_vocab=payload["template_vocab"])
    dgv_sys.load_payload(payload, strict=True)
    return DeltaGroundVLMEvalAdapter(
        dgv_sys,
        restore_lora_mode="none",
        fold_occlusion_twins=bool(dgv_cfg.v2_mode),
        stage_entry_latch=bool(dgv_cfg.v2_mode),
        state_stats=payload.get("state_stats"),
    )


def build_deltaground_vlm_decoupled_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
    *,
    stage_ckpt_path: str | Path | None = None,
    pointer_ckpt_path: str | Path | None = None,
) -> Any:
    """`#376` Action A: Zero-retraining Decoupled `v1` Stage (`98.8%` transitions, `0` timeouts) + `v2b` Neutral-Arm `<y, x>` Pointer (`0 MiB` extra VLM!)."""
    del hop_mode
    from .deltaground_vlm import (
        DecoupledStagePointerVLMAdapter,
        DeltaGroundVLMConfig,
        DeltaGroundVLMEvalAdapter,
        DeltaGroundVLMSystem,
        attach_dg_as_base_vlm_lora,
    )

    if ckpt_path is None and (stage_ckpt_path is None or pointer_ckpt_path is None):
        raise ValueError("`--memory-ckpt` (or explicit `stage_ckpt_path` + `pointer_ckpt_path`) must be provided")

    if stage_ckpt_path is None or pointer_ckpt_path is None:
        assert ckpt_path is not None
        p_in = Path(ckpt_path).resolve()
        probe = torch.load(p_in, map_location="cpu", weights_only=False)
        probe_pay = probe.get("ema_deltaground_vlm") or probe.get("deltaground_vlm") or probe
        is_v2 = bool(probe_pay["deltaground_vlm_config"].get("v2_mode", False))
        root_dir = p_in.parents[1]
        if is_v2:
            pointer_ckpt_path = pointer_ckpt_path or p_in
            stage_ckpt_path = stage_ckpt_path or (root_dir / "p2_m9_sym_deltaground_vlm" / "best_unseen_ema.pt")
        else:
            stage_ckpt_path = stage_ckpt_path or p_in
            pointer_ckpt_path = pointer_ckpt_path or (root_dir / "p2_m9v2b_sym_shortcut_free" / "best_unseen_ema.pt")

    st_raw = torch.load(Path(stage_ckpt_path), map_location="cpu", weights_only=False)
    st_pay = st_raw.get("ema_deltaground_vlm") or st_raw.get("deltaground_vlm") or st_raw
    pt_raw = torch.load(Path(pointer_ckpt_path), map_location="cpu", weights_only=False)
    pt_pay = pt_raw.get("ema_deltaground_vlm") or pt_raw.get("deltaground_vlm") or pt_raw

    dev = torch.device(device)
    pi05_model = pi05_model.to(dev)
    pi05_model.config.max_token_len = max(int(pi05_model.config.max_token_len), 128)

    st_cfg = DeltaGroundVLMConfig.from_dict(st_pay["deltaground_vlm_config"])
    pt_cfg = DeltaGroundVLMConfig.from_dict(pt_pay["deltaground_vlm_config"])

    st_sys = DeltaGroundVLMSystem(
        st_cfg, pi05_model, template_vocab=st_pay["template_vocab"], skip_lora_inject=False, vlm_lora_mode="dg"
    )
    st_sys.load_payload(st_pay, strict=True)

    pt_sys = DeltaGroundVLMSystem(
        pt_cfg, pi05_model, template_vocab=pt_pay["template_vocab"], skip_lora_inject=True, vlm_lora_mode="base_policy"
    )
    pt_sys.head.load_state_dict(pt_pay["head_state_dict"], strict=True)
    n_att = attach_dg_as_base_vlm_lora(
        pi05_model,
        pt_pay["dg_lora_state_dict"],
        dg_rank=pt_cfg.dg_lora_rank,
        dg_alpha=pt_cfg.dg_lora_alpha,
    )
    assert n_att > 0, f"Failed to attach v2b dg_lora into base_lora slots from {pointer_ckpt_path}"

    st_adapter = DeltaGroundVLMEvalAdapter(
        st_sys,
        restore_lora_mode="none",
        fold_occlusion_twins=True,
        stage_entry_latch=False,
        state_stats=st_pay.get("state_stats"),
    )
    pt_adapter = DeltaGroundVLMEvalAdapter(
        pt_sys,
        restore_lora_mode="none",
        fold_occlusion_twins=True,
        stage_entry_latch=False,
        state_stats=pt_pay.get("state_stats"),
    )
    return DecoupledStagePointerVLMAdapter(st_adapter, pt_adapter)


def _resolve_host_perceptual_pi05(spec_pi05_model: Pi05Model) -> Pi05Model:
    vt = getattr(spec_pi05_model, "vision_tower", None)
    ref = getattr(vt, "_host_perceptual_pi05", None) if vt is not None else None
    host = ref() if callable(ref) else None
    if host is None:
        raise ValueError(
            "`build_deltaground_vlm_cross_backbone_*` requires primary `perceptual_framesamp_modul_pt` "
            "policy (`v3_s250` / `P2-M7B`) to share `vision_tower` (`eval_pi05.py --specialist-ckpt-dir`)."
        )
    return host


def build_deltaground_vlm_cross_backbone_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> Any:
    """Zero-retraining `P2-M10` Cross-Backbone Specialist Adapter (`0 MiB` extra VLM, `0.0`-diff `v3_s250`!).

    Mounts `P2-M9-Hyb-v2c` (`p2_m9v2c_hyb_gsstd/best_unseen_ema.pt`: `ENTRY 66.4% [18.21 px]`,
    `ButtonUnmask 83%`, `PickHighlight 79%`, `ButtonUnmaskSwap 70%`, `VideoUnmaskSwap 60%`, `VideoRepick 55%`)
    directly onto the ALREADY-loaded primary `perceptual_framesamp_modul_pt` VLM via `DualSwitchableLoRALinear`
    (`restore_lora_mode="policy"`, leaving `max_token_len=64` untouched for `0.0`-diff `v3_s250` on 10 base tasks!),
    and feeds its `100%` `gs_std` `<y, x>` subgoal strings into `symbolic_grounded_subgoal_pt` (`93.3–95.2%` ceiling!).
    """
    del hop_mode
    from .deltaground_vlm import DeltaGroundVLMConfig, DeltaGroundVLMEvalAdapter, DeltaGroundVLMSystem

    if ckpt_path is None:
        raise ValueError("`--specialist-memory-ckpt` must be provided to `build_deltaground_vlm_cross_backbone_adapter`")
    host_hyb_pi05 = _resolve_host_perceptual_pi05(pi05_model)
    dev = torch.device(device)
    pi05_model = pi05_model.to(dev)
    pi05_model.config.max_token_len = max(int(pi05_model.config.max_token_len), 128)
    host_hyb_pi05 = host_hyb_pi05.to(dev)

    has_policy_lora = any(
        (getattr(m, "lora_a", None) is not None) or (getattr(m, "lora_A", None) is not None)
        for m in host_hyb_pi05.paligemma_with_expert.modules()
    )
    has_base_lora = any(
        getattr(m, "base_lora_a", None) is not None for m in host_hyb_pi05.paligemma_with_expert.modules()
    )
    restore_mode = "policy" if has_policy_lora else ("base_policy" if has_base_lora else "none")

    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    payload = raw.get("ema_deltaground_vlm") or raw.get("deltaground_vlm") or raw
    dgv_cfg = DeltaGroundVLMConfig.from_dict(payload["deltaground_vlm_config"])
    dgv_sys = DeltaGroundVLMSystem(dgv_cfg, host_hyb_pi05, template_vocab=payload["template_vocab"])
    dgv_sys.load_payload(payload, strict=True)
    return DeltaGroundVLMEvalAdapter(
        dgv_sys,
        restore_lora_mode=restore_mode,  # type: ignore[arg-type]
        fold_occlusion_twins=bool(dgv_cfg.v2_mode),
        stage_entry_latch=bool(dgv_cfg.v2_mode),
        state_stats=payload.get("state_stats"),
    )


def build_deltaground_vlm_cross_backbone_decoupled_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
    *,
    stage_ckpt_path: str | Path | None = None,
    pointer_ckpt_path: str | Path | None = None,
) -> Any:
    """Cross-Backbone Decoupled `v1` Sym Stage (`98.8%` transitions on `Symbolic-VLM`) + `v2c` Hyb Pointer (`18.21 px` on `Perceptual-VLM`)!"""
    del hop_mode
    from .deltaground_vlm import (
        DecoupledStagePointerVLMAdapter,
        DeltaGroundVLMConfig,
        DeltaGroundVLMEvalAdapter,
        DeltaGroundVLMSystem,
    )

    if ckpt_path is None and pointer_ckpt_path is None:
        raise ValueError("`--specialist-memory-ckpt` (`p2_m9v2c_hyb_gsstd/best_unseen_ema.pt`) must be provided")
    host_hyb_pi05 = _resolve_host_perceptual_pi05(pi05_model)
    p_ptr = Path(pointer_ckpt_path or ckpt_path).resolve()  # type: ignore[arg-type]
    if stage_ckpt_path is not None:
        p_stg = Path(stage_ckpt_path).resolve()
    else:
        cand_w15 = p_ptr.parents[1] / "p2_m9_sym_stage_gsstd_v1" / "best_unseen_ema.pt"
        p_stg = cand_w15 if cand_w15.exists() else (p_ptr.parents[1] / "p2_m9_sym_deltaground_vlm" / "best_unseen_ema.pt")

    dev = torch.device(device)
    pi05_model = pi05_model.to(dev)
    pi05_model.config.max_token_len = max(int(pi05_model.config.max_token_len), 128)
    host_hyb_pi05 = host_hyb_pi05.to(dev)

    has_policy_lora = any(
        (getattr(m, "lora_a", None) is not None) or (getattr(m, "lora_A", None) is not None)
        for m in host_hyb_pi05.paligemma_with_expert.modules()
    )
    has_base_lora = any(
        getattr(m, "base_lora_a", None) is not None for m in host_hyb_pi05.paligemma_with_expert.modules()
    )
    restore_mode = "policy" if has_policy_lora else ("base_policy" if has_base_lora else "none")

    st_raw = torch.load(p_stg, map_location="cpu", weights_only=False)
    st_pay = st_raw.get("ema_deltaground_vlm") or st_raw.get("deltaground_vlm") or st_raw
    pt_raw = torch.load(p_ptr, map_location="cpu", weights_only=False)
    pt_pay = pt_raw.get("ema_deltaground_vlm") or pt_raw.get("deltaground_vlm") or pt_raw

    st_cfg = DeltaGroundVLMConfig.from_dict(st_pay["deltaground_vlm_config"])
    pt_cfg = DeltaGroundVLMConfig.from_dict(pt_pay["deltaground_vlm_config"])

    st_sys = DeltaGroundVLMSystem(st_cfg, pi05_model, template_vocab=st_pay["template_vocab"])
    st_sys.load_payload(st_pay, strict=True)

    pt_sys = DeltaGroundVLMSystem(pt_cfg, host_hyb_pi05, template_vocab=pt_pay["template_vocab"])
    pt_sys.load_payload(pt_pay, strict=True)

    st_adapter = DeltaGroundVLMEvalAdapter(
        st_sys,
        restore_lora_mode="none",
        fold_occlusion_twins=True,
        stage_entry_latch=False,
        state_stats=st_pay.get("state_stats"),
    )
    pt_adapter = DeltaGroundVLMEvalAdapter(
        pt_sys,
        restore_lora_mode=restore_mode,  # type: ignore[arg-type]
        fold_occlusion_twins=True,
        stage_entry_latch=False,
        state_stats=pt_pay.get("state_stats"),
    )
    return DecoupledStagePointerVLMAdapter(st_adapter, pt_adapter)


def build_grounded_hybrid_adapter_m7d_dynamic(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """Zero-drift `#361(2)` `P2-M7D` Dynamic-LoRA Adapter (`"policy"` on 6 `<y,x>` tasks, `"none"` = `0.0`-diff `v3_s250` on 10 tasks!)."""
    return build_grounded_hybrid_adapter(
        ckpt_path,
        pi05_model,
        device=device,
        hop_mode=hop_mode,
        min_template_confidence=0.0,
        abstain_ordinal_counting=False,
        dynamic_policy_lora_on_subgoal=True,
    )


def build_continuous_vlm_grounded_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "auto",
) -> UnifiedGroundedHybridEvalAdapter:
    """`#395(2)` `P2-M11` Single-Checkpoint Continuous-VLM-Grounded Adapter (`max_token_len=64` ALWAYS, `0.0` `v3_s250` diff on 10 base tasks!)."""
    return build_grounded_hybrid_adapter_m7d_dynamic(ckpt_path, pi05_model, device=device, hop_mode=hop_mode)


def build_reactive_precision_adapter(
    ckpt_path: str | Path | None,
    pi05_model: Pi05Model,
    device: str | torch.device = "cuda",
    hop_mode: str = "l1",
) -> DeltaMemEvalAdapter:
    """`#405` `P2-M12` Zero-Oracle Reactive/Precision Adapter (`StopCube` `excl=1/stride=1`, `InsertPeg` `L=2 climax`, `0.0` `v3_s250` diff on 14 tasks!)."""
    from .deltaground_vlm import attach_base_vlm_lora, attach_reactive_lora, set_vlm_lora_mode

    if ckpt_path is None:
        raise ValueError("`--memory-ckpt` must be provided to `build_reactive_precision_adapter`")
    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    delta_payload = raw.get("ema_delta") or raw.get("model_delta") or raw
    cfg = Pi05MemConfig.from_dict(delta_payload["pi05_mem_config"])
    cfg.use_reactive_precision_mode = True
    cfg.hybrid_anchor_ratio = max(float(cfg.hybrid_anchor_ratio), 0.875)
    if int(cfg.mem_token_dim) >= 512:
        cfg.hybrid_num_keyframes = 32
        cfg.hybrid_canvas_tokens = 512
        cfg.q_rope_offset = 512

    dev = torch.device(device)
    prec = pi05_model.config.precision
    wrapper = DeltaMemPi05Policy(cfg, pi05_model=pi05_model, device=dev, precision=prec)
    # Load ONLY `mem_core` (`hop2_refiner` + frozen `v3_s250` `L=1` weights) without injecting direct `"policy"` LoRA,
    # then attach `"reactive_policy"` (`P2-M12`) + `"base_policy"` (`v3_s250` `252` tensors) cleanly as switchable buffers!
    wrapper.load_delta_state_dict(delta_payload, strict_mem=False, skip_lora=True)
    if int(cfg.hybrid_canvas_tokens) == int(cfg.q_rope_offset or 512):
        for blk in wrapper.pi05.paligemma_with_expert.layers:
            if blk.mem_attn is not None and hasattr(blk.mem_attn, "k_rope_span_frames"):
                blk.mem_attn.k_rope_span_frames = False

    non_reac_mode = "none"
    base_vlm_lora = raw.get("base_vlm_lora") if isinstance(raw, dict) else None
    if isinstance(base_vlm_lora, dict) and len(base_vlm_lora) > 0:
        if attach_base_vlm_lora(wrapper.pi05, base_vlm_lora) > 0:
            non_reac_mode = "base_policy"
    attach_reactive_lora(wrapper.pi05, delta_payload.get("pi05_delta", {}))
    set_vlm_lora_mode(wrapper.pi05, non_reac_mode)  # type: ignore[arg-type]
    wrapper.eval()

    reac_hop2_sd = raw.get("reactive_hop2_refiner") if isinstance(raw, dict) else None
    eff_hop = (
        ("imitation_l2" if hop_mode in ("auto", "l1") else hop_mode)
        if (isinstance(reac_hop2_sd, dict) and len(reac_hop2_sd) > 0 and bool(cfg.use_residual_hop2))
        else hop_mode
    )
    ad = DeltaMemEvalAdapter(cfg, wrapper.mem_core, wrapper.pi05, device=dev, hop_mode=eff_hop)
    if isinstance(reac_hop2_sd, dict) and len(reac_hop2_sd) > 0:
        from .memory_modules import ResidualMultiHopRefiner

        r_hop2 = ResidualMultiHopRefiner(cfg).to(device=dev, dtype=wrapper.pi05.compute_dtype)
        r_hop2.load_state_dict(reac_hop2_sd, strict=True)
        r_hop2.gamma_hop2.data = r_hop2.gamma_hop2.data.float()
        r_hop2.eval()
        ad.reactive_hop2_refiner = r_hop2
    ad.switch_reactive_lora = True
    ad.non_reactive_lora_mode = non_reac_mode
    ad.dynamic_policy_lora_on_subgoal = True
    ad.reset()
    return ad


class PromptGatedSpecialistPolicy:
    """Zero-oracle `#361(1)` `P2-M10` Public-Prompt Specialist Policy (`< 7.5 GiB` with per-episode swap, `< 14.2 GiB` co-resident).

    Routes strictly from the public user instruction `task_goal` (`should_route_to_symbolic_specialist`, `0` oracle!):
      - `True` (6 `<row_y, col_x>` reveal/highlight/repick tasks: `Permanence` 4 tasks + `PickHighlight` + `VideoRepick`)
        -> `sym_policy` (`symbolic_grounded_subgoal_pt` [`93.3–95.2%` ceiling] + `P2-M9-Sym` `DeltaGround-VLM`).
      - `False` (10 continuous/counting/order/imitation tasks)
        -> `hyb_policy` (`perceptual_framesamp_modul_pt` + `v3_s250` [`30/50 = 60.0%`] or `P2-M7B s300 ordinal_only`).
    """

    def __init__(
        self,
        sym_policy: Any,
        hyb_policy: Any,
        *,
        device: str | torch.device = "cuda",
        offload_inactive_to_cpu: bool = True,
    ) -> None:
        self.sym_policy = sym_policy
        self.hyb_policy = hyb_policy
        self.device = torch.device(device)
        self.offload_inactive_to_cpu = bool(offload_inactive_to_cpu) and (self.device.type == "cuda")
        self._active_kind: Literal["sym", "hyb"] = "hyb"
        self._suppress_reset = False
        if self.offload_inactive_to_cpu:
            self._move_policy(self.sym_policy, torch.device("cpu"))
            self._move_policy(self.hyb_policy, self.device)
        self.model = self.hyb_policy.model

    @staticmethod
    def _move_policy(pol: Any, dev: torch.device) -> None:
        pol.device = dev
        pol.model.to(dev)
        ad = getattr(pol, "memory_adapter", None)
        if ad is not None:
            if hasattr(ad, "to_device"):
                ad.to_device(dev)
            else:
                ad.device = dev
                for attr in ("mem_core", "model", "head"):
                    sub = getattr(ad, attr, None)
                    if isinstance(sub, nn.Module):
                        sub.to(dev)
                for sub_ad_name in ("dg_adapter", "hyb_adapter", "stage_adapter", "pointer_adapter"):
                    sub_ad = getattr(ad, sub_ad_name, None)
                    if sub_ad is not None:
                        if hasattr(sub_ad, "to_device"):
                            sub_ad.to_device(dev)
                        else:
                            sub_ad.device = dev
                            for attr in ("mem_core", "model", "head"):
                                sub = getattr(sub_ad, attr, None)
                                if isinstance(sub, nn.Module):
                                    sub.to(dev)

    def _activate(self, task_goal: str) -> Any:
        from .deltaground import should_route_to_symbolic_specialist

        want: Literal["sym", "hyb"] = "sym" if should_route_to_symbolic_specialist(task_goal) else "hyb"
        if want != self._active_kind:
            if self.offload_inactive_to_cpu:
                old_p = self.sym_policy if self._active_kind == "sym" else self.hyb_policy
                new_p = self.sym_policy if want == "sym" else self.hyb_policy
                self._move_policy(old_p, torch.device("cpu"))
                torch.cuda.empty_cache()
                self._move_policy(new_p, self.device)
            self._active_kind = want
            self.model = (self.sym_policy if want == "sym" else self.hyb_policy).model
        return self.sym_policy if want == "sym" else self.hyb_policy

    @property
    def suppress_reset(self) -> bool:
        return self._suppress_reset

    @suppress_reset.setter
    def suppress_reset(self, val: bool) -> None:
        self._suppress_reset = bool(val)
        self.sym_policy.suppress_reset = bool(val)
        self.hyb_policy.suppress_reset = bool(val)

    def reset(self) -> None:
        if self._suppress_reset:
            return
        self.force_reset()

    def force_reset(self) -> None:
        self.sym_policy.force_reset()
        self.hyb_policy.force_reset()

    def set_rng_seed(self, seed: int) -> None:
        if hasattr(self.sym_policy, "set_rng_seed"):
            self.sym_policy.set_rng_seed(seed)
        if hasattr(self.hyb_policy, "set_rng_seed"):
            self.hyb_policy.set_rng_seed(seed)

    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray] | None,
        state_list: Sequence[np.ndarray],
        *,
        task_goal: str = "",
    ) -> None:
        pol = self._activate(task_goal)
        pol.ingest_demo_frames(front_rgb_list, wrist_rgb_list, state_list, task_goal=task_goal)

    def predict_chunk(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        task_goal: str,
        *,
        step_idx: int = 0,
        exec_start_idx: int = 0,
    ) -> Any:
        pol = self._activate(task_goal)
        return pol.predict_chunk(
            front_rgb,
            wrist_rgb,
            state,
            task_goal,
            step_idx=step_idx,
            exec_start_idx=exec_start_idx,
        )


def build_m10_specialist_policy(
    *,
    sym_ckpt_dir: str | Path = "checkpoints/phase2/init/symbolic_grounded_subgoal_pt",
    m9_sym_ckpt: str | Path = "checkpoints/phase2/p2_m9_sym_deltaground_vlm/best_unseen_ema.pt",
    hyb_ckpt_dir: str | Path = "checkpoints/phase2/init/perceptual_framesamp_modul_pt",
    hyb_mem_ckpt: str | Path = "checkpoints/phase2/p2_m5_hybrid_v3_stageB/step_000250_ema.pt",
    hyb_adapter_fn: str = "build_eval_adapter",
    device: str = "cuda",
    offload_inactive_to_cpu: bool = True,
    num_inference_steps: int = 10,
) -> PromptGatedSpecialistPolicy:
    """1-call factory building `P2-M10` Zero-Oracle Public-Prompt Specialist Policy (`#361(1)`)."""
    from pi05 import Pi05RoboMMEPolicy

    init_sym_dev = "cpu" if (offload_inactive_to_cpu and str(device).startswith("cuda")) else device
    sym_pol = Pi05RoboMMEPolicy(
        sym_ckpt_dir,
        device=init_sym_dev,
        stats_variant="mme_vla",
        num_inference_steps=num_inference_steps,
    )
    sym_pol.memory_adapter = build_deltaground_vlm_adapter(m9_sym_ckpt, sym_pol.model, device=init_sym_dev)

    hyb_pol = Pi05RoboMMEPolicy(
        hyb_ckpt_dir,
        device=device,
        stats_variant="framesamp_modul",
        num_inference_steps=num_inference_steps,
    )
    hyb_builder = globals()[hyb_adapter_fn]
    hyb_pol.memory_adapter = hyb_builder(hyb_mem_ckpt, hyb_pol.model, device=device)
    return PromptGatedSpecialistPolicy(
        sym_pol,
        hyb_pol,
        device=device,
        offload_inactive_to_cpu=offload_inactive_to_cpu,
    )


def build_m10_decoupled_specialist_policy(
    *,
    sym_ckpt_dir: str | Path = "checkpoints/phase2/init/symbolic_grounded_subgoal_pt",
    m9_stage_ckpt: str | Path = "checkpoints/phase2/p2_m9_sym_deltaground_vlm/best_unseen_ema.pt",
    m9_pointer_ckpt: str | Path = "checkpoints/phase2/p2_m9v2b_sym_shortcut_free/best_unseen_ema.pt",
    hyb_ckpt_dir: str | Path = "checkpoints/phase2/init/perceptual_framesamp_modul_pt",
    hyb_mem_ckpt: str | Path = "checkpoints/phase2/p2_m5_hybrid_v3_stageB/step_000250_ema.pt",
    hyb_adapter_fn: str = "build_eval_adapter",
    device: str = "cuda",
    offload_inactive_to_cpu: bool = True,
    num_inference_steps: int = 10,
) -> PromptGatedSpecialistPolicy:
    """`#376` Action A 1-call factory: `v3_s250` (`32/50`) + `Symbolic-VLM` driven by `v1` Stage (`98.8%`) + `v2b` Neutral-Arm Pointer (`17.9 px`)."""
    from pi05 import Pi05RoboMMEPolicy

    init_sym_dev = "cpu" if (offload_inactive_to_cpu and str(device).startswith("cuda")) else device
    sym_pol = Pi05RoboMMEPolicy(
        sym_ckpt_dir,
        device=init_sym_dev,
        stats_variant="mme_vla",
        num_inference_steps=num_inference_steps,
    )
    sym_pol.memory_adapter = build_deltaground_vlm_decoupled_adapter(
        m9_pointer_ckpt,
        sym_pol.model,
        device=init_sym_dev,
        stage_ckpt_path=m9_stage_ckpt,
        pointer_ckpt_path=m9_pointer_ckpt,
    )

    hyb_pol = Pi05RoboMMEPolicy(
        hyb_ckpt_dir,
        device=device,
        stats_variant="framesamp_modul",
        num_inference_steps=num_inference_steps,
    )
    hyb_builder = globals()[hyb_adapter_fn]
    hyb_pol.memory_adapter = hyb_builder(hyb_mem_ckpt, hyb_pol.model, device=device)
    return PromptGatedSpecialistPolicy(
        sym_pol,
        hyb_pol,
        device=device,
        offload_inactive_to_cpu=offload_inactive_to_cpu,
    )
