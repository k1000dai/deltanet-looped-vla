"""`StatefulPolicy` Rollout Wrapper & 1-Command Evaluator for `SmolVLA-DeltaLoop` (`U6`) Checkpoints.

Bridges any saved `SmolVLADeltaLoopPolicy` `.pt` checkpoint directly to `src/robomme_eval/`
(`evaluate_policy`, `ChallengePolicyAdapter`) with:
- **Zero Extra Vision Encoder in VRAM (`< 1.9 GB` VRAM/Worker on Local RTX 5080):**
  Reuses `model.encode_images_to_vlm_tokens` (`512x512 -> (64, 960)` frozen `SmolVLM2` ViT +
  `4x4` PixelShuffle Connector), enabling `5x` parallel ManiSkill3 workers on a 16 GB RTX 5080!
- **Strict A3 Video-Demo Cutoff & U5 Adaptive Striding (`< 2e-4` Streaming vs Offline Parity):**
  Encodes ONLY strided written frames (`write_stride=4`, or `long_demo_stride=8` when
  `N_demo > 256` + `obs_horizon=2` last exec frame), preserving 100% of conditioning video demo
  frames at `step_idx == exec_start_idx` and supporting `--distractor-sessions k` stress tests.
- **Single-Prefill VLM Prefix KV Reuse Across All `10` Euler Steps (`t = 1.0 -> 0.0`):**
  Runs the 16-layer `SmolVLM2` prefix prefill (`L_prefix = 177`) ONCE per chunk (`~9 ms`) and
  reuses `past_key_values` + `C_t^{(L_t-1)}` across all `10` Euler steps (`~28 ms`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from robomme_eval.interface import (
        MME_VLA_NORM_STATS,
        PolicyChunkOutput,
        ROBOMME_TASKS,
        StatefulPolicy,
        to_uint8_rgb,
    )
except ImportError:
    from src.robomme_eval.interface import (
        MME_VLA_NORM_STATS,
        PolicyChunkOutput,
        ROBOMME_TASKS,
        StatefulPolicy,
        to_uint8_rgb,
    )

from .deltaloop_vla import run_dual_backend_gdn
from .smolvla_deltaloop import (
    MeanStdDeltaNormalizer,
    SmolVLADeltaLoopConfig,
    SmolVLADeltaLoopPolicy,
    cast_smolvla_precision,
    load_smolvla_tokenizer,
    tokenize_smolvla_prompts,
)


class SmolVLARoboMMEWrapper(StatefulPolicy):
    """Stateful ManiSkill3/SAPIEN evaluation wrapper for any `SmolVLADeltaLoopPolicy` checkpoint."""

    def __init__(
        self,
        checkpoint_or_model: str | Path | SmolVLADeltaLoopPolicy,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        normalizer: Optional[MeanStdDeltaNormalizer] = None,
        tokenizer: Optional[Any] = None,
        vision_token_fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_loops: Optional[int] = None,
        halt_mode: Literal["fixed", "learned_gate", "coda_conv"] = "learned_gate",
        halt_threshold: float = 0.5,
        coda_conv_tau: float = 0.08,
        ode_steps: Optional[int] = None,
        force_pytorch_gdn: bool = False,
        seed: int = 7,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_loops = num_loops
        self.halt_mode: Literal["fixed", "learned_gate", "coda_conv"] = halt_mode
        self.halt_threshold = float(halt_threshold)
        self.coda_conv_tau = float(coda_conv_tau)
        self.force_pytorch_gdn = bool(force_pytorch_gdn)
        self._vision_token_fn = vision_token_fn

        ckpt_meta: Dict[str, Any] = {}
        if isinstance(checkpoint_or_model, SmolVLADeltaLoopPolicy):
            self.model = cast_smolvla_precision(checkpoint_or_model, device=self.device, dtype=self.dtype)
            self.cfg = self.model.cfg
        else:
            ckpt_path = Path(checkpoint_or_model)
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_meta = payload
            self.cfg = SmolVLADeltaLoopConfig.from_dict(payload["model_config"])
            self.model = SmolVLADeltaLoopPolicy(self.cfg)
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
            cast_smolvla_precision(self.model, device=self.device, dtype=self.dtype)

        self.ode_steps = int(self.cfg.num_ode_steps if ode_steps is None else ode_steps)
        self.model.eval()
        self.model.requires_grad_(False)

        if self.halt_mode == "learned_gate" and self.model.halting_gate is None:
            self.halt_mode = "fixed"

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            norm_stats = ckpt_meta.get("norm_stats", MME_VLA_NORM_STATS)
            self.normalizer = MeanStdDeltaNormalizer(stats=norm_stats)

        self.tokenizer = tokenizer if tokenizer is not None else load_smolvla_tokenizer(self.cfg.vlm_model_name)
        self._prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        self._frame_cursor: int = 0
        self._demo_frames_count: int = 0
        self._pending_strided: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        self._conv_ring: List[torch.Tensor] = []
        self._S1: Optional[torch.Tensor] = None
        self._occ1: Optional[torch.Tensor] = None
        self._S2: Optional[torch.Tensor] = None
        self._occ2: Optional[torch.Tensor] = None
        self._framesamp_committed: List[torch.Tensor] = []
        self._framesamp_demo_count: int = 0
        self._prev_written_front_tok: Optional[torch.Tensor] = None
        self._prev_exec_obs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        self.reset()

    def set_seed(self, seed: int) -> None:
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    def reset(self) -> None:
        self._frame_cursor = 0
        self._demo_frames_count = 0
        self._pending_strided.clear()
        self._conv_ring.clear()
        self._framesamp_committed.clear()
        self._framesamp_demo_count = 0
        self._prev_written_front_tok = None
        self._prev_exec_obs = None

        if self.model.uses_gdn:
            H = self.cfg.gdn_heads
            K = self.cfg.gdn_d_k
            Vh = self.cfg.gdn_d_v // 2
            self._S1 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ1 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
            self._S2 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ2 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
        else:
            self._S1 = self._occ1 = self._S2 = self._occ2 = None

    def _encode_views(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._vision_token_fn is not None:
            f_tok, w_tok = self._vision_token_fn(front_rgb, wrist_rgb)
            return f_tok.to(device=self.device, dtype=self.dtype), w_tok.to(device=self.device, dtype=self.dtype)
        f_tok, w_tok = self.model.encode_images_to_vlm_tokens(front_rgb, wrist_rgb, batch_size=64)
        return f_tok.to(device=self.device, dtype=self.dtype), w_tok.to(device=self.device, dtype=self.dtype)

    def _tokenize_prompt_cached(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(prompt)
        if key not in self._prompt_cache:
            ids, mask = tokenize_smolvla_prompts(
                self.tokenizer,
                [key],
                max_length=self.cfg.tokenizer_max_length,
                device=self.device,
            )
            self._prompt_cache[key] = (ids, mask)
        return self._prompt_cache[key]

    @torch.no_grad()
    def _stream_slot_conv_single(self, raw_slot_nw_d: torch.Tensor) -> torch.Tensor:
        resampler = self.model.slot_resampler
        assert resampler is not None
        dt = resampler.slot_queries.dtype
        x_cur = raw_slot_nw_d.to(dt)
        Nw, D = x_cur.shape

        if resampler.temporal_conv is not None and resampler.conv_width > 1:
            cw = resampler.conv_width
            pad_needed = (cw - 1) - len(self._conv_ring)
            hist_list: List[torch.Tensor] = []
            if pad_needed > 0:
                zero_f = torch.zeros((Nw, D), device=self.device, dtype=dt)
                hist_list.extend([zero_f] * pad_needed)
            hist_list.extend(self._conv_ring[-(cw - 1) :])
            hist_list.append(x_cur)

            win = torch.stack(hist_list, dim=0).permute(1, 2, 0).contiguous()
            y = F.silu(resampler.temporal_conv(win)).squeeze(-1)
            self._conv_ring.append(x_cur)
            if len(self._conv_ring) > cw - 1:
                self._conv_ring.pop(0)
        else:
            y = x_cur

        return resampler.out_norm(y).view(1, 1, Nw, D)

    @torch.no_grad()
    def _compute_next_motion_rms(self, f_tok_single: torch.Tensor) -> torch.Tensor:
        f32 = f_tok_single.float().reshape(f_tok_single.shape[-2], f_tok_single.shape[-1])
        if self._prev_written_front_tok is None:
            m_val = torch.sqrt(torch.mean(f32 * f32) + 1e-8).view(1)
        else:
            diff = f32 - self._prev_written_front_tok
            m_val = torch.sqrt(torch.mean(diff * diff) + 1e-8).view(1)
        self._prev_written_front_tok = f32.detach().clone()
        return m_val

    @torch.no_grad()
    def _encode_and_queue_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        is_demo: bool,
    ) -> None:
        n = len(front_rgb_list)
        if n == 0:
            return
        need_last_exec = bool(
            (not is_demo)
            and self.cfg.obs_horizon2
            and self.model.readout_ctx_encoder is not None
        )

        if not self.model.uses_history:
            if need_last_exec:
                f_last = to_uint8_rgb(front_rgb_list[-1])[None, ...]
                w_last = to_uint8_rgb(wrist_rgb_list[-1])[None, ...]
                s_last_raw = np.asarray(state_list[-1], dtype=np.float32).reshape(1, 8)
                s_last_norm = torch.from_numpy(self.normalizer.normalize_state(s_last_raw)).to(
                    device=self.device, dtype=torch.float32
                )
                f_lt, w_lt = self._encode_views(f_last, w_last)
                self._prev_exec_obs = (f_lt, w_lt, s_last_norm)
            self._frame_cursor += n
            if is_demo:
                self._demo_frames_count += n
            return

        if is_demo:
            total_demo = self._demo_frames_count + n
            stride = (
                int(self.cfg.long_demo_stride)
                if total_demo > int(self.cfg.adaptive_demo_stride_threshold)
                else int(self.cfg.write_stride)
            )
        else:
            stride = int(self.cfg.write_stride)

        sel_local: List[int] = []
        sel_global: List[int] = []
        for i in range(n):
            g_idx = self._frame_cursor + i
            if not is_demo:
                is_strided = ((g_idx - self._demo_frames_count) % stride) == 0
            else:
                is_strided = (g_idx % stride) == 0
            if is_strided:
                sel_local.append(i)
                sel_global.append(g_idx)

        self._frame_cursor += n
        if is_demo:
            self._demo_frames_count += n

        encode_local = list(sel_local)
        if need_last_exec and ((not encode_local) or encode_local[-1] != (n - 1)):
            encode_local.append(n - 1)
        if not encode_local:
            return

        f_batch = np.stack([to_uint8_rgb(front_rgb_list[i]) for i in encode_local], axis=0)
        w_batch = np.stack([to_uint8_rgb(wrist_rgb_list[i]) for i in encode_local], axis=0)
        s_raw = np.stack([np.asarray(state_list[i], dtype=np.float32) for i in encode_local], axis=0)
        s_norm = torch.from_numpy(self.normalizer.normalize_state(s_raw)).to(
            device=self.device, dtype=torch.float32
        )

        f_tok_all, w_tok_all = self._encode_views(f_batch, w_batch)
        if need_last_exec:
            self._prev_exec_obs = (f_tok_all[-1:], w_tok_all[-1:], s_norm[-1:])

        m_sel = len(sel_local)
        if m_sel == 0:
            return

        f_tok = f_tok_all[:m_sel]
        w_tok = w_tok_all[:m_sel]
        s_str = s_norm[:m_sel]
        step_t = torch.tensor(sel_global, device=self.device, dtype=torch.long)
        demo_t = torch.full((m_sel,), 1 if is_demo else 0, device=self.device, dtype=torch.long)

        raw_slots = self.model.encode_history_frames_raw(f_tok, w_tok, s_str, step_t, demo_t)
        for k_i, g_idx in enumerate(sel_global):
            m_val = self._compute_next_motion_rms(f_tok[k_i])
            self._pending_strided.append((g_idx, raw_slots[k_i], is_demo, m_val))

    @torch.no_grad()
    def _flush_eligible_history(self, cutoff_excl: int) -> None:
        if not self.model.uses_history:
            self._pending_strided.clear()
            return

        to_commit: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        keep_pending: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        for item in self._pending_strided:
            if item[0] < cutoff_excl:
                to_commit.append(item)
            else:
                keep_pending.append(item)
        self._pending_strided = keep_pending

        if not to_commit:
            return

        if self.model.arch == "smolvla_m2_framesamp":
            for _, raw_s, is_d, _ in to_commit:
                conv_s = self._stream_slot_conv_single(raw_s)
                self._framesamp_committed.append(conv_s.view(-1))
                if is_d:
                    self._framesamp_demo_count += 1
            return

        assert (
            self.model.gdn1 is not None
            and self.model.gdn2 is not None
            and self.model.mid_norm is not None
            and self.model.mid_ffn is not None
        )
        z_frames = torch.cat([self._stream_slot_conv_single(rs) for _, rs, _, _ in to_commit], dim=1)
        m_frames = torch.stack([mv for _, _, _, mv in to_commit], dim=0).unsqueeze(0)
        W_seg = z_frames.shape[1]
        Nw = z_frames.shape[2]

        k1, v1, g1, b1, q1 = self.model.gdn1.project_frame_slots(z_frames, motion_feat=m_frames)
        assert q1 is not None
        o1, self._S1, self._occ1 = run_dual_backend_gdn(
            k=k1,
            v=v1,
            g_fp32=g1,
            beta_fp32=b1,
            q=q1,
            initial_state_fp32=self._S1,
            initial_occ_fp32=self._occ1,
            track_occ=True,
            force_pytorch=self.force_pytorch_gdn,
        )
        assert o1 is not None
        z_flat = z_frames.view(1, W_seg * Nw, -1)
        seq1 = self.model.gdn1.format_seq_readout(z_flat, o1)
        z_mid = z_flat.to(seq1.dtype) + seq1
        z_mid = z_mid + self.model.mid_ffn(self.model.mid_norm(z_mid))
        k2, v2, g2, b2, _ = self.model.gdn2.project_frame_slots(
            z_mid.view(1, W_seg, Nw, -1), motion_feat=m_frames
        )
        _, self._S2, self._occ2 = run_dual_backend_gdn(
            k=k2,
            v=v2,
            g_fp32=g2,
            beta_fp32=b2,
            initial_state_fp32=self._S2,
            initial_occ_fp32=self._occ2,
            track_occ=True,
            force_pytorch=self.force_pytorch_gdn,
        )

    def _build_current_mem_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.model.uses_history:
            return None
        if self.model.arch == "smolvla_m2_framesamp":
            B_sz = self.cfg.framesamp_buffer_size
            D = self.cfg.d_mem
            buf_tok = torch.zeros((1, B_sz, D), device=self.device, dtype=self.dtype)
            buf_mask = torch.zeros((1, B_sz), device=self.device, dtype=torch.bool)
            n_avail = len(self._framesamp_committed)
            if n_avail > 0:
                sel_idx, n_sel = self.model.select_framesamp_slot_indices(
                    n_avail, self._framesamp_demo_count, budget=B_sz, device=self.device
                )
                stacked = torch.stack(self._framesamp_committed, dim=0).to(self.dtype)
                buf_tok[0, :n_sel] = stacked[sel_idx]
                buf_mask[0, :n_sel] = True
            return {"buf_tokens": buf_tok, "buf_mask": buf_mask}

        assert self._S1 is not None and self._S2 is not None
        return {
            "S1": self._S1,
            "occ1": self._occ1,  # type: ignore[dict-item]
            "S2": self._S2,
            "occ2": self._occ2,  # type: ignore[dict-item]
        }

    # -------------------------------------------------------------------------
    # `StatefulPolicy` Public API
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        self._encode_and_queue_frames(front_rgb_list, wrist_rgb_list, state_list, is_demo=True)
        self._flush_eligible_history(cutoff_excl=self._frame_cursor)

    @torch.no_grad()
    def ingest_step_history(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        self._encode_and_queue_frames(front_rgb_list, wrist_rgb_list, state_list, is_demo=False)

    @torch.no_grad()
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
        # 1. Enforce exact A3 cutoff rule BEFORE reading memory
        eff_step = int(step_idx) if step_idx > 0 or self._frame_cursor == 0 else self._frame_cursor
        eff_exec_start = int(exec_start_idx) if exec_start_idx > 0 else self._demo_frames_count
        cutoff_excl = self.model.compute_a3_cutoff_excl(
            eff_step, eff_exec_start, exclusion_window=self.cfg.exclusion_window
        )
        self._flush_eligible_history(cutoff_excl=cutoff_excl)

        # 2. Encode current observation & prefill 16-layer `SmolVLM2` prefix KV cache ONCE
        f_img = to_uint8_rgb(front_rgb)
        w_img = to_uint8_rgb(wrist_rgb)
        raw_st = np.asarray(state, dtype=np.float32).reshape(8)
        st_norm = torch.from_numpy(self.normalizer.normalize_state(raw_st)).to(
            device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        f_tok, w_tok = self._encode_views(f_img[None, ...], w_img[None, ...])
        lang_ids, lang_mask = self._tokenize_prompt_cached(task_goal)

        prefill = self.model.prefill_vlm_kv_from_tokens(
            f_tok, w_tok, lang_ids, lang_mask, st_norm
        )

        ctx_tokens: Optional[torch.Tensor] = None
        ctx_mask: Optional[torch.Tensor] = None
        if self.model.readout_ctx_encoder is not None:
            if self.cfg.obs_horizon2 and self._prev_exec_obs is not None:
                prev_f, prev_w, prev_s = self._prev_exec_obs
            else:
                prev_f, prev_w, prev_s = f_tok, w_tok, st_norm
            ctx_tokens, ctx_mask = self.model.encode_readout_context(
                f_tok,
                w_tok,
                st_norm,
                prefill["lang_raw"],
                lang_mask,
                prev_front_vlm=prev_f,
                prev_wrist_vlm=prev_w,
                prev_state_norm=prev_s,
            )
            if self.cfg.obs_horizon2:
                self._prev_exec_obs = (f_tok, w_tok, st_norm)

        # 3. Read memory & run multi-hop Coda loop (with optional online CRN halting)
        mem_state = self._build_current_mem_state_dict()
        ro_out = self.model.run_coda_readout(
            ctx_tokens,
            ctx_mask,
            mem_state=mem_state,
            batch_size=1,
            device=self.device,
            num_loops=self.num_loops,
            halt_mode=self.halt_mode,
            halt_threshold=self.halt_threshold,
            coda_conv_tau=self.coda_conv_tau,
        )
        coda_tok = ro_out["coda"]
        mem_mask = ro_out["mem_mask"]
        loop_depth_used = float(ro_out["loop_depths"][0].item())

        # 4. Run 10-step `SmolVLA` Euler integration (`t = 1.0 -> 0.0`) reusing `past_key_values`!
        norm_actions = self.model.sample_action_chunk_from_kv(
            prefill["prefix_pad_masks"],
            prefill["past_key_values"],
            mem_tokens=coda_tok,
            mem_mask=mem_mask,
            ode_steps=self.ode_steps,
            generator=self.gen,
        )[0].cpu().numpy()
        abs_actions = self.normalizer.decode_actions(norm_actions, raw_st)

        # 5. Queue current frame for future chunks once `< cutoff_excl`
        cur_global_idx = self._frame_cursor
        if self.model.uses_history:
            should_write_cur = ((cur_global_idx - self._demo_frames_count) % self.cfg.write_stride) == 0
            if should_write_cur:
                step_t = torch.tensor([cur_global_idx], device=self.device, dtype=torch.long)
                demo_t = torch.zeros((1,), device=self.device, dtype=torch.long)
                raw_cur_slot = self.model.encode_history_frames_raw(
                    f_tok, w_tok, st_norm, step_t, demo_t
                )[0]
                m_val_cur = self._compute_next_motion_rms(f_tok[0])
                self._pending_strided.append((cur_global_idx, raw_cur_slot, False, m_val_cur))

        self._frame_cursor += 1
        return PolicyChunkOutput(
            actions=abs_actions,
            loop_depth=loop_depth_used,
            aux_info={
                "arch": self.model.arch,
                "cutoff_excl": cutoff_excl,
                "state_bytes": self.model.state_size_bytes(),
            },
        )


# Convenient alias matching `RoboMMEPolicyWrapper` naming convention
SmolVLAPolicyWrapper = SmolVLARoboMMEWrapper


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained SmolVLA-DeltaLoop checkpoint on RoboMME")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained `.pt` checkpoint")
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--dataset", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[7])
    p.add_argument("--exec-horizon", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1300)
    p.add_argument("--num-loops", type=int, default=None)
    p.add_argument("--halt-mode", default="learned_gate", choices=["fixed", "learned_gate", "coda_conv"])
    p.add_argument("--halt-threshold", type=float, default=0.5)
    p.add_argument("--coda-conv-tau", type=float, default=0.08)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--save-videos-dir", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        from robomme_eval.runner import evaluate_policy
    except ImportError:
        from src.robomme_eval.runner import evaluate_policy

    args = _build_cli().parse_args(argv)
    tasks = ROBOMME_TASKS if args.tasks == ["all"] else tuple(args.tasks)

    policy = SmolVLARoboMMEWrapper(
        args.checkpoint,
        device=args.device,
        num_loops=args.num_loops,
        halt_mode=args.halt_mode,
        halt_threshold=args.halt_threshold,
        coda_conv_tau=args.coda_conv_tau,
        ode_steps=args.ode_steps,
    )
    res = evaluate_policy(
        policy,
        tasks=tasks,
        dataset=args.dataset,
        max_episodes_per_task=args.episodes,
        model_seeds=args.seeds,
        exec_horizon=args.exec_horizon,
        max_steps=args.max_steps,
        output_json=args.output_json,
        save_videos_dir=args.save_videos_dir,
        resume=(not args.no_resume),
    )
    ov = res["summary"]["overall"]
    print(
        f"\n[{policy.model.arch}] episodes={ov['num_episodes']} "
        f"SR={ov['success_rate']*100:.2f}% "
        f"(CI95 [{ov['success_ci95_low']*100:.2f}%, {ov['success_ci95_high']*100:.2f}%]) "
        f"L_mean={ov['mean_loop_depth']:.2f} ms/chunk={ov['mean_ms_per_chunk']:.2f}"
    )
    if args.output_json:
        print(f"Saved evaluation JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
