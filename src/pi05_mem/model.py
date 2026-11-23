"""End-to-End `DeltaMem-π0.5` Policy (`src/pi05_mem/model.py`).

Unifies `@pi05-bridge`'s `src.pi05.Pi05Model` (`Yinpei/pi05_baseline` 3.44B warm-start + 18-layer
`HistoryBlock` `MemoryAttention` / `MemoryRMSNorm`) with `DeltaMemCore`:
  - `p2_b0_pi05_base`:     Zero-memory `Yinpei/pi05_baseline` pass-through (`0.0` diff verification).
  - `p2_b1_t2mem`:         `T²Mem`-style 1-Layer Online Fast-Weight TTT (`arXiv:2609.36720`).
  - `p2_b2_framesamp`:     Non-privileged strided Perceptual Frame-Sampling buffer (`FrameSamp+Modul`).
  - `p2_m1_gdn2l_ms`:      Multi-Timescale (`Short/Mid/Long`) 2-Layer `fp32` Gated DeltaNet (`IDEA.md`
                           + Consolidation Gate + `M5` read-before-write + Boundary Forget Gate).
  - `p2_m2_deltamem`:      **Flagship `DeltaMem-π0.5`**: `p2_m1_gdn2l_ms` + Non-Privileged Multi-Scale
                           (`2x2+4x4+8x8`) `GDN-Surprise` & Gripper-Event Visual Landmark Bank +
                           `Titans` Persistent Tokens + Dense Grounded Aux Supervision + Zero-Oracle
                           Self-Predicted `<x, y>` Pointer Tokens + `Mem-CFG`.
  - `p2_m3_deltamem_loop`: Stage-C Zero-Init Residual Multi-Hop Query Refinement (`gamma_0 = 0`) +
                           `M_CRN = 8` Relative-Quantile Uncertainty Gate (`SR >= L=1` at init).
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.subgoal_grounding import (
    apply_subgrid_affine_to_tokens,
    sample_subgrid_crop_affine,
    transform_coords_under_affine,
)
from pi05 import MemModMode, Pi05Config, Pi05Model, load_pi05_checkpoint
from pi05.config import PerceptualMemConfig
from pi05.perceptual_mem import PerceptualMemoryEncoder

from .config import Pi05MemConfig
from .memory_modules import (
    ContinuousVLMGroundingBridge,
    CurrentObservationContextEncoder,
    DualViewEventPerceiverWriter,
    GraftedMemoryRMSNorm,
    GroundedAuxiliarySuite,
    HybridSurpriseGDNBridge,
    MultiTimescale2LayerGDN,
    PerceptualFrameSampMemory,
    ResidualMultiHopRefiner,
    StageCCRNGate,
    SurpriseEventLandmarkBank,
    T2MemFastWeightMemory,
    VLMPrefixSurpriseSelector,
    compute_a3_cutoff_frame,
    pool_spatial_tokens_to_square,
    select_episode_write_indices,
)


def build_pi05_config_from_mem_cfg(cfg: Pi05MemConfig) -> Pi05Config:
    """Create a matching `src.pi05.Pi05Config` (supports both full 3.44B and tiny CPU test dims)."""
    mod_mode: MemModMode
    if cfg.arch == "p2_b0_pi05_base":
        mod_mode = "none"
    elif cfg.modulator_mode == "mme_vla_exact":
        mod_mode = "mme_vla_exact"
    else:
        mod_mode = "modul_residual"

    has_perc = cfg.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid")
    perc_cfg = (
        PerceptualMemConfig(
            budget=int(cfg.hybrid_canvas_tokens),
            num_views=1,
            token_per_image=16,
            use_pos_emb=True,
            use_state_emb=False,
            img_input_dim=int(cfg.vis_dim),
            memory_token_dim=int(cfg.mem_token_dim),
        )
        if has_perc
        else None
    )

    return Pi05Config(
        paligemma_width=cfg.vlm_width,
        paligemma_mlp_dim=cfg.vlm_mlp_dim,
        expert_width=cfg.expert_width,
        expert_mlp_dim=cfg.expert_mlp_dim,
        depth=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        siglip_width=96 if cfg.vis_dim < 1152 else 1152,
        siglip_depth=2 if cfg.num_layers <= 4 else 27,
        siglip_mlp_dim=128 if cfg.num_layers <= 4 else 4304,
        siglip_num_heads=4 if cfg.num_layers <= 4 else 16,
        action_dim=cfg.padded_action_dim,
        real_action_dim=cfg.raw_action_dim,
        action_horizon=cfg.action_horizon,
        exec_horizon=cfg.exec_horizon,
        max_token_len=cfg.max_token_len,
        discrete_state_input=cfg.discrete_state_input,
        num_inference_steps=cfg.ode_steps,
        mem_mod_mode=mod_mode,
        mem_num_heads=cfg.mem_attn_heads,
        mem_num_kv_heads=cfg.mem_attn_kv_heads,
        mem_head_dim=cfg.mem_attn_head_dim,
        has_perceptual_mem_encoder=has_perc,
        perceptual_mem=perc_cfg,
        precision="float32",
    )


class DeltaMemCore(nn.Module):
    """Modular Phase-2 Memory & Grounded Supervision Engine producing `(mem_tokens, mem_mask)`."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.arch = cfg.arch

        if self.arch != "p2_b0_pi05_base":
            self.writer = DualViewEventPerceiverWriter(cfg)
            self.obs_encoder = CurrentObservationContextEncoder(cfg)
        else:
            self.writer = None  # type: ignore[assignment]
            self.obs_encoder = None  # type: ignore[assignment]

        self.gdn: Optional[MultiTimescale2LayerGDN] = None
        self.landmark_bank: Optional[SurpriseEventLandmarkBank] = None
        self.hybrid_bridge: Optional[HybridSurpriseGDNBridge] = None
        self.vlm_surprise_selector: Optional[VLMPrefixSurpriseSelector] = None
        self.t2mem: Optional[T2MemFastWeightMemory] = None
        self.framesamp: Optional[PerceptualFrameSampMemory] = None

        if self.arch in (
            "p2_m1_gdn2l_ms",
            "p2_m2_deltamem",
            "p2_m3_deltamem_loop",
            "p2_m4_deltamem_aligned",
            "p2_m5_hybrid_surprise_gdn",
            "p2_m7_grounded_hybrid",
            "p2_m8_vlmprefix_surprise_hybrid",
        ):
            self.gdn = MultiTimescale2LayerGDN(cfg)
            if cfg.use_landmark_bank:
                self.landmark_bank = SurpriseEventLandmarkBank(cfg)
            if self.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid"):
                self.hybrid_bridge = HybridSurpriseGDNBridge(cfg)
            if self.arch == "p2_m8_vlmprefix_surprise_hybrid":
                self.vlm_surprise_selector = VLMPrefixSurpriseSelector(cfg)
        elif self.arch == "p2_b1_t2mem":
            self.t2mem = T2MemFastWeightMemory(cfg)
        elif self.arch == "p2_b2_framesamp":
            self.framesamp = PerceptualFrameSampMemory(cfg)

        self.aux_suite: Optional[GroundedAuxiliarySuite] = (
            GroundedAuxiliarySuite(cfg)
            if (cfg.use_grounded_aux and self.arch != "p2_b0_pi05_base")
            else None
        )

        self.hop2_refiner: Optional[ResidualMultiHopRefiner] = None
        self.crn_gate: Optional[StageCCRNGate] = None
        if cfg.use_residual_hop2 or self.arch == "p2_m3_deltamem_loop":
            self.hop2_refiner = ResidualMultiHopRefiner(cfg)
            self.crn_gate = StageCCRNGate(cfg)

        self.continuous_vlm_bridge: Optional[ContinuousVLMGroundingBridge] = (
            ContinuousVLMGroundingBridge(cfg) if cfg.use_continuous_vlm_ground else None
        )

    @staticmethod
    def smooth_surprise_ema(
        raw_surprise: torch.Tensor,             # (W,) fp32
        grip_spike: torch.Tensor,               # (W,) fp32
        is_demo: torch.Tensor,                  # (W,) bool
        boundary_reset: Optional[torch.Tensor], # Optional (W,) fp32
        ema_beta: float = 0.80,
        init_ema: float = 0.0,
    ) -> Tuple[torch.Tensor, float]:
        """`Titans` (`arXiv:2501.00663`) EMA surprise momentum across multi-frame physical swaps/reveals."""
        w = int(raw_surprise.numel())
        if w == 0:
            return raw_surprise, init_ema
        inst = raw_surprise.float() + 0.5 * (grip_spike.float() > 0.25).float() + 0.35 * grip_spike.float()
        out = torch.empty_like(inst)
        cur = float(init_ema)
        beta = float(ema_beta)
        for i in range(w):
            if boundary_reset is not None and float(boundary_reset[i].item()) > 0.5:
                cur = 0.0
            if i > 0 and bool(is_demo[i - 1].item()) and not bool(is_demo[i].item()):
                # Reset surprise accumulator cleanly at demo->exec transition without flushing GDN state
                cur = 0.0
            cur = beta * cur + (1.0 - beta) * float(inst[i].item())
            out[i] = max(float(inst[i].item()), cur)
        # Boost first & last conditioning demo frames (`Prescription 2` Eq. 195)
        demo_idx = torch.nonzero(is_demo.bool(), as_tuple=False).flatten()
        if demo_idx.numel() >= 1:
            out[demo_idx[0]] = out[demo_idx[0]] + 1.0
            out[demo_idx[-1]] = out[demo_idx[-1]] + 1.0
        return out, cur

    def forward_episode_chunks(
        self,
        *,
        ep_front_emb: torch.Tensor,                  # (T_tot, 64, D_vis)
        ep_wrist_emb: torch.Tensor,                  # (T_tot, 16, D_vis)
        ep_state_norm: torch.Tensor,                 # (T_tot, 8)
        ep_is_demo: torch.Tensor,                    # (T_tot,) bool
        exec_start_idx: int,
        chunk_step_indices: torch.Tensor,            # (M,) int64 sorted ascending
        readout_cutoff_excl: torch.Tensor,           # (M,) int64 sorted ascending
        curr_front_emb: torch.Tensor,                # (M, P_f_curr, D_vis)
        curr_wrist_emb: torch.Tensor,                # (M, P_w_curr, D_vis)
        curr_state_norm: torch.Tensor,               # (M, 8)
        prompt_pool: Optional[torch.Tensor] = None,  # Optional (M, vlm_width) or (1, vlm_width)
        ep_prompt_cos_jump: Optional[torch.Tensor] = None,  # Optional (T_tot,) fp32 prompt jump at boundaries
        perceptual_mem_encoder: Optional[nn.Module] = None,
        force_pytorch_gdn: bool = False,
        use_interleaved_readout: bool = True,
        write_stride_exec_override: Optional[int] = None,
        precision_insertpeg_climax: bool = False,
    ) -> Dict[str, Any]:
        """Compute causal `(mem_tokens, mem_mask)` for `M` execution chunks in `O(W)` single-pass time!

        Advances recurrent state incrementally across disjoint intervals `[u_{m-1}, u_m)` so every
        written frame in a `1,411`-step episode is processed **at most once** (`0` redundant prefix work)!
        """
        m_chunks = int(chunk_step_indices.numel())
        dev = curr_front_emb.device

        if self.arch == "p2_b0_pi05_base":
            return {
                "mem_tokens_l1": None,
                "mem_tokens_l2": None,
                "mem_mask": None,
                "aux_preds": None,
                "crn_logits": None,
            }

        assert self.writer is not None and self.obs_encoder is not None
        slow_mem_grad = any(p.requires_grad for p in self.writer.parameters())
        slow_ctx = contextlib.nullcontext() if slow_mem_grad else torch.no_grad()
        if prompt_pool is not None and prompt_pool.shape[0] == 1 and m_chunks > 1:
            prompt_pool_m = prompt_pool.expand(m_chunks, -1)
        else:
            prompt_pool_m = prompt_pool

        with slow_ctx:
            ctx_tokens, front_spatial_ctx = self.obs_encoder(
                curr_front_emb, curr_wrist_emb, curr_state_norm, prompt_pool=prompt_pool_m
            )  # (M, L_ctx, d_mem), (M, 64, d_mem)

            t_tot = int(ep_front_emb.shape[0])
            write_idx = select_episode_write_indices(
                t_tot,
                exec_start_idx,
                self.cfg,
                device=dev,
                write_stride_exec_override=write_stride_exec_override,
            )
            w_len = int(write_idx.numel())

            f_w = ep_front_emb[write_idx]
            w_w = ep_wrist_emb[write_idx]
            s_w = ep_state_norm[write_idx]
            d_w = ep_is_demo[write_idx]
            kin = self.writer.compute_frame_kinematics(f_w, s_w, d_w)

            raw_gdn_w, lm_slots_w = self.writer.encode_frames_raw(f_w, w_w, s_w, write_idx, d_w, kin)
            gdn_slots_w, _ = self.writer.apply_slot_conv(raw_gdn_w, conv_history=None)

        p_jump_w: Optional[torch.Tensor] = None
        if ep_prompt_cos_jump is not None and w_len > 0:
            # Aggregate any prompt jump occurring in `(write_idx[i-1], write_idx[i]]` so strided writes never miss it!
            pj = torch.zeros((w_len,), device=dev, dtype=torch.float32)
            w_list = write_idx.tolist()
            for wi in range(w_len):
                lo = 0 if wi == 0 else (w_list[wi - 1] + 1)
                hi = w_list[wi] + 1
                if hi > lo:
                    pj[wi] = ep_prompt_cos_jump[lo:hi].float().max()
            p_jump_w = pj

        # ---------------------------------------------------------------------
        # Branch 1: Multi-Timescale 2-Layer `fp32` GDN + Optional Landmark Bank
        # ---------------------------------------------------------------------
        if self.gdn is not None:
            b_reset_w: Optional[torch.Tensor] = None
            if self.gdn.boundary_gate is not None and w_len > 0:
                b_reset_w = self.gdn.boundary_gate(gdn_slots_w[:, 0, :], kin, prompt_cos_jump=p_jump_w)

            # Single parallel `no_grad` pass over all `W` frames to get exact causal GDN write-surprise!
            if (
                self.landmark_bank is not None
                or self.hybrid_bridge is not None
                or self.vlm_surprise_selector is not None
            ) and w_len > 0:
                k1_all, v1_all, g1_all, b1_all, _ = self.gdn.layer1.project_slots(
                    gdn_slots_w.unsqueeze(0),
                    vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
                    grip_spike=kin["grip_spike"].unsqueeze(0),
                    boundary_reset=b_reset_w.unsqueeze(0) if b_reset_w is not None else None,
                )
                raw_surp = self.gdn.layer1.compute_exact_write_surprise(
                    k1_all.detach(),
                    v1_all.detach(),
                    g1_all.detach(),
                    b1_all.detach(),
                    num_write_slots=self.cfg.num_write_slots,
                    force_pytorch=force_pytorch_gdn,
                ).squeeze(0)
                salience_w, _ = self.smooth_surprise_ema(
                    raw_surp,
                    kin["grip_spike"],
                    d_w,
                    b_reset_w,
                    ema_beta=self.cfg.surprise_ema_beta,
                )
                pure_vis_sal_w, _ = self.smooth_surprise_ema(
                    raw_surp,
                    torch.zeros_like(kin["grip_spike"]),
                    torch.zeros_like(d_w),
                    b_reset_w,
                    ema_beta=self.cfg.surprise_ema_beta,
                )
            else:
                salience_w = torch.zeros((w_len,), device=dev, dtype=torch.float32)
                pure_vis_sal_w = torch.zeros((w_len,), device=dev, dtype=torch.float32)

            if use_interleaved_readout:
                u_ends_list: List[int] = []
                decay_list: List[torch.Tensor] = []
                has_hist_list_i: List[bool] = []
                lm_tok_list_i: List[torch.Tensor] = []
                lm_mask_list_i: List[torch.Tensor] = []

                for ci in range(m_chunks):
                    cut_f = int(readout_cutoff_excl[ci].item())
                    q_step = int(chunk_step_indices[ci].item())
                    u_end = int(torch.searchsorted(write_idx, cut_f, right=False).item()) if w_len > 0 else 0
                    u_query = int(torch.searchsorted(write_idx, q_step, right=True).item()) if w_len > 0 else 0
                    has_hist_q = u_end > 0
                    dec_q = torch.tensor(1.0, device=dev, dtype=torch.float32)
                    if b_reset_w is not None and u_query > u_end:
                        pending_r = b_reset_w[u_end:u_query].max()
                        if float(pending_r.item()) > float(self.cfg.boundary_flush_threshold):
                            dec_q = torch.exp(-float(self.cfg.boundary_reset_strength) * pending_r)
                            has_hist_q = False

                    u_ends_list.append(u_end)
                    decay_list.append(dec_q)
                    has_hist_list_i.append(has_hist_q)

                    if self.landmark_bank is not None:
                        if not has_hist_q:
                            sel_pos = torch.empty((0,), dtype=torch.long, device=dev)
                        else:
                            sel_pos = self.landmark_bank.select_landmark_indices(
                                write_idx, d_w, salience_w, b_reset_w, cutoff_excl_frame=cut_f
                            )
                        lt, lm = self.landmark_bank.gather_padded_landmarks(
                            lm_slots_w, write_idx, d_w, salience_w, sel_pos, query_step_idx=q_step
                        )
                        lm_tok_list_i.append(lt)
                        lm_mask_list_i.append(lm)

                u_ends_t = torch.tensor(u_ends_list, device=dev, dtype=torch.long)
                decay_t = torch.stack(decay_list, dim=0)
                has_hist_b = torch.tensor(has_hist_list_i, device=dev, dtype=torch.bool)
                with slow_ctx:
                    gdn_tok_l1, cov_stats = self.gdn.forward_interleaved_episode_readout(
                        gdn_slots_w,
                        kin,
                        b_reset_w,
                        u_ends=u_ends_t,
                        decay_factors=decay_t,
                        ctx_tokens=ctx_tokens,
                        force_pytorch=force_pytorch_gdn,
                    )
                S1_b = occ1_b = S2_b = occ2_b = None
                lm_tok_b_opt = torch.stack(lm_tok_list_i, dim=0) if self.landmark_bank is not None else None
                lm_mask_b_opt = torch.stack(lm_mask_list_i, dim=0) if self.landmark_bank is not None else None
            else:
                # Reference incremental segment loop (kept for exact mathematical parity verification!)
                order = torch.argsort(readout_cutoff_excl)
                inv_order = torch.argsort(order)
                states_cur = self.gdn.init_empty_states(1, dev)
                prev_u = 0

                s1_list: List[torch.Tensor] = []
                o1_list: List[torch.Tensor] = []
                s2_list: List[torch.Tensor] = []
                o2_list: List[torch.Tensor] = []
                has_hist_list: List[bool] = []
                lm_tok_list: List[torch.Tensor] = []
                lm_mask_list: List[torch.Tensor] = []

                for ord_i in order.tolist():
                    cut_f = int(readout_cutoff_excl[ord_i].item())
                    q_step = int(chunk_step_indices[ord_i].item())
                    u_end = int(torch.searchsorted(write_idx, cut_f, right=False).item()) if w_len > 0 else 0
                    u_query = int(torch.searchsorted(write_idx, q_step, right=True).item()) if w_len > 0 else 0

                    if u_end > prev_u:
                        seg_kin = {
                            "vis_motion_rms": kin["vis_motion_rms"][prev_u:u_end],
                            "grip_spike": kin["grip_spike"][prev_u:u_end],
                        }
                        seg_br = b_reset_w[prev_u:u_end].unsqueeze(0) if b_reset_w is not None else None
                        seg_out = self.gdn.forward_segment(
                            gdn_slots_w[prev_u:u_end].unsqueeze(0),
                            seg_kin,
                            seg_br,
                            init_states=states_cur,
                            compute_surprise=False,
                            force_pytorch=force_pytorch_gdn,
                        )
                        states_cur = (seg_out["S1"], seg_out["occ1"], seg_out["S2"], seg_out["occ2"])
                        prev_u = u_end

                    s1_q, o1_q, s2_q, o2_q = states_cur
                    has_hist_q = u_end > 0
                    eff_br_for_lm = b_reset_w
                    if b_reset_w is not None and u_query > u_end:
                        pending_r = b_reset_w[u_end:u_query].max()
                        if float(pending_r.item()) > float(self.cfg.boundary_flush_threshold):
                            decay_f = torch.exp(-float(self.cfg.boundary_reset_strength) * pending_r)
                            s1_q = s1_q * decay_f
                            o1_q = o1_q * decay_f
                            s2_q = s2_q * decay_f
                            o2_q = o2_q * decay_f
                            has_hist_q = False

                    s1_list.append(s1_q)
                    o1_list.append(o1_q)
                    s2_list.append(s2_q)
                    o2_list.append(o2_q)
                    has_hist_list.append(has_hist_q)

                    if self.landmark_bank is not None:
                        if not has_hist_q:
                            sel_pos = torch.empty((0,), dtype=torch.long, device=dev)
                        else:
                            sel_pos = self.landmark_bank.select_landmark_indices(
                                write_idx, d_w, salience_w, eff_br_for_lm, cutoff_excl_frame=cut_f
                            )
                        lt, lm = self.landmark_bank.gather_padded_landmarks(
                            lm_slots_w, write_idx, d_w, salience_w, sel_pos, query_step_idx=q_step
                        )
                        lm_tok_list.append(lt)
                        lm_mask_list.append(lm)

                S1_b = torch.cat(s1_list, dim=0)[inv_order]
                occ1_b = torch.cat(o1_list, dim=0)[inv_order]
                S2_b = torch.cat(s2_list, dim=0)[inv_order]
                occ2_b = torch.cat(o2_list, dim=0)[inv_order]
                has_hist_b = torch.tensor(has_hist_list, device=dev, dtype=torch.bool)[inv_order]
                gdn_tok_l1, cov_stats = self.gdn.readout(ctx_tokens, S1_b, occ1_b, S2_b, occ2_b)
                lm_tok_b_opt = torch.stack(lm_tok_list, dim=0)[inv_order] if self.landmark_bank is not None else None
                lm_mask_b_opt = torch.stack(lm_mask_list, dim=0)[inv_order] if self.landmark_bank is not None else None

            n_gdn_total = int(gdn_tok_l1.shape[1])
            gdn_mask = has_hist_b.unsqueeze(-1).expand(m_chunks, n_gdn_total)
            if self.cfg.num_persistent_tokens > 0:
                gdn_mask = gdn_mask.clone()
                gdn_mask[:, : self.cfg.num_persistent_tokens] = True

            tok_parts = [gdn_tok_l1]
            mask_parts = [gdn_mask]
            if lm_tok_b_opt is not None and lm_mask_b_opt is not None:
                tok_parts.append(lm_tok_b_opt.to(gdn_tok_l1.dtype))
                mask_parts.append(lm_mask_b_opt)

            base_mem_l1 = torch.cat(tok_parts, dim=1)
            base_mask = torch.cat(mask_parts, dim=1)

            aux_preds = (
                self.aux_suite(front_spatial_ctx, ctx_tokens, base_mem_l1, base_mask)
                if self.aux_suite is not None
                else None
            )
            if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
                sg_tok = aux_preds["self_subgoal_tokens"].to(base_mem_l1.dtype)
                sg_mask = torch.ones((m_chunks, sg_tok.shape[1]), device=dev, dtype=torch.bool)
                mem_l1 = torch.cat([base_mem_l1, sg_tok], dim=1)
                mem_mask = torch.cat([base_mask, sg_mask], dim=1)
            else:
                mem_l1 = base_mem_l1
                mem_mask = base_mask
            mem_l1 = torch.where(mem_mask.unsqueeze(-1), mem_l1, torch.zeros_like(mem_l1))

            if self.hybrid_bridge is not None:
                if perceptual_mem_encoder is None:
                    raise ValueError("`p2_m5_hybrid_surprise_gdn` requires `perceptual_mem_encoder` (`pi05.mem_encoder`)")
                perc_c_list: List[torch.Tensor] = []
                perc_m_list: List[torch.Tensor] = []
                eff_sal_hyb = salience_w
                anc_ratio_ov: Optional[float] = None
                if precision_insertpeg_climax and w_len > 0:
                    # `#405` `InsertPeg`: Lock perceptual surprise slots onto (a) demo peg-end grasp (`grip_spike > 0.20`)
                    # and (b) final-4 demo box-hole insertion climax frames (`100%` train==eval)!
                    eff_sal_hyb = salience_w.clone()
                    sp_c = 0
                    if b_reset_w is not None:
                        br_c = torch.nonzero(b_reset_w > float(self.cfg.boundary_flush_threshold), as_tuple=False).flatten()
                        if br_c.numel() > 0:
                            sp_c = int(br_c[-1].item())
                    d_rel_c = torch.nonzero(d_w[sp_c:], as_tuple=False).flatten()
                    if d_rel_c.numel() > 0:
                        d_abs_c = sp_c + d_rel_c
                        gr_mask = kin["grip_spike"][d_abs_c] > 0.20
                        if bool(gr_mask.any()):
                            eff_sal_hyb[d_abs_c[gr_mask]] += 3.0
                        eff_sal_hyb[d_abs_c[-min(4, int(d_abs_c.numel())) :]] += 3.0
                    anc_ratio_ov = min(float(self.cfg.hybrid_anchor_ratio), 0.75)
                for ci in range(m_chunks):
                    q_step_ci = int(chunk_step_indices[ci].item())
                    # `#259` Bug #2 Fix: Perceptual canvas uses `excl=0` (`q_step_ci + 1`), keeping `excl=16` strictly on GDN!
                    sel_u, loc_pos, sess_s_ci = self.hybrid_bridge.select_hybrid_keyframes(
                        write_idx,
                        d_w,
                        eff_sal_hyb,
                        b_reset_w,
                        cutoff_excl_frame=q_step_ci + 1,
                        query_step_idx=q_step_ci,
                        anchor_ratio_override=anc_ratio_ov,
                    )
                    f_sel = f_w[sel_u] if int(sel_u.numel()) > 0 else f_w[:0]
                    c_tok, c_m = self.hybrid_bridge.encode_perceptual_canvas(
                        f_sel,
                        loc_pos,
                        perceptual_mem_encoder,
                        fallback_curr_front=curr_front_emb[ci],
                        query_local_step=max(0, q_step_ci - sess_s_ci),
                    )
                    perc_c_list.append(c_tok)
                    perc_m_list.append(c_m)
                perc_canvas_b = torch.stack(perc_c_list, dim=0)
                perc_mask_b = torch.stack(perc_m_list, dim=0)
                mem_l1 = self.hybrid_bridge(perc_canvas_b, perc_mask_b, mem_l1, mem_mask, ctx_tokens)
                mem_mask = perc_mask_b

            vlm_prefix_tok_b: Optional[torch.Tensor] = None
            vlm_prefix_mask_b: Optional[torch.Tensor] = None
            if self.vlm_surprise_selector is not None:
                vp_t_list: List[torch.Tensor] = []
                vp_m_list: List[torch.Tensor] = []
                for ci in range(m_chunks):
                    q_step_ci = int(chunk_step_indices[ci].item())
                    v_sel = self.vlm_surprise_selector.select_vlm_surprise_indices(
                        write_idx,
                        d_w,
                        salience_w,
                        pure_vis_sal_w,
                        b_reset_w,
                        cutoff_excl_frame=q_step_ci + 1,
                    )
                    f_vlm = f_w[v_sel] if int(v_sel.numel()) > 0 else f_w[:0]
                    vt_ci, vm_ci = self.vlm_surprise_selector.encode_vlm_surprise_tokens(f_vlm)
                    vp_t_list.append(vt_ci)
                    vp_m_list.append(vm_ci)
                vlm_prefix_tok_b = torch.stack(vp_t_list, dim=0)
                vlm_prefix_mask_b = torch.stack(vp_m_list, dim=0)

            mem_l2: Optional[torch.Tensor] = None
            crn_logits: Optional[torch.Tensor] = None
            if self.hop2_refiner is not None and self.crn_gate is not None:
                np_off = int(self.cfg.num_persistent_tokens)
                gdn_only_l1 = gdn_tok_l1[:, np_off : np_off + self.cfg.num_gdn_readout_tokens, :]
                q_bias2 = self.hop2_refiner.compute_query_bias(gdn_only_l1)
                if use_interleaved_readout:
                    gdn_hop2_only, _ = self.gdn.forward_interleaved_episode_readout(
                        gdn_slots_w,
                        kin,
                        b_reset_w,
                        u_ends=u_ends_t,
                        decay_factors=decay_t,
                        ctx_tokens=ctx_tokens,
                        extra_query_bias=q_bias2,
                        force_pytorch=force_pytorch_gdn,
                    )
                else:
                    assert S1_b is not None and S2_b is not None
                    gdn_hop2_only, _ = self.gdn.readout(
                        ctx_tokens, S1_b, occ1_b, S2_b, occ2_b, extra_query_bias=q_bias2
                    )
                mem_l2 = self.hop2_refiner(mem_l1, mem_mask, gdn_hop2_only, ctx_tokens)
                crn_logits = self.crn_gate(ctx_tokens, mem_l1.detach(), mem_mask, cov_stats)

            return {
                "mem_tokens_l1": mem_l1,
                "mem_tokens_l2": mem_l2,
                "mem_mask": mem_mask,
                "vlm_prefix_tokens": vlm_prefix_tok_b,
                "vlm_prefix_mask": vlm_prefix_mask_b,
                "aux_preds": aux_preds,
                "crn_logits": crn_logits,
                "boundary_reset_max": float(b_reset_w.max().item()) if (b_reset_w is not None and w_len > 0) else 0.0,
            }

        # ---------------------------------------------------------------------
        # Branch 2: `T²Mem` 1-Layer Fast-Weight TTT Baseline (`P2-B1`)
        # ---------------------------------------------------------------------
        if self.t2mem is not None:
            b_reset_w = (
                self.t2mem.boundary_gate(gdn_slots_w[:, 0, :], kin, prompt_cos_jump=p_jump_w)
                if (self.t2mem.boundary_gate is not None and w_len > 0)
                else None
            )
            order = torch.argsort(readout_cutoff_excl)
            inv_order = torch.argsort(order)
            ttt_state = self.t2mem.init_state(1, dev)
            prev_u = 0
            w_list_s: List[torch.Tensor] = []
            b_list_s: List[torch.Tensor] = []
            has_h: List[bool] = []

            for ord_i in order.tolist():
                cut_f = int(readout_cutoff_excl[ord_i].item())
                u_end = int(torch.searchsorted(write_idx, cut_f, right=False).item()) if w_len > 0 else 0
                if u_end > prev_u:
                    seg_br = b_reset_w[prev_u:u_end].unsqueeze(0) if b_reset_w is not None else None
                    ttt_state = self.t2mem.update_segment(
                        gdn_slots_w[prev_u:u_end].unsqueeze(0), seg_br, state=ttt_state
                    )
                    prev_u = u_end
                w_list_s.append(ttt_state[0])
                b_list_s.append(ttt_state[1])
                has_h.append(u_end > 0)

            st_b = (torch.cat(w_list_s, dim=0)[inv_order], torch.cat(b_list_s, dim=0)[inv_order])
            mem_l1 = self.t2mem.query_readout(ctx_tokens, st_b)
            mem_mask = torch.tensor(has_h, device=dev, dtype=torch.bool)[inv_order].unsqueeze(-1).expand_as(
                mem_l1[:, :, 0]
            )
            aux_preds = (
                self.aux_suite(front_spatial_ctx, ctx_tokens, mem_l1, mem_mask)
                if self.aux_suite is not None
                else None
            )
            if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
                sg_tok = aux_preds["self_subgoal_tokens"].to(mem_l1.dtype)
                sg_mask = torch.ones((m_chunks, sg_tok.shape[1]), device=dev, dtype=torch.bool)
                mem_l1 = torch.cat([mem_l1, sg_tok], dim=1)
                mem_mask = torch.cat([mem_mask, sg_mask], dim=1)
            return {
                "mem_tokens_l1": mem_l1,
                "mem_tokens_l2": None,
                "mem_mask": mem_mask,
                "aux_preds": aux_preds,
                "crn_logits": None,
            }

        # ---------------------------------------------------------------------
        # Branch 3: Strided Perceptual Frame-Sampling Baseline (`P2-B2`)
        # ---------------------------------------------------------------------
        assert self.framesamp is not None
        toks_fs: List[torch.Tensor] = []
        masks_fs: List[torch.Tensor] = []
        for ci in range(m_chunks):
            ft, fm = self.framesamp.gather_frames(
                gdn_slots_w,
                write_idx,
                cutoff_excl_frame=int(readout_cutoff_excl[ci].item()),
                query_step_idx=int(chunk_step_indices[ci].item()),
            )
            toks_fs.append(ft)
            masks_fs.append(fm)
        mem_l1 = torch.stack(toks_fs, dim=0)
        mem_mask = torch.stack(masks_fs, dim=0)
        aux_preds = (
            self.aux_suite(front_spatial_ctx, ctx_tokens, mem_l1, mem_mask)
            if self.aux_suite is not None
            else None
        )
        if aux_preds is not None and "self_subgoal_tokens" in aux_preds:
            sg_tok = aux_preds["self_subgoal_tokens"].to(mem_l1.dtype)
            sg_mask = torch.ones((m_chunks, sg_tok.shape[1]), device=dev, dtype=torch.bool)
            mem_l1 = torch.cat([mem_l1, sg_tok], dim=1)
            mem_mask = torch.cat([mem_mask, sg_mask], dim=1)
        return {
            "mem_tokens_l1": mem_l1,
            "mem_tokens_l2": None,
            "mem_mask": mem_mask,
            "aux_preds": aux_preds,
            "crn_logits": None,
        }


class ExpertDropoutLoRALinear(nn.Module):
    """Regularized ActionExpert LoRA (`y = base(x) + (drop(x) @ lora_a @ lora_b) * scaling`, Step-0 = `0.0` diff)."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(1, rank))
        self.lora_drop: nn.Module = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(in_features, self.rank, dtype=torch.float32, device=device))
        self.lora_b = nn.Parameter(torch.zeros(self.rank, out_features, dtype=torch.float32, device=device))
        nn.init.normal_(self.lora_a, mean=0.0, std=1.0 / math.sqrt(in_features))
        if dtype == torch.bfloat16:
            self.lora_a.data = self.lora_a.data.to(torch.bfloat16)
            self.lora_b.data = self.lora_b.data.to(torch.bfloat16)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lx = self.lora_drop(x.to(self.lora_a.dtype))
        delta = torch.matmul(torch.matmul(lx, self.lora_a), self.lora_b) * self.scaling
        return out + delta.to(out.dtype)


class DeltaMemPi05Policy(nn.Module):
    """Complete Phase-2 Policy coupling `src.pi05.Pi05Model` with `DeltaMemCore` across Stages A, B, C."""

    def __init__(
        self,
        cfg: Pi05MemConfig,
        *,
        pi05_model: Optional[Pi05Model] = None,
        pi05_ckpt_dir: Optional[str | Path] = None,
        device: str | torch.device = "cpu",
        precision: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        dev = torch.device(device)

        if pi05_model is not None:
            self.pi05 = pi05_model
        elif pi05_ckpt_dir is not None:
            self.pi05, self.loaded_norm_stats = load_pi05_checkpoint(
                pi05_ckpt_dir,
                device=dev,
                precision="bfloat16" if precision == "bfloat16" else "float32",
            )
        else:
            p_cfg = build_pi05_config_from_mem_cfg(cfg)
            p_cfg.precision = "bfloat16" if precision == "bfloat16" else "float32"
            self.pi05 = Pi05Model(p_cfg).to(dev)
            self.loaded_norm_stats = None

        self.pi05.config.max_token_len = max(int(self.pi05.config.max_token_len), int(cfg.max_token_len))

        if cfg.arch != "p2_b0_pi05_base":
            mod_mode: MemModMode = (
                "mme_vla_exact" if cfg.modulator_mode == "mme_vla_exact" else "modul_residual"
            )
            if self.pi05.config.mem_mod_mode != mod_mode:
                self.pi05.enable_memory_modulation(mod_mode)
                self.pi05.apply_precision(self.pi05.config.precision)
                self.pi05.to(dev)
            if cfg.modulator_mode == "grafted_residual":
                for blk in self.pi05.paligemma_with_expert.layers:
                    graft_norm = GraftedMemoryRMSNorm(
                        dim=self.pi05.config.expert_width,
                        eps=self.pi05.config.rms_norm_eps,
                        init_gate=float(cfg.graft_gate_init),
                    ).to(device=dev, dtype=self.pi05.compute_dtype)
                    graft_norm.graft_gate.data = graft_norm.graft_gate.data.float()
                    if blk.mem_rms_norm_ffn is not None and getattr(blk.mem_rms_norm_ffn, "dense", None) is not None:
                        graft_norm.dense.weight.data.copy_(blk.mem_rms_norm_ffn.dense.weight.data)
                        if blk.mem_rms_norm_ffn.dense.bias is not None and graft_norm.dense.bias is not None:
                            graft_norm.dense.bias.data.copy_(blk.mem_rms_norm_ffn.dense.bias.data)
                    blk.mem_rms_norm_ffn = graft_norm  # type: ignore[assignment]
            if cfg.q_rope_offset is not None:
                for blk in self.pi05.paligemma_with_expert.layers:
                    if blk.mem_attn is not None and hasattr(blk.mem_attn, "q_rope_offset"):
                        blk.mem_attn.q_rope_offset = int(cfg.q_rope_offset)
                    if (
                        cfg.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid")
                        and blk.mem_attn is not None
                        and hasattr(blk.mem_attn, "k_rope_span_frames")
                    ):
                        blk.mem_attn.k_rope_span_frames = True

        if (
            cfg.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid")
            and self.pi05.mem_encoder is None
        ):
            pm_cfg = PerceptualMemConfig(
                budget=int(cfg.hybrid_canvas_tokens),
                num_views=1,
                token_per_image=16,
                use_pos_emb=True,
                use_state_emb=False,
                img_input_dim=int(cfg.vis_dim),
                memory_token_dim=int(cfg.mem_token_dim),
            )
            self.pi05.config.has_perceptual_mem_encoder = True
            self.pi05.config.perceptual_mem = pm_cfg
            self.pi05.mem_encoder = PerceptualMemoryEncoder(pm_cfg).to(
                device=dev, dtype=self.pi05.compute_dtype
            )

        self.mem_core = DeltaMemCore(cfg).to(device=dev)
        if precision == "bfloat16":
            self._cast_mem_core_bf16()
        self.has_vlm_lora = False
        self.has_expert_lora = False
        self._has_custom_dense_expert = False

    def graft_history_from_checkpoint(self, framesamp_ckpt_dir: str | Path) -> int:
        """Stream-load ONLY `mem_encoder` + 18L `mem_attn` + `mem_rms_norm_ffn.dense` from `perceptual_framesamp_modul_pt` (`< 90 MB` RAM)."""
        from safetensors import safe_open

        sf_path = Path(framesamp_ckpt_dir) / "model.safetensors"
        own_sd = self.pi05.state_dict()
        copied = 0
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if (
                    k.startswith("mem_encoder.")
                    or ".mem_attn." in k
                    or ".mem_rms_norm_ffn.dense." in k
                ) and k in own_sd:
                    own_sd[k].copy_(sf.get_tensor(k).to(device=own_sd[k].device, dtype=own_sd[k].dtype))
                    copied += 1
        return copied

    def inject_expert_lora(self) -> List[nn.Parameter]:
        """Inject regularized Step-0 `0.0`-diff LoRA (`r=16, dropout=0.05`) into `ActionExpert-300M` while keeping dense base frozen."""
        added: List[nn.Parameter] = []
        for block in self.pi05.paligemma_with_expert.layers:
            for p in block.expert.parameters():
                p.requires_grad = False
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(block.expert, attr)
                if isinstance(mod, nn.Linear):
                    wrapped = ExpertDropoutLoRALinear(
                        mod,
                        rank=self.cfg.expert_lora_rank,
                        alpha=self.cfg.expert_lora_alpha,
                        dropout=self.cfg.expert_lora_dropout,
                    )
                    setattr(block.expert, attr, wrapped)
                    added.extend([wrapped.lora_a, wrapped.lora_b])
            if self.cfg.lora_target_ffn:
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    mod = getattr(block.expert.mlp, attr)
                    if isinstance(mod, nn.Linear):
                        wrapped = ExpertDropoutLoRALinear(
                            mod,
                            rank=self.cfg.expert_lora_rank,
                            alpha=self.cfg.expert_lora_alpha,
                            dropout=self.cfg.expert_lora_dropout,
                        )
                        setattr(block.expert.mlp, attr, wrapped)
                        added.extend([wrapped.lora_a, wrapped.lora_b])
        self.has_expert_lora = True
        return added

    def _cast_mem_core_bf16(self) -> None:
        """Cast heavy projections & norms in `mem_core` to `bf16` while keeping GDN decay/scalar gates in `fp32`."""
        self.mem_core.to(torch.bfloat16)
        for name, param in self.mem_core.named_parameters():
            if any(
                k in name
                for k in (
                    "A_log",
                    "dt_bias",
                    "gamma_hop2",
                    "gamma_hybrid",
                    "gamma_detail",
                    "gamma_vlm",
                    "roi_role_emb",
                    "salience_gate",
                    "lr_bias",
                    "vlm_hist_slot_emb",
                )
            ):
                param.data = param.data.float()
        for _, buf in self.mem_core.named_buffers():
            if buf.is_floating_point():
                buf.data = buf.data.float()

    def configure_stage(
        self,
        stage: str,
        *,
        freeze_mem_in_stage_b: bool = True,
        freeze_hooks_in_stage_a: bool = False,
        enable_grad_ckpt: bool = True,
    ) -> Dict[str, int]:
        """Configure exact parameter freeze/unfreeze masks for Stage A, Stage B, or Stage C."""
        self.cfg.train_stage = stage  # type: ignore[assignment]
        if stage == "stage_a":
            self.pi05.freeze_backbone()
            if freeze_hooks_in_stage_a:
                for p in self.pi05.memory_hook_parameters():
                    p.requires_grad = False
            # Keep `mem_rms_norm_ffn.dense.bias == 0` ONLY on Track-1 `modul_residual` so `mem_mod_vec == 0`
            # gives exact `0.0` modulation — NEVER wipe pretrained FiLM bias on `mme_vla_exact` (`#249` #2)!
            if self.cfg.modulator_mode not in ("mme_vla_exact", "grafted_residual"):
                for name, p in self.pi05.named_parameters():
                    if name.endswith("mem_rms_norm_ffn.dense.bias"):
                        p.data.zero_()
                        p.requires_grad = False
            self.pi05.set_gradient_checkpointing(False)
            for p in self.mem_core.parameters():
                p.requires_grad = True
            if self.mem_core.hop2_refiner is not None:
                for p in self.mem_core.hop2_refiner.parameters():
                    p.requires_grad = False
            if self.mem_core.crn_gate is not None:
                for p in self.mem_core.crn_gate.parameters():
                    p.requires_grad = False

        elif stage == "stage_b":
            self.pi05.freeze_backbone()
            if self.cfg.use_vlm_lora and not self.has_vlm_lora:
                self.pi05.inject_vlm_lora(
                    rank=self.cfg.lora_rank,
                    alpha=self.cfg.lora_alpha,
                    include_mlp=self.cfg.lora_target_ffn,
                )
                self.has_vlm_lora = True
            elif self.has_vlm_lora and self.cfg.use_vlm_lora:
                for name, p in self.pi05.named_parameters():
                    if ".vlm." in name and "lora_" in name:
                        p.requires_grad = True

            if self.cfg.use_expert_lora:
                if not self.has_expert_lora:
                    self.inject_expert_lora()
                else:
                    for name, p in self.pi05.named_parameters():
                        if ".expert." in name and "lora_" in name:
                            p.requires_grad = True
            elif not self.cfg.freeze_dense_expert_in_stage_b:
                for p in self.pi05.action_expert_parameters(include_io_Heads=True):
                    p.requires_grad = True

            for p in self.pi05.memory_hook_parameters():
                p.requires_grad = True
            if self.cfg.modulator_mode not in ("mme_vla_exact", "grafted_residual"):
                for name, p in self.pi05.named_parameters():
                    if name.endswith("mem_rms_norm_ffn.dense.bias"):
                        p.data.zero_()
                        p.requires_grad = False
            self.pi05.set_gradient_checkpointing(enable_grad_ckpt)

            for p in self.mem_core.parameters():
                p.requires_grad = not freeze_mem_in_stage_b
            # Keep zero-oracle aux / self-subgoal pointer head & hybrid bridge adaptable in Stage B!
            if self.mem_core.aux_suite is not None:
                for p in self.mem_core.aux_suite.parameters():
                    p.requires_grad = True
            if self.mem_core.hybrid_bridge is not None:
                for p in self.mem_core.hybrid_bridge.parameters():
                    p.requires_grad = True
            if self.mem_core.continuous_vlm_bridge is not None:
                for p in self.mem_core.continuous_vlm_bridge.parameters():
                    p.requires_grad = True
            if self.mem_core.vlm_surprise_selector is not None:
                for p in self.mem_core.vlm_surprise_selector.parameters():
                    p.requires_grad = True
            if self.mem_core.hop2_refiner is not None:
                for p in self.mem_core.hop2_refiner.parameters():
                    p.requires_grad = False
            if self.mem_core.crn_gate is not None:
                for p in self.mem_core.crn_gate.parameters():
                    p.requires_grad = False

        elif stage == "stage_c":
            if self.mem_core.hop2_refiner is None or self.mem_core.crn_gate is None:
                self.cfg.use_residual_hop2 = True
                dev = next(self.parameters()).device
                dt = next(self.mem_core.parameters()).dtype if list(self.mem_core.parameters()) else torch.float32
                self.mem_core.hop2_refiner = ResidualMultiHopRefiner(self.cfg).to(device=dev, dtype=dt)
                self.mem_core.hop2_refiner.gamma_hop2.data = self.mem_core.hop2_refiner.gamma_hop2.data.float()
                self.mem_core.crn_gate = StageCCRNGate(self.cfg).to(device=dev, dtype=dt)

            for p in self.parameters():
                p.requires_grad = False
            self.pi05.set_gradient_checkpointing(False)
            for p in self.mem_core.hop2_refiner.parameters():
                p.requires_grad = True
            for p in self.mem_core.crn_gate.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown stage={stage!r}; expected 'stage_a', 'stage_b', or 'stage_c'")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return {"trainable_params": n_train, "total_params": n_total}

    def build_optimizer_param_groups(
        self,
        *,
        lr_mem: float = 2e-4,
        lr_gate: float = 5e-4,
        lr_policy: float = 3e-5,
        lr_hooks: Optional[float] = None,
        stage_b_mem_lr_scale: float = 0.1,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """Differential AdamW parameter groups (`#249` Catch #1: dedicated `hooks` group never zeroed by `stage_b_mem_lr_scale`!)."""
        # On Track 2 (`mme_vla_exact`) & `P2-M7A` (`grafted_residual`), `mem_rms_norm_ffn.dense` is 80k-step pretrained
        # so keep `dense` in `hooks` (`lr_hooks`), while putting `graft_gate` (`fp32`) in `gates` (`lr_gate`)!
        if self.cfg.modulator_mode == "mme_vla_exact":
            gate_ids: set[int] = set()
        elif self.cfg.modulator_mode == "grafted_residual":
            gate_ids = {
                id(p) for n, p in self.pi05.named_parameters() if n.endswith(".graft_gate")
            }
        else:
            gate_ids = {id(p) for p in self.pi05.memory_gate_parameters()}
        for name, p in self.mem_core.named_parameters():
            if any(
                k in name
                for k in (
                    "gamma_hop2",
                    "gamma_hybrid",
                    "gamma_detail",
                    "gamma_vlm",
                    "roi_role_emb",
                    "salience_gate",
                    "gate_proj",
                    "boundary_gate",
                    "A_log",
                    "dt_bias",
                    "vlm_hist_slot_emb",
                )
            ):
                gate_ids.add(id(p))

        hook_ids = {id(p) for p in self.pi05.memory_hook_parameters()} - gate_ids
        mem_ids = {id(p) for p in self.mem_core.parameters()} - gate_ids

        gate_params: List[nn.Parameter] = []
        hook_decay: List[nn.Parameter] = []
        hook_nodecay: List[nn.Parameter] = []
        mem_params: List[nn.Parameter] = []
        policy_decay: List[nn.Parameter] = []
        policy_nodecay: List[nn.Parameter] = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in gate_ids:
                gate_params.append(p)
            elif pid in hook_ids:
                if p.ndim <= 1 or name.endswith(".bias") or "norm" in name:
                    hook_nodecay.append(p)
                else:
                    hook_decay.append(p)
            elif pid in mem_ids:
                mem_params.append(p)
            else:
                if p.ndim <= 1 or "lora_" in name or name.endswith(".bias") or "norm" in name:
                    policy_nodecay.append(p)
                else:
                    policy_decay.append(p)

        eff_lr_mem = (
            lr_mem * stage_b_mem_lr_scale if self.cfg.train_stage == "stage_b" else lr_mem
        )
        if lr_hooks is not None:
            eff_lr_hooks = float(lr_hooks)
        elif self.cfg.train_stage == "stage_b":
            eff_lr_hooks = (
                float(lr_policy)
                if self.cfg.modulator_mode == "mme_vla_exact"
                else max(float(lr_policy) * 2.5, 8e-5)
            )
        else:
            eff_lr_hooks = (
                min(float(lr_mem), 5e-5)
                if self.cfg.modulator_mode == "mme_vla_exact"
                else float(lr_mem)
            )

        hook_wd = (
            float(self.cfg.policy_weight_decay)
            if self.cfg.modulator_mode == "mme_vla_exact"
            else float(weight_decay)
        )

        groups: List[Dict[str, Any]] = []
        if gate_params:
            groups.append({"name": "gates", "params": gate_params, "lr": lr_gate, "weight_decay": 0.0})
        if mem_params:
            groups.append({"name": "memory", "params": mem_params, "lr": eff_lr_mem, "weight_decay": weight_decay})
        if hook_decay:
            groups.append({"name": "hooks_decay", "params": hook_decay, "lr": eff_lr_hooks, "weight_decay": hook_wd})
        if hook_nodecay:
            groups.append({"name": "hooks_nodecay", "params": hook_nodecay, "lr": eff_lr_hooks, "weight_decay": 0.0})
        if policy_decay:
            groups.append(
                {
                    "name": "policy_decay",
                    "params": policy_decay,
                    "lr": lr_policy,
                    "weight_decay": float(self.cfg.policy_weight_decay),
                }
            )
        if policy_nodecay:
            groups.append({"name": "policy_nodecay", "params": policy_nodecay, "lr": lr_policy, "weight_decay": 0.0})
        return groups

    def _compute_weighted_fm_loss(self, sq_err: torch.Tensor) -> torch.Tensor:
        """Reduce `[B, H, D]` squared flow error keeping all 32 ODE dims supervised (`#241`)."""
        d_act = int(sq_err.shape[-1])
        if self.cfg.loss_on_valid_8d_only or d_act <= 8:
            err8 = sq_err[:, :, :8]
            if d_act >= 8 and abs(self.cfg.gripper_dim_weight - 1.0) > 1e-6:
                w8 = torch.ones((1, 1, min(8, d_act)), device=sq_err.device, dtype=sq_err.dtype)
                w8[0, 0, 7] = float(self.cfg.gripper_dim_weight)
                return (err8 * w8).mean()
            return err8.mean()

        # Supervise all 32 ODE dimensions so `x_t[:, :, 8:32]` denoises cleanly to 0 across Euler steps:
        # dims 0..6 = 1.0, dim 7 (gripper) = gripper_dim_weight (1.5), dims 8..31 = padded_dim_loss_weight (0.25)
        w = torch.full((1, 1, d_act), float(self.cfg.padded_dim_loss_weight), device=sq_err.device, dtype=torch.float32)
        w[0, 0, :7] = 1.0
        w[0, 0, 7] = float(self.cfg.gripper_dim_weight)
        scale = float(self.cfg.fm_loss_scale) if self.cfg.train_stage == "stage_b" else 1.0
        return (sq_err.float() * w).mean() * scale

    def _project_vis_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """Project cached `1152-D` SigLIP tokens to `2048-D` on GPU via frozen `vision_tower.head`."""
        if x.shape[-1] == self.cfg.vis_dim:
            return x
        head = self.pi05.vision_tower.head
        if x.shape[-1] == head.in_features and head.out_features == self.cfg.vis_dim:
            with torch.no_grad():
                w_dt = head.weight.dtype
                return head(x.to(w_dt))
        raise ValueError(
            f"Visual feature dim {x.shape[-1]} matches neither cfg.vis_dim={self.cfg.vis_dim} "
            f"nor vision_tower.head ({head.in_features}->{head.out_features})"
        )

    def forward_training_episode_batch(
        self,
        episodes: Sequence[Dict[str, Any]],
        *,
        max_chunks_per_ep: int = 6,
        force_pytorch_gdn: bool = False,
    ) -> Dict[str, Any]:
        """Run a multi-episode training step across Stage A, Stage B, or Stage C."""
        dev = self.pi05.device
        stage = self.cfg.train_stage
        loss_dims = 8 if self.cfg.loss_on_valid_8d_only else self.cfg.padded_action_dim

        all_front_256: List[torch.Tensor] = []
        all_wrist_256: List[torch.Tensor] = []
        all_lang_tok: List[torch.Tensor] = []
        all_lang_mask: List[torch.Tensor] = []
        all_vlm_tok: List[torch.Tensor] = []
        all_vlm_mask: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        ep_slices: List[Tuple[int, int, Dict[str, Any], torch.Tensor]] = []
        aux_batches: Dict[str, List[torch.Tensor]] = {
            "target_xy": [],
            "has_xy": [],
            "secondary_target_xy": [],
            "has_secondary_xy": [],
            "completed_count": [],
            "remaining_count": [],
            "ordinal_index": [],
            "stage_id": [],
            "color_id": [],
        }

        cursor = 0
        for ep in episodes:
            m_tot = int(ep["chunk_step_indices"].numel())
            if m_tot > max_chunks_per_ep:
                perm = torch.randperm(m_tot)[:max_chunks_per_ep]
                sel = torch.sort(perm).values
            else:
                sel = torch.arange(m_tot, dtype=torch.long)

            m_sel = int(sel.numel())
            f_cur = (
                ep["exec_front_256"][sel].to(dev, non_blocking=True)
                if "exec_front_256" in ep and ep["exec_front_256"] is not None
                else ep["front_emb"][ep["chunk_step_indices"][sel]].to(dev, non_blocking=True)
            )
            w_cur = (
                ep["exec_wrist_256"][sel].to(dev, non_blocking=True)
                if "exec_wrist_256" in ep and ep["exec_wrist_256"] is not None
                else ep["wrist_emb"][ep["chunk_step_indices"][sel]].to(dev, non_blocking=True)
            )
            lt = ep["lang_tokens"].to(dev, non_blocking=True).unsqueeze(0).expand(m_sel, -1)
            lm = ep["lang_masks"].to(dev, non_blocking=True).unsqueeze(0).expand(m_sel, -1)
            if "chunk_vlm_tokens" in ep and ep["chunk_vlm_tokens"] is not None:
                vt = ep["chunk_vlm_tokens"][sel].to(dev, non_blocking=True)
                vm = ep["chunk_vlm_masks"][sel].to(dev, non_blocking=True)
            else:
                vt, vm = lt, lm
            act = ep["target_action_chunks"][sel].to(dev, non_blocking=True)

            all_front_256.append(f_cur)
            all_wrist_256.append(w_cur)
            all_lang_tok.append(lt)
            all_lang_mask.append(lm)
            all_vlm_tok.append(vt)
            all_vlm_mask.append(vm)
            all_actions.append(act)
            ep_slices.append((cursor, cursor + m_sel, ep, sel))
            cursor += m_sel

            if "chunk_target_xy_m11" in ep:
                aux_batches["target_xy"].append(ep["chunk_target_xy_m11"][sel].to(dev))
                aux_batches["has_xy"].append(ep["chunk_has_xy"][sel].to(dev))
            if "chunk_secondary_xy" in ep:
                sec_m11 = ep["chunk_secondary_xy"][sel].to(dev).float() * 2.0 - 1.0
                aux_batches["secondary_target_xy"].append(sec_m11)
                aux_batches["has_secondary_xy"].append(ep["chunk_has_secondary_xy"][sel].to(dev))
            if "chunk_count_state" in ep:
                cs = ep["chunk_count_state"][sel].to(dev)
                aux_batches["completed_count"].append(cs[:, 0])
                aux_batches["remaining_count"].append(cs[:, 1])
            if "chunk_ordinal_id" in ep:
                aux_batches["ordinal_index"].append(ep["chunk_ordinal_id"][sel].to(dev))
            if "chunk_phase_id" in ep:
                aux_batches["stage_id"].append(ep["chunk_phase_id"][sel].to(dev))
            if "chunk_color_id" in ep:
                aux_batches["color_id"].append(ep["chunk_color_id"][sel].to(dev))

        front_b = self._project_vis_if_needed(torch.cat(all_front_256, dim=0))
        wrist_b = self._project_vis_if_needed(torch.cat(all_wrist_256, dim=0))
        lang_tok_b = torch.cat(all_lang_tok, dim=0)
        lang_mask_b = torch.cat(all_lang_mask, dim=0)
        vlm_tok_b = torch.cat(all_vlm_tok, dim=0)
        vlm_mask_b = torch.cat(all_vlm_mask, dim=0)
        actions_b = torch.cat(all_actions, dim=0)
        total_b = int(actions_b.shape[0])

        aux_targets: Dict[str, torch.Tensor] = {
            k: torch.cat(v, dim=0) for k, v in aux_batches.items() if len(v) > 0
        }

        # Stage-B Spatial Sub-Grid Crop Augmentation (`RandomCrop 0.94..1.00`) + Exact Inverse `<x, y>`!
        if self.training and stage == "stage_b" and self.cfg.subgrid_crop_aug:
            theta = sample_subgrid_crop_affine(
                total_b,
                scale_range=(self.cfg.subgrid_scale_min, 1.0),
                max_extra_shift=self.cfg.subgrid_max_shift,
                device=dev,
            )
            sf = int(round(front_b.shape[1] ** 0.5))
            front_b = apply_subgrid_affine_to_tokens(front_b, theta, grid_hw=(sf, sf))
            if "target_xy" in aux_targets and "has_xy" in aux_targets:
                txy, hxy = transform_coords_under_affine(aux_targets["target_xy"], aux_targets["has_xy"], theta)
                aux_targets["target_xy"] = txy
                aux_targets["has_xy"] = hxy
            if "secondary_target_xy" in aux_targets and "has_secondary_xy" in aux_targets:
                sxy, hsxy = transform_coords_under_affine(
                    aux_targets["secondary_target_xy"], aux_targets["has_secondary_xy"], theta
                )
                aux_targets["secondary_target_xy"] = sxy
                aux_targets["has_secondary_xy"] = hsxy
            if self.cfg.feat_jitter_std > 0.0:
                f_std = front_b.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(front_b.dtype)
                w_std = wrist_b.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(wrist_b.dtype)
                front_b = front_b + torch.randn_like(front_b) * (self.cfg.feat_jitter_std * f_std)
                wrist_b = wrist_b + torch.randn_like(wrist_b) * (self.cfg.feat_jitter_std * w_std)

        # Prompt pooled embeddings from frozen PaliGemma token embedder (`no_grad`, `< 0.1 ms`)
        with torch.no_grad():
            t_emb = self.pi05.paligemma_with_expert.embed_language(lang_tok_b.long())
            lm_f = lang_mask_b.to(t_emb.dtype).unsqueeze(-1)
            prompt_pool_b = (t_emb * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)

        # Run per-episode causal memory forward (`O(W)` per episode)
        mem_l1_list: List[torch.Tensor] = []
        mem_l2_list: List[torch.Tensor] = []
        mem_mask_list: List[torch.Tensor] = []
        vlm_prefix_tok_list: List[torch.Tensor] = []
        vlm_prefix_mask_list: List[torch.Tensor] = []
        crn_logit_list: List[torch.Tensor] = []
        aux_pred_lists: Dict[str, List[torch.Tensor]] = {}
        max_b_reset = 0.0

        for s_i, e_i, ep, sel in ep_slices:
            ep_f = self._project_vis_if_needed(ep["front_emb"].to(dev, non_blocking=True))
            ep_w = self._project_vis_if_needed(ep["wrist_emb"].to(dev, non_blocking=True))
            ep_st = ep["state_norm"].to(dev, non_blocking=True)
            ep_dm = ep["is_demo"].to(dev, non_blocking=True)
            c_steps = ep["chunk_step_indices"][sel].to(dev)
            c_cuts = ep["readout_cutoff_excl"][sel].to(dev)
            cur_st = ep_st[c_steps]
            ep_pj = ep.get("prompt_cos_jump")
            if ep_pj is not None:
                ep_pj = ep_pj.to(dev, non_blocking=True)

            mem_out = self.mem_core.forward_episode_chunks(
                ep_front_emb=ep_f,
                ep_wrist_emb=ep_w,
                ep_state_norm=ep_st,
                ep_is_demo=ep_dm,
                exec_start_idx=int(ep["exec_start_idx"]),
                chunk_step_indices=c_steps,
                readout_cutoff_excl=c_cuts,
                curr_front_emb=front_b[s_i:e_i],
                curr_wrist_emb=wrist_b[s_i:e_i],
                curr_state_norm=cur_st,
                prompt_pool=prompt_pool_b[s_i:e_i],
                ep_prompt_cos_jump=ep_pj,
                perceptual_mem_encoder=self.pi05.mem_encoder,
                force_pytorch_gdn=force_pytorch_gdn,
                write_stride_exec_override=ep.get("write_stride_exec_override"),
                precision_insertpeg_climax=bool(ep.get("precision_insertpeg_climax", False)),
            )
            max_b_reset = max(max_b_reset, float(mem_out.get("boundary_reset_max", 0.0)))
            if mem_out["mem_tokens_l1"] is not None:
                use_ep_l2 = (
                    stage == "stage_b"
                    and bool(ep.get("use_hop2_l2_override", False))
                    and mem_out["mem_tokens_l2"] is not None
                )
                mem_l1_list.append(mem_out["mem_tokens_l2"] if use_ep_l2 else mem_out["mem_tokens_l1"])
                mem_mask_list.append(mem_out["mem_mask"])
            if mem_out.get("vlm_prefix_tokens") is not None and mem_out.get("vlm_prefix_mask") is not None:
                vlm_prefix_tok_list.append(mem_out["vlm_prefix_tokens"])
                vlm_prefix_mask_list.append(mem_out["vlm_prefix_mask"])
            if mem_out["mem_tokens_l2"] is not None:
                mem_l2_list.append(mem_out["mem_tokens_l2"])
            if mem_out["crn_logits"] is not None:
                crn_logit_list.append(mem_out["crn_logits"])
            if mem_out["aux_preds"] is not None:
                for ak, av in mem_out["aux_preds"].items():
                    if ak != "self_subgoal_tokens":
                        aux_pred_lists.setdefault(ak, []).append(av)

        mem_l1_b = torch.cat(mem_l1_list, dim=0) if mem_l1_list else None
        mem_mask_b = torch.cat(mem_mask_list, dim=0) if mem_mask_list else None
        if (
            self.mem_core.continuous_vlm_bridge is not None
            and mem_l1_b is not None
            and mem_mask_b is not None
            and any("chunk_cont_vlm_ground" in ep and ep["chunk_cont_vlm_ground"] is not None for _, _, ep, _ in ep_slices)
        ):
            cg_parts: Dict[str, List[torch.Tensor]] = {
                "front_vlm_8x8": [],
                "hm_maps": [],
                "roi_vecs": [],
                "sem_probs": [],
                "xy_coords": [],
                "active_mask": [],
            }
            v_dim = int(self.cfg.vlm_width)
            for s_i, e_i, ep, sel in ep_slices:
                m_ep = int(e_i - s_i)
                cg_ep = ep.get("chunk_cont_vlm_ground")
                if isinstance(cg_ep, dict):
                    for ck in cg_parts:
                        cg_parts[ck].append(cg_ep[ck][sel].to(dev, non_blocking=True))
                else:
                    cg_parts["front_vlm_8x8"].append(torch.zeros((m_ep, 64, v_dim), device=dev, dtype=mem_l1_b.dtype))
                    cg_parts["hm_maps"].append(torch.zeros((m_ep, 4, 256), device=dev, dtype=torch.float32))
                    cg_parts["roi_vecs"].append(torch.zeros((m_ep, 7, v_dim), device=dev, dtype=mem_l1_b.dtype))
                    cg_parts["sem_probs"].append(torch.zeros((m_ep, 48), device=dev, dtype=torch.float32))
                    cg_parts["xy_coords"].append(torch.zeros((m_ep, 8), device=dev, dtype=torch.float32))
                    cg_parts["active_mask"].append(torch.zeros((m_ep,), device=dev, dtype=torch.bool))
            cg_batch = {ck: torch.cat(cv, dim=0) for ck, cv in cg_parts.items()}
            mem_l1_b = self.mem_core.continuous_vlm_bridge(mem_l1_b, mem_mask_b, cg_batch)
        vlm_prefix_tok_b = torch.cat(vlm_prefix_tok_list, dim=0) if vlm_prefix_tok_list else None
        vlm_prefix_mask_b = torch.cat(vlm_prefix_mask_list, dim=0) if vlm_prefix_mask_list else None

        if (
            self.training
            and stage == "stage_b"
            and vlm_prefix_mask_b is not None
            and float(self.cfg.vlm_prefix_drop_prob) > 0.0
        ):
            drop_vp = torch.rand((total_b,), device=dev) < float(self.cfg.vlm_prefix_drop_prob)
            if bool(drop_vp.any()):
                vlm_prefix_mask_b = vlm_prefix_mask_b.clone()
                vlm_prefix_mask_b[drop_vp] = False

        # Auxiliary grounded losses (`0` oracle at test time!)
        aux_loss_dict: Dict[str, torch.Tensor] = {
            "aux_total": torch.tensor(0.0, device=dev),
            "xy_loss": torch.tensor(0.0, device=dev),
            "count_loss": torch.tensor(0.0, device=dev),
            "subgoal_loss": torch.tensor(0.0, device=dev),
        }
        if self.mem_core.aux_suite is not None and aux_pred_lists and stage in ("stage_a", "stage_b"):
            merged_preds = {k: torch.cat(v, dim=0) for k, v in aux_pred_lists.items()}
            aux_loss_dict = self.mem_core.aux_suite.compute_aux_losses(merged_preds, aux_targets)

        # Stage-B `Mem-CFG` Condition-Masking Dropout (`p_mem_drop = 0.10` -> `mem_mask[drop] = False`, Track-1 only!)
        if (
            self.training
            and stage == "stage_b"
            and mem_mask_b is not None
            and self.cfg.mem_cfg_dropout > 0.0
            and self.cfg.modulator_mode != "mme_vla_exact"
        ):
            drop_rows = torch.rand((total_b,), device=dev) < float(self.cfg.mem_cfg_dropout)
            if bool(drop_rows.any()):
                mem_mask_b = mem_mask_b.clone()
                mem_mask_b[drop_rows] = False

        # ---------------------------------------------------------------------
        # Stage A: Frozen VLM+AE (`no_grad` KV prefill) -> ultra-fast ActionExpert loss
        # ---------------------------------------------------------------------
        if stage == "stage_a":
            with torch.no_grad():
                kv_cache = self.pi05.prefill_prefix_kv(
                    vlm_tok_b,
                    vlm_mask_b,
                    image_embs=[front_b, wrist_b],
                    extra_prefix_embs=vlm_prefix_tok_b,
                    extra_prefix_mask=vlm_prefix_mask_b,
                    return_prefix_hidden=False,
                )
            sq_err = self.pi05.compute_loss_from_kv(
                kv_cache,  # type: ignore[arg-type]
                actions_b,
                mem_tokens=mem_l1_b,
                mem_mask=mem_mask_b,
                loss_dims=loss_dims,
            )
            fm_loss = self._compute_weighted_fm_loss(sq_err)
            total_loss = fm_loss + aux_loss_dict["aux_total"]
            return {
                "loss": total_loss,
                "fm_loss": fm_loss.detach(),
                "boundary_reset_max": max_b_reset,
                **aux_loss_dict,
            }

        # ---------------------------------------------------------------------
        # Stage B: Alternating Co-Adaptation (`ActionExpert-300M` + `PaliGemma-2B` LoRA)
        # ---------------------------------------------------------------------
        if stage == "stage_b":
            if self.has_vlm_lora or vlm_prefix_tok_b is not None:
                sq_err = self.pi05.forward_joint(
                    actions_b,
                    vlm_tok_b,
                    vlm_mask_b,
                    image_embs=[front_b, wrist_b],
                    extra_prefix_embs=vlm_prefix_tok_b,
                    extra_prefix_mask=vlm_prefix_mask_b,
                    mem_tokens=mem_l1_b,
                    mem_mask=mem_mask_b,
                    loss_dims=loss_dims,
                )
            else:
                with torch.no_grad():
                    kv_cache = self.pi05.prefill_prefix_kv(
                        vlm_tok_b, vlm_mask_b, image_embs=[front_b, wrist_b], return_prefix_hidden=False
                    )
                sq_err = self.pi05.compute_loss_from_kv(
                    kv_cache,  # type: ignore[arg-type]
                    actions_b,
                    mem_tokens=mem_l1_b,
                    mem_mask=mem_mask_b,
                    loss_dims=loss_dims,
                )
            fm_loss = self._compute_weighted_fm_loss(sq_err)
            total_loss = fm_loss + float(self.cfg.stage_b_aux_scale) * aux_loss_dict["aux_total"]
            return {
                "loss": total_loss,
                "fm_loss": fm_loss.detach(),
                "boundary_reset_max": max_b_reset,
                **aux_loss_dict,
            }

        # ---------------------------------------------------------------------
        # Stage C: Residual Hop-2 + `M_CRN = 8` Common-Random-Number Halting Gate
        # ---------------------------------------------------------------------
        assert mem_l2_list and crn_logit_list and self.mem_core.crn_gate is not None
        mem_l2_b = torch.cat(mem_l2_list, dim=0)
        crn_logits_b = torch.cat(crn_logit_list, dim=0)

        with torch.no_grad():
            kv_cache = self.pi05.prefill_prefix_kv(
                vlm_tok_b,
                vlm_mask_b,
                image_embs=[front_b, wrist_b],
                extra_prefix_embs=vlm_prefix_tok_b,
                extra_prefix_mask=vlm_prefix_mask_b,
                return_prefix_hidden=False,
            )

        # Primary Hop-2 FM loss on 1 draw (with gradients to `hop2_refiner`) + `M_CRN - 1` extra `no_grad` draws!
        m_crn = max(1, int(self.cfg.crn_draws))
        noise_0 = self.pi05.sample_noise(total_b, dev)
        time_0 = self.pi05.sample_time(total_b, dev)

        with torch.no_grad():
            err_l1_0 = self.pi05.compute_loss_from_kv(
                kv_cache,  # type: ignore[arg-type]
                actions_b,
                noise=noise_0,
                time=time_0,
                mem_tokens=mem_l1_b,
                mem_mask=mem_mask_b,
                loss_dims=loss_dims,
            ).mean(dim=(1, 2))

        err_l2_0_full = self.pi05.compute_loss_from_kv(
            kv_cache,  # type: ignore[arg-type]
            actions_b,
            noise=noise_0,
            time=time_0,
            mem_tokens=mem_l2_b,
            mem_mask=mem_mask_b,
            loss_dims=loss_dims,
        )
        fm_loss_hop2 = self._compute_weighted_fm_loss(err_l2_0_full)

        l1_acc = err_l1_0.clone()
        l2_acc = err_l2_0_full.detach().mean(dim=(1, 2))
        if m_crn > 1:
            with torch.no_grad():
                for _ in range(m_crn - 1):
                    ns = self.pi05.sample_noise(total_b, dev)
                    tm = self.pi05.sample_time(total_b, dev)
                    l1_acc = l1_acc + self.pi05.compute_loss_from_kv(
                        kv_cache,  # type: ignore[arg-type]
                        actions_b,
                        noise=ns,
                        time=tm,
                        mem_tokens=mem_l1_b,
                        mem_mask=mem_mask_b,
                        loss_dims=loss_dims,
                    ).mean(dim=(1, 2))
                    l2_acc = l2_acc + self.pi05.compute_loss_from_kv(
                        kv_cache,  # type: ignore[arg-type]
                        actions_b,
                        noise=ns,
                        time=tm,
                        mem_tokens=mem_l2_b.detach(),
                        mem_mask=mem_mask_b,
                        loss_dims=loss_dims,
                    ).mean(dim=(1, 2))

        adv = (l1_acc - l2_acc) / float(m_crn)  # Positive iff Hop-2 beats Hop-1 on paired CRNs!
        q_thresh = max(
            float(self.cfg.crn_min_delta),
            float(torch.quantile(adv, float(self.cfg.crn_quantile)).item()),
        )
        self.mem_core.crn_gate.calibrated_delta.fill_(q_thresh)
        target_hop2 = (adv > q_thresh).float()
        gate_loss = F.binary_cross_entropy_with_logits(crn_logits_b, target_hop2)

        total_loss = fm_loss_hop2 + 0.1 * gate_loss
        return {
            "loss": total_loss,
            "fm_loss": fm_loss_hop2.detach(),
            "fm_loss_l1": (l1_acc / float(m_crn)).mean().detach(),
            "gate_loss": gate_loss.detach(),
            "hop2_rate": (torch.sigmoid(crn_logits_b.detach()) > 0.5).float().mean(),
            "gamma_hop2": float(torch.tanh(self.mem_core.hop2_refiner.gamma_hop2.detach()).item()),
            "boundary_reset_max": max_b_reset,
            **aux_loss_dict,
        }

    # =========================================================================
    # Compact Delta Checkpoint Save / Load (`~85 MB` Stage A, `~680 MB` Stage B)
    # =========================================================================

    def export_delta_state_dict(self) -> Dict[str, Any]:
        """Export only `mem_core` + modified/hook/LoRA/Expert weights (never duplicating frozen 2B VLM!)."""
        pi05_delta: Dict[str, torch.Tensor] = {}
        hook_param_names = {
            n
            for n, _ in self.pi05.named_parameters()
            if (
                "mem_attn" in n
                or "mem_rms_norm_ffn" in n
                or "lora_" in n
                or (self.cfg.arch == "p2_m7_grounded_hybrid" and n.startswith("mem_encoder."))
            )
        }
        include_expert = (
            self._has_custom_dense_expert
            or (
                (self.cfg.train_stage in ("stage_b", "stage_c"))
                and not (self.has_expert_lora or self.cfg.freeze_dense_expert_in_stage_b)
            )
        )
        for name, param in self.pi05.named_parameters():
            if (
                param.requires_grad
                or name in hook_param_names
                or (
                    include_expert
                    and (
                        ".expert." in name
                        or "expert_norm." in name
                        or name.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_"))
                    )
                )
            ):
                pi05_delta[name] = param.detach().cpu()

        return {
            "pi05_mem_config": self.cfg.to_dict(),
            "has_vlm_lora": self.has_vlm_lora,
            "has_expert_lora": self.has_expert_lora,
            "mem_core": {k: v.detach().cpu() for k, v in self.mem_core.state_dict().items()},
            "pi05_delta": pi05_delta,
        }

    def load_delta_state_dict(
        self,
        payload: Dict[str, Any],
        *,
        strict_mem: bool = False,
        skip_lora: bool = False,
    ) -> None:
        """Restore `mem_core` + `pi05_delta` (activating hooks & LoRA automatically if present)."""
        dev = self.pi05.device
        if (not skip_lora) and bool(payload.get("has_vlm_lora", False)) and not self.has_vlm_lora:
            self.pi05.inject_vlm_lora(
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                include_mlp=self.cfg.lora_target_ffn,
            )
            self.pi05.to(dev)
            self.has_vlm_lora = True

        if (not skip_lora) and bool(payload.get("has_expert_lora", False)) and not self.has_expert_lora:
            self.inject_expert_lora()
            self.pi05.to(dev)

        if payload.get("mem_core"):
            if self.cfg.use_residual_hop2 and self.mem_core.hop2_refiner is None:
                dt = next(self.mem_core.parameters()).dtype
                self.mem_core.hop2_refiner = ResidualMultiHopRefiner(self.cfg).to(device=dev, dtype=dt)
                self.mem_core.hop2_refiner.gamma_hop2.data = self.mem_core.hop2_refiner.gamma_hop2.data.float()
                self.mem_core.crn_gate = StageCCRNGate(self.cfg).to(device=dev, dtype=dt)
            if self.cfg.use_continuous_vlm_ground and self.mem_core.continuous_vlm_bridge is None:
                dt = next(self.mem_core.parameters()).dtype
                self.mem_core.continuous_vlm_bridge = ContinuousVLMGroundingBridge(self.cfg).to(device=dev, dtype=dt)
                self.mem_core.continuous_vlm_bridge.gamma_vlm.data = (
                    self.mem_core.continuous_vlm_bridge.gamma_vlm.data.float()
                )
                self.mem_core.continuous_vlm_bridge.gamma_vlm_direct.data = (
                    self.mem_core.continuous_vlm_bridge.gamma_vlm_direct.data.float()
                )
                self.mem_core.continuous_vlm_bridge.roi_role_emb.data = (
                    self.mem_core.continuous_vlm_bridge.roi_role_emb.data.float()
                )
            self.mem_core.load_state_dict(payload["mem_core"], strict=strict_mem)

        pi05_delta: Dict[str, torch.Tensor] = payload.get("pi05_delta", {})
        if pi05_delta:
            own_sd = self.pi05.state_dict()
            for k, v in pi05_delta.items():
                if skip_lora and "lora_" in k:
                    continue
                if "action_out_proj." in k or "action_in_proj." in k or (".expert." in k and "lora_" not in k):
                    self._has_custom_dense_expert = True
                if k in own_sd:
                    own_sd[k].copy_(v.to(device=own_sd[k].device, dtype=own_sd[k].dtype))
                elif k.endswith(".weight") and (k[:-7] + ".base.weight") in own_sd:
                    dst_k = k[:-7] + ".base.weight"
                    own_sd[dst_k].copy_(v.to(device=own_sd[dst_k].device, dtype=own_sd[dst_k].dtype))
                elif k.endswith(".base.weight") and (k[:-12] + ".weight") in own_sd:
                    dst_k = k[:-12] + ".weight"
                    own_sd[dst_k].copy_(v.to(device=own_sd[dst_k].device, dtype=own_sd[dst_k].dtype))
                elif k.endswith(".base.bias") and (k[:-10] + ".bias") in own_sd:
                    dst_k = k[:-10] + ".bias"
                    own_sd[dst_k].copy_(v.to(device=own_sd[dst_k].device, dtype=own_sd[dst_k].dtype))
