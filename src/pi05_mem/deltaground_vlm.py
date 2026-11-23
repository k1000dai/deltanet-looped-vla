"""`P2-M9: DeltaGround-VLM` (`src/pi05_mem/deltaground_vlm.py`) — Pretrained `PaliGemma-2B` Grounded Subgoal & `16x16` Pointer Engine (`#334`).

Eliminates `<40M` scratch `DeltaGround`'s unseen-scene container/color layout memorization (`VideoUnmaskSwap`)
with `0 MiB` extra base VLM VRAM by running a lightweight `dg_vlm_lora (r=16, dropout=0.10)` + causal
`MultiTimescale2LayerGDN` (`EpisodeBoundaryForgetGate` `k=3` isolated) + `K=6` chronological `8x8`
`GDN-Surprise` reveal keyframes + `16x16` Peak-Window `<row_y, col_x>` Pointer Head directly on
`perceptual_framesamp_modul_pt`'s shared `2.6B` `PaliGemma-2B` VLM:

  1. **Prefix Sequence (`[front_256, wrist_16, task_lang_64, reveal_6x64 (384t), gdn_state_24]`, `prefix_ar_mask == 0`):**
     - `front_256`: `16x16` current front `SigLIP` tokens (`2048-D`, RoPE `0..255`).
     - `wrist_16`: `4x4` pooled current wrist `SigLIP` tokens (`2048-D` + zero-init `wrist_view_emb`).
     - `task_lang_64`: `PaliGemma-2B` subword embeddings of `"Task: ..."` (`pretrained 2.6B English <-> SigLIP alignment!`).
     - `reveal_384`: `K = 6` chronological `8x8` `GDN-Surprise` reveal/highlight/demo frames (`demo_quota=4` when
       `demo > 0`, `early_quota=2` pure visual `s_norm` when `demo == 0`, temporal NMS, zero-init `vlm_hist_slot_emb`).
     - `gdn_state_24`: `22` `MultiTimescale2LayerGDN` (`Short/Mid/Long` + persistent) readout tokens projected to `2048-D`
       + `2` 18-D `[state_norm, Δstate_norm, |Δgripper|, arm_speed]` contact kinematics tokens.
  2. **Zero-Copy Dual-LoRA Hot-Swapping (`DualSwitchableLoRALinear`, `< 60 MiB` VRAM, `0` Retraining of `P2-M7B` `s300`!):**
     - Holds both `policy` VLM LoRA (`P2-M7B` `r=32`) and `dg` VLM LoRA (`P2-M9` `r=16, dropout=0.10`) on the
       shared frozen `PaliGemma-2B` linear layers, toggling `"dg"` during `predict_subgoal` and `"policy"` (or `"none"`
       for `symbolic_grounded_subgoal_pt`) during action inference in `< 0.01 ms` with zero weight copies!
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.subgoal_grounding import norm_xy_to_pixel_xy, spatial_softmax_2d
from models.deltaloop_vla import SwiGLU, build_1d_sincos_embed, run_dual_backend_gdn
from pi05 import load_pi05_checkpoint
from pi05.model import Pi05Model, make_attn_mask
from pi05_data.dataset import load_pi05_shards
from robomme_data.constants import ROBOMME_TASKS
from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS

from .config import Pi05MemConfig
from .deltaground import (
    StageEntrySubgoalLatch,
    _drop_shard_page_cache,
    apply_peak_window_to_preds,
    build_occlusion_twin_template_pairs,
    compute_twin_summed_max_prob,
    fold_occlusion_twin_logits,
    is_ordinal_counting_template,
    render_grounded_subgoal_from_norm_xy,
    should_abstain_by_task_goal,
    should_route_to_reactive_precision,
    should_route_to_symbolic_specialist,
)
from .memory_modules import (
    DualViewEventPerceiverWriter,
    MultiTimescale2LayerGDN,
    VLMPrefixSurpriseSelector,
    pool_spatial_tokens_to_square,
    select_episode_write_indices,
)

LoRAMode = Literal["policy", "base_policy", "reactive_policy", "dg", "none"]


@dataclass
class DeltaGroundVLMConfig:
    """Configuration for `P2-M9: DeltaGround-VLM` sharing `PaliGemma-2B`."""

    vis_dim: int = 2048
    vlm_width: int = 2048
    front_tokens_hist: int = 64
    wrist_tokens_hist: int = 16
    front_tokens_curr: int = 256
    wrist_tokens_curr_pool: int = 16
    max_prompt_len: int = 64
    state_dim: int = 8

    write_stride_exec: int = 2
    write_stride_demo: int = 2
    adaptive_demo_threshold: int = 512
    long_demo_stride: int = 4
    exclusion_window: int = 0

    d_mem: int = 512
    mem_enc_heads: int = 8
    num_write_slots: int = 8
    slot_conv_kernel: int = 4
    gdn_heads: int = 12
    gdn_d_k: int = 64
    gdn_d_v: int = 64
    num_gdn_readout_tokens: int = 18
    num_persistent_tokens: int = 4
    boundary_flush_threshold: float = 0.5
    boundary_reset_strength: float = 12.0
    surprise_ema_beta: float = 0.80

    reveal_keyframes: int = 6
    reveal_tokens_per_frame: int = 64
    reveal_demo_quota: int = 4
    reveal_early_quota: int = 2

    dg_lora_rank: int = 16
    dg_lora_alpha: float = 16.0
    dg_lora_dropout: float = 0.10
    dg_lora_include_mlp: bool = True

    d_head: int = 512
    head_dropout: float = 0.10
    num_templates: int = 133
    max_count_classes: int = 12
    num_stage_classes: int = 15
    num_color_classes: int = 9

    # `P2-M9-v2` (`#367` / `#368` / `#376`) Shortcut-Free, Swap-Complete, 2D-Aligned Architecture (`False` = bit-exact `v1`!)
    v2_mode: bool = False
    v2_uniform_anchors: int = 7
    v2_front_drop: float = 0.10
    v2_non_entry_xy_weight: float = 0.0
    v2_coda_heads: int = 8
    v2_feat_noise_std: float = 0.02
    v2_patch_drop_prob: float = 0.04
    v2_keep_exec_clock: bool = False
    v2_raw_siglip_reveal: bool = False
    v2_apex_peaks: int = 0
    v2_hr16_sincos: bool = False
    v2_pyramid_apex_hr16: int = 0
    xy_loss_weight: float = 1.0
    # `#426` Wave-16 (`P2-M9v5`): Bug #1 two-sided transient prominence + Bug #2 causal patch-flow reveal advection
    v2_reveal_prominence_tau: int = 0
    v2_advect_reveal: bool = False
    v2_advect_radius: int = 1
    v2_advect_beta: float = 20.0
    v2_advect_id_bias: float = 1.0
    # Phase-3 (`DeltaGround-VLM v3 + CORM + ReFINE`): session-median reveal selector (`t=0` included) + transition weighting
    v3_mode: bool = False
    v3_transition_ce_boost: float = 2.5

    def to_pi05_mem_cfg(self) -> Pi05MemConfig:
        return Pi05MemConfig(
            arch="p2_m8_vlmprefix_surprise_hybrid",
            vlm_width=self.vlm_width,
            vis_dim=self.vis_dim,
            front_tokens_hist=self.front_tokens_hist,
            wrist_tokens_hist=self.wrist_tokens_hist,
            front_tokens_curr=self.front_tokens_curr,
            wrist_tokens_curr=self.front_tokens_curr,
            state_dim=self.state_dim,
            exclusion_window=self.exclusion_window,
            write_stride_exec=self.write_stride_exec,
            write_stride_demo=self.write_stride_demo,
            adaptive_demo_threshold=self.adaptive_demo_threshold,
            long_demo_stride=self.long_demo_stride,
            d_mem=self.d_mem,
            mem_enc_heads=self.mem_enc_heads,
            num_write_slots=self.num_write_slots,
            slot_conv_kernel=self.slot_conv_kernel,
            gdn_heads=self.gdn_heads,
            gdn_d_k=self.gdn_d_k,
            gdn_d_v=self.gdn_d_v,
            num_gdn_readout_tokens=self.num_gdn_readout_tokens,
            num_persistent_tokens=self.num_persistent_tokens,
            use_landmark_bank=False,
            mem_token_dim=self.d_mem,
            boundary_flush_threshold=self.boundary_flush_threshold,
            boundary_reset_strength=self.boundary_reset_strength,
            surprise_ema_beta=self.surprise_ema_beta,
            vlm_prefix_surprise_keyframes=self.reveal_keyframes,
            vlm_prefix_tokens_per_frame=self.reveal_tokens_per_frame,
            vlm_prefix_demo_quota=self.reveal_demo_quota,
            vlm_prefix_early_quota=self.reveal_early_quota,
            vlm_prefix_drop_prob=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeltaGroundVLMConfig":
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def tiny_test_preset(cls, **overrides: Any) -> "DeltaGroundVLMConfig":
        kw: Dict[str, Any] = dict(
            vis_dim=128,
            vlm_width=128,
            front_tokens_hist=64,
            wrist_tokens_hist=16,
            front_tokens_curr=64,
            wrist_tokens_curr_pool=16,
            max_prompt_len=24,
            d_mem=96,
            mem_enc_heads=4,
            num_write_slots=4,
            gdn_heads=6,
            gdn_d_k=32,
            gdn_d_v=32,
            num_gdn_readout_tokens=9,
            num_persistent_tokens=2,
            reveal_keyframes=2,
            reveal_tokens_per_frame=16,
            reveal_demo_quota=1,
            reveal_early_quota=1,
            dg_lora_rank=8,
            dg_lora_alpha=8.0,
            dg_lora_dropout=0.0,
            d_head=96,
            head_dropout=0.0,
            num_templates=12,
        )
        kw.update(overrides)
        return cls(**kw)


class DualSwitchableLoRALinear(nn.Module):
    """Zero-copy dual-adapter LoRA wrapper on `PaliGemma-2B` supporting `"policy"`, `"dg"`, and `"none"` (`#334`)."""

    def __init__(
        self,
        existing_mod: nn.Module,
        *,
        dg_rank: int = 16,
        dg_alpha: float = 16.0,
        dg_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if isinstance(existing_mod, DualSwitchableLoRALinear):
            self.base = existing_mod.base
            self.has_policy_lora = existing_mod.has_policy_lora
            self.policy_drop: nn.Module = existing_mod.policy_drop
            if self.has_policy_lora:
                self.lora_a = existing_mod.lora_a
                self.lora_b = existing_mod.lora_b
                self.scaling = existing_mod.scaling
            else:
                self.register_parameter("lora_a", None)
                self.register_parameter("lora_b", None)
                self.scaling = 1.0
        elif hasattr(existing_mod, "base") and hasattr(existing_mod, "lora_a") and hasattr(existing_mod, "lora_b"):
            self.base = existing_mod.base  # type: ignore[assignment]
            self.has_policy_lora = True
            self.policy_drop = getattr(existing_mod, "lora_drop", getattr(existing_mod, "drop", nn.Identity()))
            self.lora_a = existing_mod.lora_a  # type: ignore[assignment]
            self.lora_b = existing_mod.lora_b  # type: ignore[assignment]
            self.scaling = float(getattr(existing_mod, "scaling", 1.0))
        elif isinstance(existing_mod, nn.Linear):
            self.base = existing_mod
            self.has_policy_lora = False
            self.policy_drop = nn.Identity()
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
            self.scaling = 1.0
        else:
            raise TypeError(f"Unsupported module for DualSwitchableLoRALinear: {type(existing_mod)}")

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = int(self.base.in_features)
        out_f = int(self.base.out_features)
        dev = self.base.weight.device
        dt = self.base.weight.dtype
        self.dg_rank = max(0, int(dg_rank))
        self.dg_scaling = float(dg_alpha) / float(max(1, self.dg_rank)) if self.dg_rank > 0 else 0.0
        self.dg_drop = nn.Dropout(float(dg_dropout)) if (dg_dropout > 0.0 and self.dg_rank > 0) else nn.Identity()
        if self.dg_rank > 0:
            self.dg_lora_a = nn.Parameter(torch.empty((in_f, self.dg_rank), device=dev, dtype=dt))
            self.dg_lora_b = nn.Parameter(torch.zeros((self.dg_rank, out_f), device=dev, dtype=dt))
            nn.init.normal_(self.dg_lora_a, mean=0.0, std=1.0 / math.sqrt(float(in_f)))
        else:
            self.register_parameter("dg_lora_a", None)
            self.register_parameter("dg_lora_b", None)
        prev_base_a = existing_mod.base_lora_a if isinstance(existing_mod, DualSwitchableLoRALinear) else None
        prev_base_b = existing_mod.base_lora_b if isinstance(existing_mod, DualSwitchableLoRALinear) else None
        prev_base_sc = float(existing_mod.base_scaling) if isinstance(existing_mod, DualSwitchableLoRALinear) else float(self.scaling)
        prev_reac_a = existing_mod.reactive_lora_a if isinstance(existing_mod, DualSwitchableLoRALinear) else None
        prev_reac_b = existing_mod.reactive_lora_b if isinstance(existing_mod, DualSwitchableLoRALinear) else None
        prev_reac_sc = float(existing_mod.reactive_scaling) if isinstance(existing_mod, DualSwitchableLoRALinear) else float(self.scaling)
        self.register_buffer("base_lora_a", prev_base_a, persistent=False)
        self.register_buffer("base_lora_b", prev_base_b, persistent=False)
        self.base_scaling: float = prev_base_sc
        self.register_buffer("reactive_lora_a", prev_reac_a, persistent=False)
        self.register_buffer("reactive_lora_b", prev_reac_b, persistent=False)
        self.reactive_scaling: float = prev_reac_sc
        self.active_mode: LoRAMode = "dg"

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        mode = self.active_mode
        if mode == "dg":
            if self.dg_lora_a is not None and self.dg_lora_b is not None:
                dx = self.dg_drop(x.to(self.dg_lora_a.dtype))
                delta = torch.matmul(torch.matmul(dx, self.dg_lora_a), self.dg_lora_b) * self.dg_scaling
                return out + delta.to(out.dtype)
            return out
        if mode == "policy" and self.has_policy_lora and self.lora_a is not None and self.lora_b is not None:
            px = self.policy_drop(x.to(self.lora_a.dtype))
            delta = torch.matmul(torch.matmul(px, self.lora_a), self.lora_b) * self.scaling
            return out + delta.to(out.dtype)
        if mode == "base_policy" and self.base_lora_a is not None and self.base_lora_b is not None:
            bx = x.to(self.base_lora_a.dtype)
            delta = torch.matmul(torch.matmul(bx, self.base_lora_a), self.base_lora_b) * self.base_scaling
            return out + delta.to(out.dtype)
        if mode == "reactive_policy" and self.reactive_lora_a is not None and self.reactive_lora_b is not None:
            rx = x.to(self.reactive_lora_a.dtype)
            delta = torch.matmul(torch.matmul(rx, self.reactive_lora_a), self.reactive_lora_b) * self.reactive_scaling
            return out + delta.to(out.dtype)
        return out


def inject_dg_vlm_lora(
    pi05_model: Pi05Model,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.10,
    include_mlp: bool = True,
) -> List[nn.Parameter]:
    """Attach `dg_lora_{a,b}` to `PaliGemma` Stream-0 while wrapping ALL 7 projections + optional `ActionExpert` LoRA (`#381`/`#391`)!"""
    added: List[nn.Parameter] = []
    mlp_r = int(rank) if include_mlp else 0
    for block in pi05_model.paligemma_with_expert.layers:
        vlm = block.vlm
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            wrapped = DualSwitchableLoRALinear(
                getattr(vlm, attr), dg_rank=rank, dg_alpha=alpha, dg_dropout=dropout
            )
            setattr(vlm, attr, wrapped)
            if wrapped.dg_lora_a is not None and wrapped.dg_lora_b is not None:
                added.extend([wrapped.dg_lora_a, wrapped.dg_lora_b])
        for attr in ("gate_proj", "up_proj", "down_proj"):
            wrapped = DualSwitchableLoRALinear(
                getattr(vlm.mlp, attr), dg_rank=mlp_r, dg_alpha=alpha, dg_dropout=dropout
            )
            setattr(vlm.mlp, attr, wrapped)
            if wrapped.dg_lora_a is not None and wrapped.dg_lora_b is not None:
                added.extend([wrapped.dg_lora_a, wrapped.dg_lora_b])
        # `#391(2)` `P2-M7E`: If `ActionExpert` has `policy` LoRA, wrap with `dg_rank=0` so `"base_policy"` = `0.0` = bit-exact `v3_s250`!
        exp = getattr(block, "expert", None)
        if exp is not None:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(exp, attr, None)
                if mod is not None and (hasattr(mod, "lora_a") or isinstance(mod, DualSwitchableLoRALinear)):
                    setattr(exp, attr, DualSwitchableLoRALinear(mod, dg_rank=0, dg_alpha=0.0, dg_dropout=0.0))
            exp_mlp = getattr(exp, "mlp", None)
            if exp_mlp is not None:
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    mod = getattr(exp_mlp, attr, None)
                    if mod is not None and (hasattr(mod, "lora_a") or isinstance(mod, DualSwitchableLoRALinear)):
                        setattr(exp_mlp, attr, DualSwitchableLoRALinear(mod, dg_rank=0, dg_alpha=0.0, dg_dropout=0.0))
    return added


def set_vlm_lora_mode(pi05_model: Pi05Model, mode: LoRAMode) -> None:
    """Switch active LoRA (`"dg"`, `"policy"`, `"base_policy"`, or `"none"`) across all 18 `PaliGemma` + `ActionExpert` layers in `< 0.01 ms`."""
    for block in pi05_model.paligemma_with_expert.layers:
        vlm = block.vlm
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(vlm, attr)
            if isinstance(mod, DualSwitchableLoRALinear):
                mod.active_mode = mode
        for attr in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(vlm.mlp, attr)
            if isinstance(mod, DualSwitchableLoRALinear):
                mod.active_mode = mode
        exp = getattr(block, "expert", None)
        if exp is not None:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(exp, attr, None)
                if isinstance(mod, DualSwitchableLoRALinear):
                    mod.active_mode = mode
            exp_mlp = getattr(exp, "mlp", None)
            if exp_mlp is not None:
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    mod = getattr(exp_mlp, attr, None)
                    if isinstance(mod, DualSwitchableLoRALinear):
                        mod.active_mode = mode


def attach_base_vlm_lora(pi05_model: Pi05Model, base_lora_sd: Mapping[str, torch.Tensor]) -> int:
    """Attach frozen `v3_s250` `base_lora_{a,b}` buffers so `"base_policy"` matches `v3_s250` `0.000e+00` bit-for-bit!"""
    n_attached = 0
    dev = pi05_model.device
    for li, block in enumerate(pi05_model.paligemma_with_expert.layers):
        vlm = block.vlm
        for sub_path, holder in (
            ("q_proj", vlm),
            ("k_proj", vlm),
            ("v_proj", vlm),
            ("o_proj", vlm),
            ("mlp.gate_proj", vlm.mlp),
            ("mlp.up_proj", vlm.mlp),
            ("mlp.down_proj", vlm.mlp),
        ):
            attr = sub_path.split(".")[-1]
            mod = getattr(holder, attr)
            if not isinstance(mod, DualSwitchableLoRALinear):
                mod = DualSwitchableLoRALinear(mod, dg_rank=0, dg_alpha=0.0, dg_dropout=0.0)
                setattr(holder, attr, mod)
            ka = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.lora_a"
            kb = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.lora_b"
            if ka in base_lora_sd and kb in base_lora_sd:
                dt = mod.lora_a.dtype if mod.lora_a is not None else mod.base.weight.dtype
                mod.base_lora_a = base_lora_sd[ka].to(device=dev, dtype=dt).contiguous()
                mod.base_lora_b = base_lora_sd[kb].to(device=dev, dtype=dt).contiguous()
                n_attached += 2
    return n_attached


def attach_reactive_lora(
    pi05_model: Pi05Model,
    reactive_lora_sd: Mapping[str, torch.Tensor],
    *,
    vlm_scaling: float = 1.0,
    expert_scaling: float = 1.0,
) -> int:
    """Attach `P2-M12` (`#405`) `.vlm.` + `.expert.` LoRA buffers under `"reactive_policy"` (`0 MiB` extra VLM!)."""
    n_attached = 0
    dev = pi05_model.device
    for li, block in enumerate(pi05_model.paligemma_with_expert.layers):
        vlm = block.vlm
        for sub_path, holder in (
            ("q_proj", vlm),
            ("k_proj", vlm),
            ("v_proj", vlm),
            ("o_proj", vlm),
            ("mlp.gate_proj", vlm.mlp),
            ("mlp.up_proj", vlm.mlp),
            ("mlp.down_proj", vlm.mlp),
        ):
            attr = sub_path.split(".")[-1]
            mod = getattr(holder, attr)
            ka = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.lora_a"
            kb = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.lora_b"
            if ka in reactive_lora_sd and kb in reactive_lora_sd:
                if not isinstance(mod, DualSwitchableLoRALinear):
                    mod = DualSwitchableLoRALinear(mod, dg_rank=0, dg_alpha=0.0, dg_dropout=0.0)
                    setattr(holder, attr, mod)
                dt = mod.base.weight.dtype
                mod.reactive_lora_a = reactive_lora_sd[ka].to(device=dev, dtype=dt).contiguous()
                mod.reactive_lora_b = reactive_lora_sd[kb].to(device=dev, dtype=dt).contiguous()
                mod.reactive_scaling = float(vlm_scaling)
                n_attached += 2
        exp = getattr(block, "expert", None)
        if exp is not None:
            exp_holders: List[Tuple[str, Any]] = [
                ("q_proj", exp),
                ("k_proj", exp),
                ("v_proj", exp),
                ("o_proj", exp),
            ]
            if getattr(exp, "mlp", None) is not None:
                exp_holders.extend(
                    [
                        ("mlp.gate_proj", exp.mlp),
                        ("mlp.up_proj", exp.mlp),
                        ("mlp.down_proj", exp.mlp),
                    ]
                )
            for sub_path, holder in exp_holders:
                attr = sub_path.split(".")[-1]
                mod = getattr(holder, attr)
                ka = f"paligemma_with_expert.layers.{li}.expert.{sub_path}.lora_a"
                kb = f"paligemma_with_expert.layers.{li}.expert.{sub_path}.lora_b"
                if ka in reactive_lora_sd and kb in reactive_lora_sd:
                    if not isinstance(mod, DualSwitchableLoRALinear):
                        mod = DualSwitchableLoRALinear(mod, dg_rank=0, dg_alpha=0.0, dg_dropout=0.0)
                        setattr(holder, attr, mod)
                    dt = mod.base.weight.dtype
                    mod.reactive_lora_a = reactive_lora_sd[ka].to(device=dev, dtype=dt).contiguous()
                    mod.reactive_lora_b = reactive_lora_sd[kb].to(device=dev, dtype=dt).contiguous()
                    mod.reactive_scaling = float(expert_scaling)
                    n_attached += 2
    return n_attached


def attach_dg_as_base_vlm_lora(
    pi05_model: Pi05Model,
    dg_lora_sd: Mapping[str, torch.Tensor],
    *,
    dg_rank: int = 12,
    dg_alpha: float = 12.0,
) -> int:
    """Attach a second `DeltaGround-VLM` (`v2b`) `dg_lora_{a,b}` into `base_lora_{a,b}` (`"base_policy"`, `0 MiB` extra VLM, `#376`)."""
    n_attached = 0
    dev = pi05_model.device
    scale = float(dg_alpha) / float(max(1, dg_rank))
    for li, block in enumerate(pi05_model.paligemma_with_expert.layers):
        vlm = block.vlm
        for sub_path, holder in (
            ("q_proj", vlm),
            ("k_proj", vlm),
            ("v_proj", vlm),
            ("o_proj", vlm),
            ("mlp.gate_proj", vlm.mlp),
            ("mlp.up_proj", vlm.mlp),
            ("mlp.down_proj", vlm.mlp),
        ):
            attr = sub_path.split(".")[-1]
            mod = getattr(holder, attr)
            if not isinstance(mod, DualSwitchableLoRALinear):
                continue
            ka = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.dg_lora_a"
            kb = f"paligemma_with_expert.layers.{li}.vlm.{sub_path}.dg_lora_b"
            if ka in dg_lora_sd and kb in dg_lora_sd:
                dt = mod.dg_lora_a.dtype if mod.dg_lora_a is not None else mod.base.weight.dtype
                mod.base_lora_a = dg_lora_sd[ka].to(device=dev, dtype=dt).contiguous()
                mod.base_lora_b = dg_lora_sd[kb].to(device=dev, dtype=dt).contiguous()
                mod.base_scaling = scale
                n_attached += 2
            else:
                mod.base_lora_a = None
                mod.base_lora_b = None
    return n_attached


def build_patch_2d_fourier_basis(
    side: int,
    *,
    fourier_dim: int = 32,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic continuous 2D `(x, y)` patch-center Fourier basis (`align_corners=False`, `(side*side, 2+2*fourier_dim)`)."""
    centers = ((torch.arange(side, device=device, dtype=torch.float32) + 0.5) / float(side)) * 2.0 - 1.0
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    xf = xx.reshape(-1)
    yf = yy.reshape(-1)
    fx = build_1d_sincos_embed(xf * 16.0, fourier_dim, max_period=64.0)
    fy = build_1d_sincos_embed(yf * 16.0, fourier_dim, max_period=64.0)
    return torch.cat([xf.unsqueeze(-1), yf.unsqueeze(-1), fx, fy], dim=-1)


@torch.no_grad()
def select_v2_hybrid_reveal_indices(
    w_steps: torch.Tensor,
    pvis_w: torch.Tensor,
    vis_rms_w: torch.Tensor,
    boundary_reset_w: Optional[torch.Tensor],
    *,
    cutoff_excl_frame: int,
    num_keyframes: int = 12,
    uniform_anchors: int = 7,
    flush_thresh: float = 0.5,
    front_hist_w: Optional[torch.Tensor] = None,
    is_demo_w: Optional[torch.Tensor] = None,
    apex_peaks: int = 0,
) -> Tuple[torch.Tensor, int]:
    """Select `K` chronological frames (`v2b/c`: uniform + surprise; `v2d` `#391(1)`: `apex_peaks` $\\Delta_{\\mathrm{ref}}(t)$ + uniform + surprise)."""
    dev = w_steps.device
    w_tot = int(w_steps.numel())
    if w_tot == 0:
        return torch.empty((0,), device=dev, dtype=torch.long), 0
    u_cut = int(torch.searchsorted(w_steps, int(cutoff_excl_frame), right=False).item())
    if u_cut <= 0:
        return torch.empty((0,), device=dev, dtype=torch.long), 0

    start_pos = 0
    if boundary_reset_w is not None:
        resets = torch.nonzero(boundary_reset_w[:u_cut] > float(flush_thresh), as_tuple=False).flatten()
        if resets.numel() > 0:
            start_pos = int(resets[-1].item())

    n_sess = u_cut - start_pos
    if n_sess <= 0:
        return torch.empty((0,), device=dev, dtype=torch.long), start_pos
    if n_sess <= int(num_keyframes):
        return torch.arange(start_pos, u_cut, device=dev, dtype=torch.long), start_pos

    k_tot = int(num_keyframes)
    chosen: List[int] = []
    chosen_set: set[int] = set()

    # `#391(1)` `v2d`: Pick `apex_peaks` Reference-State Patch Deviation `\Delta_ref(t) = max_p ||F_t[p] - F_start[p]||_2`
    # NMS peaks FIRST so static container lift apexes (`VideoUnmask*`) & highlights (`PickHighlight`) are never suppressed!
    if int(apex_peaks) > 0 and front_hist_w is not None and int(front_hist_w.shape[0]) >= u_cut:
        f_sess = front_hist_w[start_pos:u_cut].float()
        delta_ref = torch.sqrt((f_sess - f_sess[0:1]).pow(2).mean(dim=-1) + 1e-8).max(dim=-1).values
        dref_list = delta_ref.detach().cpu().tolist()
        if is_demo_w is not None and int(is_demo_w.numel()) >= u_cut and bool(is_demo_w[start_pos:u_cut].any().item()):
            d_flags = is_demo_w[start_pos:u_cut].detach().cpu().tolist()
            apex_pool = [i for i, is_d in enumerate(d_flags) if is_d and i > 0]
            if not apex_pool:
                apex_pool = list(range(1, n_sess))
        else:
            apex_pool = list(range(1, n_sess))

        q_apex = min(int(apex_peaks), k_tot, len(apex_pool))
        if q_apex > 0:
            apex_ranked = sorted(apex_pool, key=lambda i: dref_list[i], reverse=True)
            base_sep = min(8, max(3, len(apex_pool) // max(1, q_apex + 1)))
            for sep in (base_sep, 2, 1):
                for i_rel in apex_ranked:
                    cand = start_pos + i_rel
                    if cand in chosen_set:
                        continue
                    if all(abs(cand - c) >= sep for c in chosen):
                        chosen.append(cand)
                        chosen_set.add(cand)
                        if len(chosen) >= q_apex:
                            break
                if len(chosen) >= q_apex:
                    break

        q_uni = min(max(2, int(uniform_anchors)), max(0, k_tot - len(chosen)), n_sess)
        for ai in range(q_uni):
            base_rel = int(round(ai * float(n_sess - 1) / float(max(1, q_uni - 1))))
            for off in (0, -1, 1, -2, 2):
                r_try = base_rel + off
                if 0 <= r_try < n_sess and (start_pos + r_try) not in chosen_set:
                    chosen.append(start_pos + r_try)
                    chosen_set.add(start_pos + r_try)
                    break
    else:
        q_uni = min(max(2, int(uniform_anchors)), k_tot, n_sess)
        for ai in range(q_uni):
            pos = start_pos + int(round(ai * float(n_sess - 1) / float(max(1, q_uni - 1))))
            if pos not in chosen_set:
                chosen.append(pos)
                chosen_set.add(pos)

    rem = k_tot - len(chosen)
    if rem > 0:
        scores = (pvis_w[start_pos:u_cut].float() + 0.5 * vis_rms_w[start_pos:u_cut].float()).detach().cpu().tolist()
        ranked = sorted(range(n_sess), key=lambda i: scores[i], reverse=True)
        min_sep = max(2, n_sess // (k_tot + 1))
        for i_rel in ranked:
            cand = start_pos + i_rel
            if cand in chosen_set:
                continue
            if all(abs(cand - c) >= min_sep for c in chosen):
                chosen.append(cand)
                chosen_set.add(cand)
                if len(chosen) >= k_tot:
                    break
        if len(chosen) < k_tot:
            for i_rel in ranked:
                cand = start_pos + i_rel
                if cand not in chosen_set:
                    chosen.append(cand)
                    chosen_set.add(cand)
                    if len(chosen) >= k_tot:
                        break

    return torch.tensor(sorted(chosen[:k_tot]), device=dev, dtype=torch.long), start_pos


def select_v2_pyramid_reveal_indices(
    w_steps: torch.Tensor,
    pvis_w: torch.Tensor,
    vis_rms_w: torch.Tensor,
    boundary_reset_w: Optional[torch.Tensor],
    cutoff_excl_frame: int,
    *,
    k_apex: int = 4,
    k_swap: int = 12,
    uniform_anchors: int = 8,
    flush_thresh: float = 0.5,
    front_hist_w: Optional[torch.Tensor] = None,
    is_demo_w: Optional[torch.Tensor] = None,
    prominence_tau: int = 0,
    v3_median_ref: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """`#410` Hybrid-Pyramid selector: `k_apex` (`4x256t` `16x16` $\\Delta_{\\mathrm{ref}}(t)$) + `k_swap` (`12x64t` `8x8` swap/motion).

    `prominence_tau > 0` (`#426` Bug #1 fix): rank apexes by causal two-sided transient prominence
    `max_p min(||F_t - F_{t-tau}||, ||F_t - F_{min(u_cut-1, t+tau)}||)` instead of one-sided `||F_t - F_0||`,
    so resting post-swap cups / a moved-away arm (persistent changes) are rejected and transient reveals win.
    """
    dev = w_steps.device
    w_tot = int(w_steps.numel())
    empty = torch.empty((0,), device=dev, dtype=torch.long)
    if w_tot == 0:
        return empty, empty, 0
    u_cut = int(torch.searchsorted(w_steps, int(cutoff_excl_frame), right=False).item())
    if u_cut <= 0:
        return empty, empty, 0

    start_pos = 0
    if boundary_reset_w is not None:
        resets = torch.nonzero(boundary_reset_w[:u_cut] > float(flush_thresh), as_tuple=False).flatten()
        if resets.numel() > 0:
            start_pos = int(resets[-1].item())

    n_sess = u_cut - start_pos
    if n_sess <= 0:
        return empty, empty, start_pos

    apex_chosen: List[int] = []
    apex_set: set[int] = set()
    if int(k_apex) > 0 and front_hist_w is not None and int(front_hist_w.shape[0]) >= u_cut:
        with torch.no_grad():
            if bool(v3_median_ref):
                f_sess = front_hist_w[start_pos:u_cut].float()
                f_med = f_sess.median(dim=0, keepdim=True).values
                delta_ref = torch.sqrt((f_sess - f_med).pow(2).mean(dim=-1) + 1e-8).max(dim=-1).values
            elif int(prominence_tau) > 0:
                f_sess_n = front_hist_w[start_pos:u_cut]
                tau = int(prominence_tau)
                idx_s = torch.arange(n_sess, device=f_sess_n.device)
                prev_i = (idx_s - tau).clamp(min=0)
                next_i = (idx_s + tau).clamp(max=n_sess - 1)
                d_prev = torch.sqrt((f_sess_n - f_sess_n[prev_i]).float().pow(2).mean(dim=-1) + 1e-8)
                d_next = torch.sqrt((f_sess_n - f_sess_n[next_i]).float().pow(2).mean(dim=-1) + 1e-8)
                delta_ref = torch.minimum(d_prev, d_next).max(dim=-1).values
            else:
                f_sess = front_hist_w[start_pos:u_cut].float()
                delta_ref = torch.sqrt((f_sess - f_sess[0:1]).pow(2).mean(dim=-1) + 1e-8).max(dim=-1).values
        dref_list = delta_ref.detach().cpu().tolist()
        min_i = 0 if bool(v3_median_ref) else 1
        if is_demo_w is not None and int(is_demo_w.numel()) >= u_cut and bool(is_demo_w[start_pos:u_cut].any().item()):
            d_flags = is_demo_w[start_pos:u_cut].detach().cpu().tolist()
            apex_pool = [i for i, is_d in enumerate(d_flags) if is_d and i >= min_i]
            if not apex_pool:
                apex_pool = list(range(min_i, n_sess)) if n_sess > min_i else [0]
        else:
            apex_pool = list(range(min_i, n_sess)) if n_sess > min_i else [0]

        q_apex = min(int(k_apex), len(apex_pool))
        if q_apex > 0 and bool(v3_median_ref):
            # Always anchor `start_pos` (`t=0` lifted container reveal frame!) as slot 0 before ranking median deviations!
            apex_chosen.append(start_pos)
            apex_set.add(start_pos)
        if q_apex > 0:
            apex_ranked = sorted(apex_pool, key=lambda i: dref_list[i], reverse=True)
            base_sep = min(8, max(3, len(apex_pool) // max(1, q_apex + 1)))
            for sep in (base_sep, 2, 1):
                for i_rel in apex_ranked:
                    cand = start_pos + i_rel
                    if cand in apex_set:
                        continue
                    if all(abs(cand - c) >= sep for c in apex_chosen):
                        apex_chosen.append(cand)
                        apex_set.add(cand)
                        if len(apex_chosen) >= q_apex:
                            break
                if len(apex_chosen) >= q_apex:
                    break

    swap_chosen: List[int] = []
    used_set: set[int] = set(apex_set)
    ks = max(0, int(k_swap))
    if ks > 0:
        q_uni = min(max(2, int(uniform_anchors)), ks, n_sess)
        for ai in range(q_uni):
            base_rel = int(round(ai * float(n_sess - 1) / float(max(1, q_uni - 1))))
            for off in (0, -1, 1, -2, 2):
                r_try = base_rel + off
                cand = start_pos + r_try
                if 0 <= r_try < n_sess and cand not in used_set:
                    swap_chosen.append(cand)
                    used_set.add(cand)
                    break
        if len(swap_chosen) < ks:
            scores = (pvis_w[start_pos:u_cut].float() + 0.5 * vis_rms_w[start_pos:u_cut].float()).detach().cpu().tolist()
            ranked = sorted(range(n_sess), key=lambda i: scores[i], reverse=True)
            min_sep = max(2, n_sess // max(2, ks + 1))
            for sep in (min_sep, 1):
                for i_rel in ranked:
                    cand = start_pos + i_rel
                    if cand in used_set:
                        continue
                    if all(abs(cand - c) >= sep for c in swap_chosen):
                        swap_chosen.append(cand)
                        used_set.add(cand)
                        if len(swap_chosen) >= ks:
                            break
                if len(swap_chosen) >= ks:
                    break

    apex_t = torch.tensor(sorted(apex_chosen[: max(0, int(k_apex))]), device=dev, dtype=torch.long)
    swap_t = torch.tensor(sorted(swap_chosen[:ks]), device=dev, dtype=torch.long)
    return apex_t, swap_t, start_pos


def build_local_neighbor_mask(side: int, radius: int, device: torch.device) -> torch.Tensor:
    """`(side^2, side^2)` bool mask of Chebyshev-`radius` grid neighbourhoods (raster order)."""
    c = torch.arange(int(side), device=device)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    y = yy.reshape(-1)
    x = xx.reshape(-1)
    return ((y[:, None] - y[None, :]).abs() <= int(radius)) & ((x[:, None] - x[None, :]).abs() <= int(radius))


@torch.no_grad()
def compute_patch_flow_warps(
    f_src: torch.Tensor,     # (N, P, D) frame u (8x8 pooled history features)
    f_dst: torch.Tensor,     # (N, P, D) frame u+1
    *,
    beta: float,
    id_bias: float,
    nbr_mask: torch.Tensor,  # (P, P) bool
    chunk: int = 128,
) -> torch.Tensor:
    """`#426` Bug #2: row-stochastic backward warps `W[q, p] = softmax_{p in N(q)}(beta cos(F_dst[q], F_src[p]) + b_id 1[p=q])`."""
    n, p_n, _ = f_src.shape
    out = torch.empty((n, p_n, p_n), device=f_src.device, dtype=torch.float32)
    eye = torch.eye(p_n, device=f_src.device, dtype=torch.float32)
    for c0 in range(0, n, max(1, int(chunk))):
        c1 = min(n, c0 + max(1, int(chunk)))
        fs = F.normalize(f_src[c0:c1].float(), dim=-1)
        fd = F.normalize(f_dst[c0:c1].float(), dim=-1)
        sim = torch.einsum("nqd,npd->nqp", fd, fs)
        logits = float(beta) * sim + float(id_bias) * eye
        logits = logits.masked_fill(~nbr_mask, -1e9)
        out[c0:c1] = F.softmax(logits, dim=-1)
    return out


@torch.no_grad()
def accumulate_apex_advection(
    w_flow: torch.Tensor,        # (>= u_last, P, P): w_flow[u] warps frame u -> u+1
    apex_pos: Sequence[int],     # absolute write positions of the selected apex frames
    u_last: int,                 # last written position (< u_cut)
    w_cur: torch.Tensor,         # (P, P) warp from frame `u_last` -> current front
    k_total: int,
) -> torch.Tensor:
    """Compose causal warps `W_cur @ W_{u_last-1} @ ... @ W_{a_k}` per apex -> `(k_total, P, P)` (identity padding)."""
    p_n = int(w_cur.shape[0])
    dev = w_cur.device
    eye = torch.eye(p_n, device=dev, dtype=torch.float32)
    out = eye.unsqueeze(0).repeat(int(k_total), 1, 1)
    n_a = min(len(apex_pos), int(k_total))
    if n_a == 0:
        return out
    a_t = torch.tensor([int(a) for a in apex_pos[:n_a]], device=dev, dtype=torch.long)
    acc = eye.unsqueeze(0).repeat(n_a, 1, 1)
    for u in range(int(a_t.min().item()), int(u_last)):
        act = (a_t <= u).view(n_a, 1, 1)
        acc = torch.where(act, torch.matmul(w_flow[u].unsqueeze(0), acc), acc)
    out[:n_a] = torch.matmul(w_cur.float().unsqueeze(0), acc)
    return out


class RevealCrossAttnBlock(nn.Module):
    """Translation-equivariant `front (16x16) <- reveal (K x 8x8)` cross-attention with 2D spatial proximity bias (`#367`/`#368`)."""

    def __init__(self, d: int, num_heads: int = 8, dropout: float = 0.15) -> None:
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.head_dim = self.d // self.num_heads
        self.q_norm = nn.RMSNorm(d)
        self.kv_norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o_proj.weight)
        # Per-head Gaussian spatial proximity slope `softplus(log_prox)` (`0` seat bias; pure relative `||(x,y)-(x',y')||^2`!)
        self.log_prox = nn.Parameter(torch.linspace(-1.0, 1.2, steps=self.num_heads))
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)

    def forward(
        self,
        q_in: torch.Tensor,            # (B, P_f, d)
        kv_in: torch.Tensor,           # (B, K*P_r, d)
        kv_valid_mask: torch.Tensor,   # (B, K*P_r) bool
        dist2_fr: torch.Tensor,        # (1, 1, P_f, K*P_r) fp32 squared 2D distance in [-1,1]^2
    ) -> torch.Tensor:
        bsz, pf, _ = q_in.shape
        kr = int(kv_in.shape[1])
        dt = self.q_proj.weight.dtype
        qh = self.q_proj(self.q_norm(q_in.to(dt))).view(bsz, pf, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(self.kv_norm(kv_in.to(dt))).view(bsz, kr, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(self.kv_norm(kv_in.to(dt))).view(bsz, kr, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(float(self.head_dim))
        logits = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * scale
        prox_w = F.softplus(self.log_prox.float()).view(1, self.num_heads, 1, 1)
        logits = logits - prox_w * dist2_fr
        has_any = kv_valid_mask.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~kv_valid_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(logits, dim=-1).to(dt)
        attn = torch.where(has_any.unsqueeze(1).unsqueeze(-1), attn, torch.zeros_like(attn))
        ctx = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(bsz, pf, self.d)
        x = q_in + self.drop(self.o_proj(ctx))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


class VLMGroundedPointerCoda(nn.Module):
    """Reads `PaliGemma-2B` `x0_out` (`16x16` front + pooled lang/GDN + `v2` `reveal_vlm`) -> `<row_y, col_x>` & `133` templates."""

    def __init__(self, cfg: DeltaGroundVLMConfig) -> None:
        super().__init__()
        d_vlm = int(cfg.vlm_width)
        d = int(cfg.d_head)
        self.d_head = d
        self.v2_mode = bool(cfg.v2_mode)
        self.front_n = int(cfg.front_tokens_curr)
        self.front_side = int(round(self.front_n**0.5))
        self.rev_k = int(cfg.reveal_keyframes)
        self.rev_p = int(cfg.reveal_tokens_per_frame)
        self.rev_side = int(round(self.rev_p**0.5))
        self.pyr_apex_k = max(0, int(cfg.v2_pyramid_apex_hr16))
        self.fourier_dim = 32

        self.front_in = nn.Sequential(nn.RMSNorm(d_vlm), nn.Linear(d_vlm, d))
        self.ctx_film = nn.Linear(2 * d_vlm, 2 * d)
        nn.init.zeros_(self.ctx_film.weight)
        nn.init.zeros_(self.ctx_film.bias)

        if self.v2_mode:
            self.front_drop = nn.Dropout(cfg.v2_front_drop)
            self.rev_in = nn.Sequential(nn.RMSNorm(d_vlm), nn.Linear(d_vlm, d))
            if cfg.v2_raw_siglip_reveal:
                self.raw_rev_norm = nn.RMSNorm(d_vlm)
                self.raw_rev_proj = nn.Linear(d_vlm, d, bias=False)
                nn.init.zeros_(self.raw_rev_proj.weight)
            else:
                self.raw_rev_norm = None
                self.raw_rev_proj = None
            self.rev_film = nn.Linear(2 * d_vlm, 2 * d)
            nn.init.zeros_(self.rev_film.weight)
            nn.init.zeros_(self.rev_film.bias)
            self.local_time_q = nn.Sequential(nn.Linear(2 * d_vlm, d), nn.SiLU(), nn.Linear(d, d, bias=False))
            self.local_rev_out = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.local_rev_out.weight)
            self.cross_blocks = nn.ModuleList(
                [RevealCrossAttnBlock(d, num_heads=cfg.v2_coda_heads, dropout=cfg.v2_front_drop) for _ in range(2)]
            )

        # `#426` Wave-16: advected (post-swap registered) apex reveal fusion (zero-init -> step-0 == v4 coda!)
        self.adv_on = bool(self.v2_mode and cfg.v2_advect_reveal and self.pyr_apex_k > 0)
        if self.adv_on:
            self.adv_time_q = nn.Sequential(nn.Linear(2 * d_vlm, d), nn.SiLU(), nn.Linear(d, d, bias=False))
            self.adv_norm = nn.RMSNorm(d)
            self.adv_out = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.adv_out.weight)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=1),
        )
        self.grid_k_proj = nn.Linear(d, d, bias=False)
        self.pri_q_proj = nn.Sequential(nn.Linear(2 * d_vlm, d), nn.SiLU(), nn.Linear(d, d, bias=False))
        self.sec_q_proj = nn.Sequential(nn.Linear(2 * d_vlm, d), nn.SiLU(), nn.Linear(d, d, bias=False))
        self.offset_head = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 4))
        nn.init.zeros_(self.offset_head[-1].weight)  # type: ignore[union-attr]
        nn.init.zeros_(self.offset_head[-1].bias)    # type: ignore[union-attr]

        trunk_in_dim = (3 * d_vlm + 2 * d) if self.v2_mode else (2 * d_vlm + 2 * d + 4 + 4 * self.fourier_dim)
        self.readout_trunk = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(trunk_in_dim, d * 2),
            nn.RMSNorm(d * 2),
            SwiGLU(d * 2, d * 4),
            nn.Dropout(cfg.head_dropout),
            nn.RMSNorm(d * 2),
        )
        self.template_head = nn.Linear(d * 2, cfg.num_templates)
        self.completed_count_head = nn.Linear(d * 2, cfg.max_count_classes)
        self.remaining_count_head = nn.Linear(d * 2, cfg.max_count_classes)
        self.ordinal_head = nn.Linear(d * 2, cfg.max_count_classes)
        self.stage_head = nn.Linear(d * 2, cfg.num_stage_classes)
        self.color_head = nn.Linear(d * 2, cfg.num_color_classes)

    def forward(
        self,
        front_vlm: torch.Tensor,   # (B, P_f, D_vlm)
        lang_pool: torch.Tensor,   # (B, D_vlm)
        gdn_pool: torch.Tensor,    # (B, D_vlm)
        *,
        reveal_vlm: Optional[torch.Tensor] = None,   # Optional (B, K*P_r, D_vlm) for `v2_mode`
        reveal_raw: Optional[torch.Tensor] = None,   # Optional (B, K*P_r, D_vlm) unblurred SigLIP for `v2c`
        reveal_mask: Optional[torch.Tensor] = None,  # Optional (B, K*P_r) bool
        reveal_advect: Optional[torch.Tensor] = None,  # Optional (B, K_apex, P_c, P_c) causal apex->current warps (`#426`)
    ) -> Dict[str, torch.Tensor]:
        bsz = int(front_vlm.shape[0])
        s_side = self.front_side
        dt = self.grid_k_proj.weight.dtype
        ctx = torch.cat([lang_pool.to(dt), gdn_pool.to(dt)], dim=-1)

        g0 = self.front_in(front_vlm.to(dt))
        fg, fb = self.ctx_film(ctx).chunk(2, dim=-1)
        g_mod = g0 * (1.0 + 0.30 * torch.tanh(fg).unsqueeze(1)) + 0.30 * fb.unsqueeze(1)

        rev_pool: Optional[torch.Tensor] = None
        if self.v2_mode and reveal_vlm is not None and reveal_mask is not None:
            dev = front_vlm.device
            rm_f = reveal_mask.to(dt).unsqueeze(-1)
            rev_pool = (reveal_vlm.to(dt) * rm_f).sum(dim=1) / rm_f.sum(dim=1).clamp(min=1.0)

            g_mod = self.front_drop(g_mod)
            r0 = self.rev_in(reveal_vlm.to(dt))
            if self.raw_rev_proj is not None and self.raw_rev_norm is not None and reveal_raw is not None:
                r0 = r0 + self.raw_rev_proj(self.raw_rev_norm(reveal_raw.to(dt)))
            rg, rb = self.rev_film(ctx).chunk(2, dim=-1)
            r_mod = r0 * (1.0 + 0.30 * torch.tanh(rg).unsqueeze(1)) + 0.30 * rb.unsqueeze(1)

            # 1) Exact `align_corners=False` co-located temporal reveal alignment (`1:1` direct on `16x16` apex, `0` blur!)
            p_f = s_side * s_side
            if self.pyr_apex_k > 0:
                n_apex_tok = self.pyr_apex_k * p_f
                k_swap = max(0, self.rev_k - self.pyr_apex_k)
                r_apex = r_mod[:, :n_apex_tok, :].reshape(bsz, self.pyr_apex_k, p_f, self.d_head)
                f_valid_a = reveal_mask[:, :n_apex_tok].reshape(bsz, self.pyr_apex_k, p_f).any(dim=-1)
                if k_swap > 0:
                    r_sw_raw = r_mod[:, n_apex_tok:, :]
                    if self.rev_side == s_side:
                        r_sw_up = r_sw_raw.reshape(bsz, k_swap, p_f, self.d_head)
                    else:
                        r_sw_4d = (
                            r_sw_raw.reshape(bsz * k_swap, self.rev_side, self.rev_side, self.d_head)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                        r_sw_up = (
                            F.interpolate(r_sw_4d.float(), size=(s_side, s_side), mode="bilinear", align_corners=False)
                            .to(dt)
                            .view(bsz, k_swap, self.d_head, p_f)
                            .permute(0, 1, 3, 2)
                            .contiguous()
                        )
                    r_up_seq = torch.cat([r_apex, r_sw_up], dim=1)
                    f_valid_s = reveal_mask[:, n_apex_tok:].reshape(bsz, k_swap, self.rev_p).any(dim=-1)
                    f_valid = torch.cat([f_valid_a, f_valid_s], dim=1)
                else:
                    r_up_seq = r_apex
                    f_valid = f_valid_a
            elif self.rev_side == s_side:
                r_up_seq = r_mod.view(bsz, self.rev_k, p_f, self.d_head)
                f_valid = reveal_mask.view(bsz, self.rev_k, self.rev_p).any(dim=-1)
            else:
                r_4d = (
                    r_mod.view(bsz * self.rev_k, self.rev_side, self.rev_side, self.d_head)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                r_up = F.interpolate(r_4d.float(), size=(s_side, s_side), mode="bilinear", align_corners=False).to(dt)
                r_up_seq = (
                    r_up.view(bsz, self.rev_k, self.d_head, p_f)
                    .permute(0, 1, 3, 2)
                    .contiguous()
                )  # (B, K, P_f, d)
                f_valid = reveal_mask.view(bsz, self.rev_k, self.rev_p).any(dim=-1)  # (B, K)
            tq = self.local_time_q(ctx)[:, None, None, :]  # (B, 1, 1, d)
            t_logits = (r_up_seq.float() * tq.float()).sum(dim=-1) / math.sqrt(float(self.d_head))  # (B, K, P_f)
            has_any_f = f_valid.any(dim=-1, keepdim=True).unsqueeze(-1)
            t_logits = t_logits.masked_fill(~f_valid.unsqueeze(-1), -1e9)
            t_attn = F.softmax(t_logits, dim=1).to(dt)
            t_attn = torch.where(has_any_f, t_attn, torch.zeros_like(t_attn))
            r_local = (t_attn.unsqueeze(-1) * r_up_seq).sum(dim=1)  # (B, P_f, d)
            g_mod = g_mod + self.local_rev_out(r_local)

            # 1b) `#426` Bug #2: advect each 16x16 apex reveal map along causal 8x8 patch flow onto the CURRENT front
            if self.adv_on and reveal_advect is not None:
                ka = self.pyr_apex_k
                c_side = s_side // 2
                r_ap = r_mod[:, : ka * p_f, :].reshape(bsz, ka, c_side, 2, c_side, 2, self.d_head)
                r_ap = r_ap.permute(0, 1, 2, 4, 3, 5, 6).reshape(bsz, ka, c_side * c_side, 4, self.d_head)
                adv = torch.einsum("bkqp,bkpsd->bkqsd", reveal_advect.to(device=dev, dtype=dt), r_ap)
                adv16 = (
                    adv.reshape(bsz, ka, c_side, c_side, 2, 2, self.d_head)
                    .permute(0, 1, 2, 4, 3, 5, 6)
                    .reshape(bsz, ka, p_f, self.d_head)
                )
                f_valid_adv = reveal_mask[:, : ka * p_f].reshape(bsz, ka, p_f).any(dim=-1)
                aq = self.adv_time_q(ctx)[:, None, None, :]
                a_logits = (adv16.float() * aq.float()).sum(dim=-1) / math.sqrt(float(self.d_head))
                a_logits = a_logits.masked_fill(~f_valid_adv.unsqueeze(-1), -1e9)
                a_attn = F.softmax(a_logits, dim=1).to(dt)
                has_adv = f_valid_adv.any(dim=-1, keepdim=True).unsqueeze(-1)
                a_attn = torch.where(has_adv, a_attn, torch.zeros_like(a_attn))
                r_adv = (a_attn.unsqueeze(-1) * adv16).sum(dim=1)
                g_mod = g_mod + self.adv_out(self.adv_norm(r_adv))

            # 2) Translation-equivariant `(16x16) <- reveal` cross-attention for swapping containers (`dist2_fr` only!)
            c_f1 = ((torch.arange(s_side, device=dev, dtype=torch.float32) + 0.5) / float(s_side)) * 2.0 - 1.0
            yy_f, xx_f = torch.meshgrid(c_f1, c_f1, indexing="ij")
            xy_f_flat = torch.stack([xx_f.reshape(-1), yy_f.reshape(-1)], dim=-1)
            xy_f = xy_f_flat.unsqueeze(1)  # (P_f, 1, 2)
            c_r1 = ((torch.arange(self.rev_side, device=dev, dtype=torch.float32) + 0.5) / float(self.rev_side)) * 2.0 - 1.0
            yy_r, xx_r = torch.meshgrid(c_r1, c_r1, indexing="ij")
            xy_s_flat = torch.stack([xx_r.reshape(-1), yy_r.reshape(-1)], dim=-1)
            if self.pyr_apex_k > 0:
                k_swap_xy = max(0, self.rev_k - self.pyr_apex_k)
                xy_parts = [xy_f_flat.repeat(self.pyr_apex_k, 1)]
                if k_swap_xy > 0:
                    xy_parts.append(xy_s_flat.repeat(k_swap_xy, 1))
                xy_r = torch.cat(xy_parts, dim=0).unsqueeze(0)
            else:
                xy_r = xy_s_flat.repeat(self.rev_k, 1).unsqueeze(0)
            dist2_fr = ((xy_f - xy_r) ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)
            for c_blk in self.cross_blocks:
                g_mod = c_blk(g_mod, r_mod, reveal_mask, dist2_fr)

        g_2d = g_mod.permute(0, 2, 1).contiguous().view(bsz, self.d_head, s_side, s_side)
        g_ref = (g_2d + self.spatial_conv(g_2d)).view(bsz, self.d_head, s_side * s_side).permute(0, 2, 1).contiguous()

        k_grid = self.grid_k_proj(g_ref)
        scale = 1.0 / math.sqrt(float(self.d_head))
        q_pri = self.pri_q_proj(ctx).unsqueeze(1)
        q_sec = self.sec_q_proj(ctx).unsqueeze(1)
        hm_pri = torch.matmul(q_pri, k_grid.transpose(-2, -1)).squeeze(1) * scale
        hm_sec = torch.matmul(q_sec, k_grid.transpose(-2, -1)).squeeze(1) * scale
        hm_logits = torch.stack([hm_pri, hm_sec], dim=1).float()

        cell_offsets = torch.tanh(self.offset_head(g_ref).float()) * (1.0 / float(s_side))
        with torch.autocast(device_type=front_vlm.device.type, enabled=False):
            xy_pri_base, probs_pri = spatial_softmax_2d(hm_logits[:, 0, :], grid_hw=(s_side, s_side), align_corners=False)
            xy_sec_base, probs_sec = spatial_softmax_2d(hm_logits[:, 1, :], grid_hw=(s_side, s_side), align_corners=False)
            off_pri = (probs_pri.unsqueeze(-1) * cell_offsets[:, :, 0:2]).sum(dim=1)
            off_sec = (probs_sec.unsqueeze(-1) * cell_offsets[:, :, 2:4]).sum(dim=1)
            xy_pri = (xy_pri_base + off_pri).clamp(-1.0, 1.0)
            xy_sec = (xy_sec_base + off_sec).clamp(-1.0, 1.0)

        g_map_f32 = g_ref.float().permute(0, 2, 1).contiguous().view(bsz, self.d_head, s_side, s_side)
        sample_grid = torch.stack([xy_pri, xy_sec], dim=1).unsqueeze(2)
        sampled = F.grid_sample(g_map_f32, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        crop_pri = sampled[:, :, 0, 0].to(dt)
        crop_sec = sampled[:, :, 1, 0].to(dt)

        if self.v2_mode:
            if rev_pool is None:
                rev_pool = torch.zeros_like(lang_pool.to(dt))
            trunk_in = torch.cat([ctx, rev_pool, crop_pri, crop_sec], dim=-1)
        else:
            if rev_pool is None:
                rev_pool = torch.zeros_like(lang_pool.to(dt))
            fx1 = build_1d_sincos_embed(xy_pri[:, 0] * 16.0, self.fourier_dim, max_period=64.0)
            fy1 = build_1d_sincos_embed(xy_pri[:, 1] * 16.0, self.fourier_dim, max_period=64.0)
            fx2 = build_1d_sincos_embed(xy_sec[:, 0] * 16.0, self.fourier_dim, max_period=64.0)
            fy2 = build_1d_sincos_embed(xy_sec[:, 1] * 16.0, self.fourier_dim, max_period=64.0)
            trunk_in = torch.cat([ctx, crop_pri, crop_sec, xy_pri.to(dt), xy_sec.to(dt), fx1.to(dt), fy1.to(dt), fx2.to(dt), fy2.to(dt)], dim=-1)
        h_out = self.readout_trunk(trunk_in)
        return {
            "template_logits": self.template_head(h_out).float(),
            "xy_pri": xy_pri,
            "xy_sec": xy_sec,
            "hm_logits": hm_logits,
            "completed_count_logits": self.completed_count_head(h_out).float(),
            "remaining_count_logits": self.remaining_count_head(h_out).float(),
            "ordinal_logits": self.ordinal_head(h_out).float(),
            "stage_logits": self.stage_head(h_out).float(),
            "color_logits": self.color_head(h_out).float(),
            "rev_pool": rev_pool,
        }


class DeltaGroundVLMHeadModule(nn.Module):
    """Trainable memory + pointer sidecar for `DeltaGround-VLM` (`< 28M` params + `dg_vlm_lora` on `Pi05Model`)."""

    def __init__(self, cfg: DeltaGroundVLMConfig, template_vocab: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        if template_vocab is not None:
            cfg.num_templates = max(2, len(template_vocab))
        self.cfg = cfg
        self.template_vocab: List[str] = (
            list(template_vocab) if template_vocab is not None else ["" for _ in range(cfg.num_templates)]
        )
        self.mem_cfg = cfg.to_pi05_mem_cfg()
        self.writer = DualViewEventPerceiverWriter(self.mem_cfg)
        self.gdn = MultiTimescale2LayerGDN(self.mem_cfg)
        self.reveal_selector = VLMPrefixSurpriseSelector(self.mem_cfg)

        self.wrist_view_emb = nn.Parameter(torch.zeros(1, 1, cfg.vlm_width))
        self.ctx_q_proj = nn.Linear(cfg.vlm_width, cfg.d_mem)
        self.gdn_to_vlm = nn.Sequential(nn.RMSNorm(cfg.d_mem), nn.Linear(cfg.d_mem, cfg.vlm_width))
        state_in_dim = (cfg.state_dim * 2 + 2 + 4) if cfg.v2_mode else (cfg.state_dim * 2 + 2)
        self.state_to_vlm = nn.Sequential(
            nn.Linear(state_in_dim, cfg.d_mem),
            nn.SiLU(),
            nn.Linear(cfg.d_mem, 2 * cfg.vlm_width),
        )
        if cfg.v2_mode:
            self.rev_delta_proj = nn.Linear(cfg.vis_dim, cfg.vlm_width, bias=False)
            nn.init.zeros_(self.rev_delta_proj.weight)
            self.rev_time_proj = nn.Linear(18, cfg.vlm_width)
            nn.init.zeros_(self.rev_time_proj.weight)
            nn.init.zeros_(self.rev_time_proj.bias)
            if cfg.v2_hr16_sincos:
                f_side = int(round(cfg.front_tokens_curr**0.5))
                c_g = ((torch.arange(f_side, dtype=torch.float32) + 0.5) / float(f_side)) * 2.0 - 1.0
                yy_g, xx_g = torch.meshgrid(c_g, c_g, indexing="ij")
                xy_g = torch.stack([xx_g.reshape(-1), yy_g.reshape(-1)], dim=-1)
                pos_g = torch.cat(
                    [
                        xy_g,
                        build_1d_sincos_embed(xy_g[:, 0] * 4.0, 8, max_period=16.0),
                        build_1d_sincos_embed(xy_g[:, 1] * 4.0, 8, max_period=16.0),
                    ],
                    dim=-1,
                )  # (P_f, 18)
                self.register_buffer("hr16_pos_basis", pos_g, persistent=False)
                self.hr16_pos_proj: Optional[nn.Linear] = nn.Linear(18, cfg.vlm_width, bias=False)
                nn.init.zeros_(self.hr16_pos_proj.weight)
            else:
                self.hr16_pos_proj = None
        else:
            self.hr16_pos_proj = None
        self.coda = VLMGroundedPointerCoda(cfg)

    def cast_bf16_keep_fp32_gates(self) -> None:
        self.to(torch.bfloat16)
        for name, p in self.named_parameters():
            if any(
                k in name
                for k in ("A_log", "dt_bias", "salience_gate", "wrist_view_emb", "vlm_hist_slot_emb", "log_prox")
            ):
                p.data = p.data.float()
        for _, b in self.named_buffers():
            if b.is_floating_point():
                b.data = b.data.float()

    def regularize_v2_vis_tokens(self, x: torch.Tensor, *, allow_patch_drop: bool = True) -> torch.Tensor:
        """Grid-preserving (`#368`-safe!) feature noise + optional patch dropout during `v2` training."""
        if not (self.cfg.v2_mode and self.training) or int(x.shape[0]) == 0:
            return x
        out = x
        if self.cfg.v2_feat_noise_std > 0.0:
            scale = float(self.cfg.v2_feat_noise_std) if allow_patch_drop else (0.5 * float(self.cfg.v2_feat_noise_std))
            rms = torch.sqrt(out.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(out.dtype)
            out = out + torch.randn_like(out) * (scale * rms)
        if allow_patch_drop and self.cfg.v2_patch_drop_prob > 0.0:
            keep_p = 1.0 - float(self.cfg.v2_patch_drop_prob)
            mask = (torch.rand((int(out.shape[0]), int(out.shape[1]), 1), device=out.device) < keep_p).to(out.dtype)
            out = out * (mask / max(1e-4, keep_p))
        return out

    def encode_v2_front_coords(self, f_cur: torch.Tensor) -> torch.Tensor:
        if self.hr16_pos_proj is not None and int(f_cur.shape[1]) == int(self.hr16_pos_basis.shape[0]):
            pos_dt = self.hr16_pos_proj.weight.dtype
            p_emb = self.hr16_pos_proj(self.hr16_pos_basis.to(device=f_cur.device, dtype=pos_dt)).to(f_cur.dtype)
            return f_cur + p_emb.unsqueeze(0)
        return f_cur

    def _upsample_vis_grid(self, x_seq: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if int(x_seq.shape[0]) == 0 or int(x_seq.shape[1]) == int(target_tokens):
            return x_seq
        s_in = int(round(int(x_seq.shape[1]) ** 0.5))
        s_out = int(round(int(target_tokens) ** 0.5))
        x_4d = x_seq.view(int(x_seq.shape[0]), s_in, s_in, -1).permute(0, 3, 1, 2).contiguous()
        y_4d = F.interpolate(x_4d.float(), size=(s_out, s_out), mode="bilinear", align_corners=False).to(x_seq.dtype)
        return y_4d.permute(0, 2, 3, 1).contiguous().view(int(x_seq.shape[0]), int(target_tokens), -1)

    def encode_v2_reveal_tokens(
        self,
        f_w_prefix: torch.Tensor,       # (U, P_hist, vis_dim)
        w_steps_prefix: torch.Tensor,   # (U,) long
        d_w_prefix: torch.Tensor,       # (U,) bool
        v_sel: torch.Tensor,            # (M_sel,) long in [start_pos, U)
        start_pos: int,
        *,
        f_sel_hr: Optional[torch.Tensor] = None,
        f_prev_hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode `K` (`8x8` or unpooled `16x16` `#395(1)`) reveal frames with optical motion `df` & relative phase `rel_t`."""
        n_sel = int(v_sel.numel())
        u_cut = int(w_steps_prefix.numel())
        p_r = int(self.cfg.reveal_tokens_per_frame)
        if f_sel_hr is not None and n_sel > 0:
            f_sel = f_sel_hr
        else:
            f_sel_raw = f_w_prefix[v_sel] if n_sel > 0 else f_w_prefix[:0]
            f_sel = self._upsample_vis_grid(f_sel_raw, p_r) if int(f_sel_raw.shape[1]) != p_r else f_sel_raw
        base_tok, rev_mask = self.reveal_selector.encode_vlm_surprise_tokens(f_sel)
        if not self.cfg.v2_mode or n_sel == 0 or u_cut == 0:
            return base_tok * rev_mask.unsqueeze(-1).to(base_tok.dtype), rev_mask

        dev = base_tok.device
        out_dt = base_tok.dtype
        lin_dt = self.rev_delta_proj.weight.dtype
        k_tot = int(self.cfg.reveal_keyframes)

        prev_sel = torch.clamp(v_sel - 1, min=int(start_pos))
        if f_sel_hr is not None and f_prev_hr is not None:
            df_sel = f_sel_hr - f_prev_hr
        else:
            df_raw = f_w_prefix[v_sel] - f_w_prefix[prev_sel]
            df_sel = self._upsample_vis_grid(df_raw, p_r) if int(df_raw.shape[1]) != p_r else df_raw
        df_pad = torch.zeros((k_tot, p_r, int(f_w_prefix.shape[-1])), device=dev, dtype=lin_dt)
        df_pad[:n_sel] = df_sel[:k_tot].to(lin_dt)
        delta_emb = self.rev_delta_proj(df_pad.view(k_tot * p_r, -1)).to(out_dt)

        s0 = float(w_steps_prefix[int(start_pos)].item())
        s_last = float(max(int(w_steps_prefix[int(start_pos)].item()) + 1, int(w_steps_prefix[u_cut - 1].item())))
        rel_t = ((w_steps_prefix[v_sel[:k_tot]].float() - s0) / max(1.0, s_last - s0)).clamp(0.0, 1.0)
        d_flt = d_w_prefix[v_sel[:k_tot]].float()
        t_sin = build_1d_sincos_embed(rel_t * 16.0, 16, max_period=64.0)
        t_feat = torch.cat([rel_t.unsqueeze(-1), d_flt.unsqueeze(-1), t_sin], dim=-1)  # (n_sel, 18)
        t_pad = torch.zeros((k_tot, 18), device=dev, dtype=lin_dt)
        t_pad[: int(t_feat.shape[0])] = t_feat.to(lin_dt)
        time_emb = self.rev_time_proj(t_pad).unsqueeze(1).expand(k_tot, p_r, -1).reshape(k_tot * p_r, -1).to(out_dt)

        tot_tok = base_tok + delta_emb + time_emb
        if self.hr16_pos_proj is not None and p_r == int(self.hr16_pos_basis.shape[0]):
            pos_dt = self.hr16_pos_proj.weight.dtype
            p_emb = self.hr16_pos_proj(self.hr16_pos_basis.to(device=dev, dtype=pos_dt)).to(out_dt)
            tot_tok = tot_tok + p_emb.repeat(k_tot, 1)
        tot_tok = tot_tok * rev_mask.unsqueeze(-1).to(out_dt)
        return tot_tok, rev_mask

    def encode_v2_pyramid_reveal_tokens(
        self,
        f_w_prefix: torch.Tensor,
        w_steps_prefix: torch.Tensor,
        d_w_prefix: torch.Tensor,
        apex_sel: torch.Tensor,
        swap_sel: torch.Tensor,
        start_pos: int,
        *,
        f_apex_hr: Optional[torch.Tensor] = None,
        f_prev_apex_hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`#410` Hybrid-Pyramid reveal encoder: `k_apex x P_f` (`4x256t`) + `k_swap x P_r` (`12x64t`) = `1,792` tokens."""
        dev = f_w_prefix.device
        out_dt = f_w_prefix.dtype
        v_dim = int(self.cfg.vlm_width)
        p_f = int(self.cfg.front_tokens_curr)
        p_s = int(self.cfg.reveal_tokens_per_frame)
        k_apex = max(0, int(self.cfg.v2_pyramid_apex_hr16))
        k_swap = max(0, int(self.cfg.reveal_keyframes) - k_apex)
        tot_len = k_apex * p_f + k_swap * p_s
        u_cut = int(w_steps_prefix.numel())
        n_a = min(int(apex_sel.numel()), k_apex)
        n_s = min(int(swap_sel.numel()), k_swap)
        if not self.cfg.v2_mode or u_cut == 0 or (n_a == 0 and n_s == 0):
            return (
                torch.zeros((tot_len, v_dim), device=dev, dtype=out_dt),
                torch.zeros((tot_len,), device=dev, dtype=torch.bool),
            )

        lin_dt = self.rev_delta_proj.weight.dtype
        slot_all = self.reveal_selector.vlm_hist_slot_emb.to(device=dev, dtype=out_dt)
        s0 = float(w_steps_prefix[int(start_pos)].item())
        s_last = float(max(int(w_steps_prefix[int(start_pos)].item()) + 1, int(w_steps_prefix[u_cut - 1].item())))

        p_emb_16: Optional[torch.Tensor] = None
        p_emb_s: Optional[torch.Tensor] = None
        if self.hr16_pos_proj is not None and p_f == int(self.hr16_pos_basis.shape[0]):
            pos_dt = self.hr16_pos_proj.weight.dtype
            p_emb_16 = self.hr16_pos_proj(self.hr16_pos_basis.to(device=dev, dtype=pos_dt)).to(out_dt)
            p_emb_s = (
                p_emb_16
                if p_s == p_f
                else pool_spatial_tokens_to_square(p_emb_16.unsqueeze(0), p_s).squeeze(0)
            )

        # 1) High-res `k_apex x 256` (`16x16`) apex reveal block
        apex_tok = torch.zeros((k_apex * p_f, v_dim), device=dev, dtype=out_dt)
        apex_mask = torch.zeros((k_apex, p_f), device=dev, dtype=torch.bool)
        if k_apex > 0 and n_a > 0:
            asel = apex_sel[:n_a]
            psel_a = torch.clamp(asel - 1, min=int(start_pos))
            if f_apex_hr is not None and int(f_apex_hr.shape[0]) >= n_a:
                fa = f_apex_hr[:n_a]
            else:
                fa_raw = f_w_prefix[asel]
                fa = self._upsample_vis_grid(fa_raw, p_f) if int(fa_raw.shape[1]) != p_f else fa_raw
            if f_apex_hr is not None and f_prev_apex_hr is not None and int(f_prev_apex_hr.shape[0]) >= n_a:
                dfa = f_apex_hr[:n_a] - f_prev_apex_hr[:n_a]
            else:
                dfa_raw = f_w_prefix[asel] - f_w_prefix[psel_a]
                dfa = self._upsample_vis_grid(dfa_raw, p_f) if int(dfa_raw.shape[1]) != p_f else dfa_raw

            a_base = torch.zeros((k_apex, p_f, v_dim), device=dev, dtype=out_dt)
            a_base[:n_a] = fa.to(out_dt) + slot_all[:n_a]
            apex_mask[:n_a] = True

            dfa_pad = torch.zeros((k_apex, p_f, int(f_w_prefix.shape[-1])), device=dev, dtype=lin_dt)
            dfa_pad[:n_a] = dfa.to(lin_dt)
            a_delta = self.rev_delta_proj(dfa_pad.view(k_apex * p_f, -1)).to(out_dt)

            rt_a = ((w_steps_prefix[asel].float() - s0) / max(1.0, s_last - s0)).clamp(0.0, 1.0)
            df_a = d_w_prefix[asel].float()
            ts_a = build_1d_sincos_embed(rt_a * 16.0, 16, max_period=64.0)
            tf_a = torch.cat([rt_a.unsqueeze(-1), df_a.unsqueeze(-1), ts_a], dim=-1)
            tp_a = torch.zeros((k_apex, 18), device=dev, dtype=lin_dt)
            tp_a[:n_a] = tf_a.to(lin_dt)
            a_time = self.rev_time_proj(tp_a).unsqueeze(1).expand(k_apex, p_f, -1).reshape(k_apex * p_f, -1).to(out_dt)

            apex_tok = a_base.view(k_apex * p_f, v_dim) + a_delta + a_time
            if p_emb_16 is not None:
                apex_tok = apex_tok + p_emb_16.repeat(k_apex, 1)
            apex_tok = apex_tok * apex_mask.view(-1, 1).to(out_dt)

        # 2) Chronological `k_swap x 64` (`8x8`) swap & motion block
        swap_tok = torch.zeros((k_swap * p_s, v_dim), device=dev, dtype=out_dt)
        swap_mask = torch.zeros((k_swap, p_s), device=dev, dtype=torch.bool)
        if k_swap > 0 and n_s > 0:
            ssel = swap_sel[:n_s]
            psel_s = torch.clamp(ssel - 1, min=int(start_pos))
            fs_raw = f_w_prefix[ssel]
            fs = pool_spatial_tokens_to_square(fs_raw, p_s) if int(fs_raw.shape[1]) != p_s else fs_raw
            dfs_raw = f_w_prefix[ssel] - f_w_prefix[psel_s]
            dfs = pool_spatial_tokens_to_square(dfs_raw, p_s) if int(dfs_raw.shape[1]) != p_s else dfs_raw

            s_base = torch.zeros((k_swap, p_s, v_dim), device=dev, dtype=out_dt)
            s_base[:n_s] = fs.to(out_dt) + slot_all[k_apex : k_apex + n_s]
            swap_mask[:n_s] = True

            dfs_pad = torch.zeros((k_swap, p_s, int(f_w_prefix.shape[-1])), device=dev, dtype=lin_dt)
            dfs_pad[:n_s] = dfs.to(lin_dt)
            s_delta = self.rev_delta_proj(dfs_pad.view(k_swap * p_s, -1)).to(out_dt)

            rt_s = ((w_steps_prefix[ssel].float() - s0) / max(1.0, s_last - s0)).clamp(0.0, 1.0)
            df_s = d_w_prefix[ssel].float()
            ts_s = build_1d_sincos_embed(rt_s * 16.0, 16, max_period=64.0)
            tf_s = torch.cat([rt_s.unsqueeze(-1), df_s.unsqueeze(-1), ts_s], dim=-1)
            tp_s = torch.zeros((k_swap, 18), device=dev, dtype=lin_dt)
            tp_s[:n_s] = tf_s.to(lin_dt)
            s_time = self.rev_time_proj(tp_s).unsqueeze(1).expand(k_swap, p_s, -1).reshape(k_swap * p_s, -1).to(out_dt)

            swap_tok = s_base.view(k_swap * p_s, v_dim) + s_delta + s_time
            if p_emb_s is not None:
                swap_tok = swap_tok + p_emb_s.repeat(k_swap, 1)
            swap_tok = swap_tok * swap_mask.view(-1, 1).to(out_dt)

        tot_tok = torch.cat([apex_tok, swap_tok], dim=0)
        tot_mask = torch.cat([apex_mask.view(-1), swap_mask.view(-1)], dim=0)
        return tot_tok, tot_mask

    @staticmethod
    def compute_v2_phys_scalars(
        d_slice: torch.Tensor,
        vis_rms_slice: torch.Tensor,
        grip_spike_slice: torch.Tensor,
    ) -> torch.Tensor:
        """Clock-free physical button/reveal indicators `[has_demo, exec_max_vis_rms, exec_max_grip_spike, exec_cum_grip_events]`."""
        dev = d_slice.device
        if int(d_slice.numel()) == 0:
            return torch.zeros((4,), device=dev, dtype=torch.float32)
        has_demo = 1.0 if bool(d_slice.any().item()) else 0.0
        exec_m = ~d_slice
        if bool(exec_m.any().item()):
            ex_vis = float(vis_rms_slice[exec_m].float().max().item())
            ex_grip = float(grip_spike_slice[exec_m].float().max().item())
            ex_ev = min(2.0, float((grip_spike_slice[exec_m].float() > 0.25).float().sum().item()) / 4.0)
        else:
            ex_vis = 0.0
            ex_grip = 0.0
            ex_ev = 0.0
        return torch.tensor([has_demo, ex_vis, ex_grip, ex_ev], device=dev, dtype=torch.float32)

    def build_state_contact_tokens(
        self,
        curr_state_norm: torch.Tensor,
        prev_state_norm: torch.Tensor,
        phys_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dt = self.gdn_to_vlm[1].weight.dtype  # type: ignore[index]
        sc = curr_state_norm.float()
        ds = sc - prev_state_norm.float()
        dg = ds[:, 7:8].abs()
        arm_spd = torch.sqrt(ds[:, :7].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        parts = [sc, ds, dg, arm_spd]
        if self.cfg.v2_mode:
            if phys_scalars is None:
                phys_scalars = torch.zeros((int(sc.shape[0]), 4), device=sc.device, dtype=torch.float32)
            parts.append(phys_scalars.float())
        kin_vec = torch.cat(parts, dim=-1).to(dt)
        return self.state_to_vlm(kin_vec).view(int(sc.shape[0]), 2, self.cfg.vlm_width)


class DeltaGroundVLMSystem(nn.Module):
    """End-to-end `P2-M9: DeltaGround-VLM` binding `DeltaGroundVLMHeadModule` + `dg_vlm_lora` onto a `Pi05Model`."""

    def __init__(
        self,
        cfg: DeltaGroundVLMConfig,
        pi05_model: Pi05Model,
        *,
        template_vocab: Optional[Sequence[str]] = None,
        enable_grad_ckpt: bool = False,
        skip_lora_inject: bool = False,
        vlm_lora_mode: LoRAMode = "dg",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pi05 = pi05_model
        self.vlm_lora_mode: LoRAMode = vlm_lora_mode
        if not skip_lora_inject:
            self.pi05.freeze_backbone()
            for p in self.pi05.parameters():
                p.requires_grad = False
            self.dg_lora_params = inject_dg_vlm_lora(
                self.pi05,
                rank=cfg.dg_lora_rank,
                alpha=cfg.dg_lora_alpha,
                dropout=cfg.dg_lora_dropout,
                include_mlp=cfg.dg_lora_include_mlp,
            )
            set_vlm_lora_mode(self.pi05, self.vlm_lora_mode)
        else:
            self.dg_lora_params = []
        self.enable_grad_ckpt = bool(enable_grad_ckpt)

        dev = self.pi05.device
        self.head = DeltaGroundVLMHeadModule(cfg, template_vocab=template_vocab).to(dev)
        if self.pi05.config.precision == "bfloat16":
            self.head.cast_bf16_keep_fp32_gates()

    def _advect_nbr_mask(self, dev: torch.device) -> torch.Tensor:
        side = int(round(int(self.cfg.front_tokens_hist) ** 0.5))
        cached = getattr(self, "_nbr_mask_cache", None)
        if cached is None or cached.device != dev:
            cached = build_local_neighbor_mask(side, int(self.cfg.v2_advect_radius), dev)
            self._nbr_mask_cache = cached
        return cached

    def project_vis(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.cfg.vis_dim:
            return x
        vt_head = self.pi05.vision_tower.head
        with torch.no_grad():
            return vt_head(x.to(vt_head.weight.dtype))

    def _run_vlm_prefix_backbone(
        self,
        prefix_tokens: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
    ) -> torch.Tensor:
        set_vlm_lora_mode(self.pi05, self.vlm_lora_mode)
        ar_mask = torch.zeros((int(prefix_tokens.shape[1]),), dtype=torch.bool, device=prefix_tokens.device)
        attn_mask = make_attn_mask(prefix_pad_masks, ar_mask)
        positions = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        x0: torch.Tensor | None = prefix_tokens
        vlm_exp = self.pi05.paligemma_with_expert
        for blk in vlm_exp.layers:
            if self.enable_grad_ckpt and self.training:

                def _step(inp: torch.Tensor, b: Any = blk) -> torch.Tensor:
                    o0, _, _ = b(inp, None, positions, attn_mask, None)
                    assert o0 is not None
                    return o0

                assert x0 is not None
                x0 = torch.utils.checkpoint.checkpoint(_step, x0, use_reentrant=False, preserve_rng_state=True)
            else:
                x0, _, _ = blk(x0, None, positions, attn_mask, None)
        assert x0 is not None
        x0_out, _ = vlm_exp.vlm_norm(x0, None)
        return x0_out

    def forward_episodes(
        self,
        episodes: Sequence[Dict[str, Any]],
        *,
        force_pytorch_gdn: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Batched multi-episode forward pass (`O(W)` GDN scan per episode -> 1 fused `PaliGemma-2B` prefix pass!)."""
        dev = self.pi05.device
        vlm_dt = self.pi05.compute_dtype
        head_dt = self.head.gdn_to_vlm[1].weight.dtype  # type: ignore[index]

        front_list: List[torch.Tensor] = []
        wrist16_list: List[torch.Tensor] = []
        lt_list: List[torch.Tensor] = []
        lm_list: List[torch.Tensor] = []
        rev_tok_list: List[torch.Tensor] = []
        adv_list: List[torch.Tensor] = []
        rev_mask_list: List[torch.Tensor] = []
        gdn_st_tok_list: List[torch.Tensor] = []
        gdn_st_mask_list: List[torch.Tensor] = []

        tgt_lists: Dict[str, List[torch.Tensor]] = {
            "template_id": [],
            "xy1_m11": [],
            "has_xy1": [],
            "xy2_m11": [],
            "has_xy2": [],
            "is_stage_entry": [],
            "completed_count": [],
            "remaining_count": [],
            "stage_id": [],
            "color_id": [],
            "ordinal_id": [],
        }

        for ep in episodes:
            q_steps = ep["chunk_step_indices"].to(dev).long()
            m_steps = int(q_steps.numel())
            if m_steps == 0:
                continue

            ep_f = self.project_vis(ep["front_emb"].to(dev, non_blocking=True))
            ep_w = self.project_vis(ep["wrist_emb"].to(dev, non_blocking=True))
            ep_st = ep["state_norm"].to(dev, non_blocking=True).float()
            ep_dm = ep["is_demo"].to(dev, non_blocking=True).bool()
            exec_start = int(ep["exec_start_idx"])
            f_cur = self.project_vis(ep["exec_front_256"].to(dev, non_blocking=True))
            w_cur = self.project_vis(ep["exec_wrist_256"].to(dev, non_blocking=True))

            lt = ep["lang_tokens"].to(dev, non_blocking=True).unsqueeze(0).expand(m_steps, -1)
            lm = ep["lang_masks"].to(dev, non_blocking=True).unsqueeze(0).expand(m_steps, -1)

            with torch.no_grad():
                l_emb = self.pi05.paligemma_with_expert.embed_language(lt.long())
                lm_f = lm.to(l_emb.dtype).unsqueeze(-1)
                l_pool = (l_emb * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)

            t_tot = int(ep_f.shape[0])
            write_idx = select_episode_write_indices(t_tot, exec_start, self.head.mem_cfg, device=dev)
            w_len = int(write_idx.numel())
            f_w = ep_f[write_idx]
            w_w = ep_w[write_idx]
            s_w = ep_st[write_idx]
            d_w = ep_dm[write_idx]
            kin = self.head.writer.compute_frame_kinematics(f_w, s_w, d_w)
            w_idx_enc = (
                torch.where(d_w, write_idx, torch.zeros_like(write_idx))
                if (self.cfg.v2_mode and not self.cfg.v2_keep_exec_clock)
                else write_idx
            )
            raw_gdn_w, _ = self.head.writer.encode_frames_raw(f_w, w_w, s_w, w_idx_enc, d_w, kin)
            gdn_slots_w, _ = self.head.writer.apply_slot_conv(raw_gdn_w)

            p_jump = ep.get("prompt_cos_jump")
            p_jump_w: Optional[torch.Tensor] = None
            if p_jump is not None and w_len > 0:
                pj_d = p_jump.to(dev)
                w_l = write_idx.tolist()
                pj = torch.zeros((w_len,), device=dev, dtype=torch.float32)
                for wi in range(w_len):
                    lo = 0 if wi == 0 else w_l[wi - 1] + 1
                    hi = w_l[wi] + 1
                    if hi > lo:
                        pj[wi] = pj_d[lo:hi].float().max()
                p_jump_w = pj

            b_reset_w = (
                self.head.gdn.boundary_gate(gdn_slots_w[:, 0, :], kin, prompt_cos_jump=p_jump_w)
                if (self.head.gdn.boundary_gate is not None and w_len > 0)
                else None
            )

            if w_len > 0:
                k1, v1, g1, b1, _ = self.head.gdn.layer1.project_slots(
                    gdn_slots_w.unsqueeze(0),
                    vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
                    grip_spike=kin["grip_spike"].unsqueeze(0),
                    boundary_reset=b_reset_w.unsqueeze(0) if b_reset_w is not None else None,
                )
                raw_surp = self.head.gdn.layer1.compute_exact_write_surprise(
                    k1.detach(),
                    v1.detach(),
                    g1.detach(),
                    b1.detach(),
                    num_write_slots=self.cfg.num_write_slots,
                    force_pytorch=force_pytorch_gdn,
                ).squeeze(0)
                beta = float(self.cfg.surprise_ema_beta)
                s_list = raw_surp.float().tolist()
                g_list = kin["grip_spike"].float().tolist()
                d_list = d_w.tolist()
                br_l = b_reset_w.float().tolist() if b_reset_w is not None else [0.0] * w_len
                cur_s = 0.0
                cur_p = 0.0
                sal_py: List[float] = []
                pvis_py: List[float] = []
                for wi in range(w_len):
                    if br_l[wi] > 0.5 or (wi > 0 and d_list[wi - 1] and not d_list[wi]):
                        cur_s = 0.0
                        cur_p = 0.0
                    inst_p = float(s_list[wi])
                    inst_s = inst_p + 0.5 * (1.0 if g_list[wi] > 0.25 else 0.0) + 0.35 * float(g_list[wi])
                    cur_p = beta * cur_p + (1.0 - beta) * inst_p
                    cur_s = beta * cur_s + (1.0 - beta) * inst_s
                    pvis_py.append(max(inst_p, cur_p))
                    sal_py.append(max(inst_s, cur_s))
                sal_w = torch.tensor(sal_py, device=dev, dtype=torch.float32)
                pvis_w = torch.tensor(pvis_py, device=dev, dtype=torch.float32)
            else:
                sal_w = torch.zeros((0,), device=dev, dtype=torch.float32)
                pvis_w = torch.zeros((0,), device=dev, dtype=torch.float32)

            f_w_reg = self.head.regularize_v2_vis_tokens(f_w, allow_patch_drop=False) if self.cfg.v2_mode else f_w
            f_cur_reg = self.head.regularize_v2_vis_tokens(f_cur, allow_patch_drop=True) if self.cfg.v2_mode else f_cur
            tw_f256 = ep.get("target_write_front_256")
            tw_abs = ep.get("target_write_abs_idx")
            tw_abs_dev = tw_abs.to(dev) if (tw_abs is not None and int(tw_abs.numel()) > 0) else None
            need_hr256 = (
                self.cfg.v2_mode
                and (
                    int(self.cfg.reveal_tokens_per_frame) > int(f_w.shape[1])
                    or int(self.cfg.v2_pyramid_apex_hr16) > 0
                )
                and tw_f256 is not None
                and tw_abs_dev is not None
            )
            hr_tgt_tokens = (
                int(self.cfg.front_tokens_curr)
                if int(self.cfg.v2_pyramid_apex_hr16) > 0
                else int(self.cfg.reveal_tokens_per_frame)
            )

            def _gather_ep_hr256(sel_w_idx: torch.Tensor) -> Optional[torch.Tensor]:
                if not need_hr256 or int(sel_w_idx.numel()) == 0:
                    return None
                assert tw_abs_dev is not None and tw_f256 is not None
                req_f = write_idx[sel_w_idx]
                n_tw = int(tw_abs_dev.numel())
                pos = torch.searchsorted(tw_abs_dev, req_f).clamp(0, n_tw - 1)
                exact_m = tw_abs_dev[pos] == req_f
                pos_cpu = pos.detach().cpu()
                hr_proj = self.project_vis(tw_f256[pos_cpu].to(dev, non_blocking=True))
                hr_reg = self.head.regularize_v2_vis_tokens(hr_proj, allow_patch_drop=False)
                if bool(exact_m.all().item()):
                    return hr_reg
                fb_up = self.head._upsample_vis_grid(f_w_reg[sel_w_idx], hr_tgt_tokens)
                return torch.where(exact_m.view(-1, 1, 1), hr_reg, fb_up)

            u_ends: List[int] = []
            has_hist_l: List[bool] = []
            phys_l: List[torch.Tensor] = []
            adv_on_ep = bool(self.head.coda.adv_on)
            if adv_on_ep:
                nbr_m = self._advect_nbr_mask(dev)
                w_flow_ep = (
                    compute_patch_flow_warps(
                        f_w[:-1], f_w[1:], beta=self.cfg.v2_advect_beta, id_bias=self.cfg.v2_advect_id_bias, nbr_mask=nbr_m
                    )
                    if w_len >= 2
                    else torch.empty((0, int(f_w.shape[1]), int(f_w.shape[1])), device=dev, dtype=torch.float32)
                )
                f_cur64 = pool_spatial_tokens_to_square(f_cur, int(f_w.shape[1]))
            for qi in range(m_steps):
                c_f = int(q_steps[qi].item()) + 1
                ue = int(torch.searchsorted(write_idx, c_f, right=False).item()) if w_len > 0 else 0
                u_ends.append(ue)
                has_hist_l.append(ue > 0)
                if self.cfg.v2_mode:
                    if int(self.cfg.v2_pyramid_apex_hr16) > 0:
                        k_ap = int(self.cfg.v2_pyramid_apex_hr16)
                        k_sw = max(0, int(self.cfg.reveal_keyframes) - k_ap)
                        apex_sel_q, swap_sel_q, sp = select_v2_pyramid_reveal_indices(
                            write_idx[:ue],
                            pvis_w[:ue],
                            kin["vis_motion_rms"][:ue],
                            b_reset_w[:ue] if b_reset_w is not None else None,
                            cutoff_excl_frame=c_f,
                            k_apex=k_ap,
                            k_swap=k_sw,
                            uniform_anchors=self.cfg.v2_uniform_anchors,
                            flush_thresh=self.cfg.boundary_flush_threshold,
                            front_hist_w=f_w[:ue],
                            is_demo_w=d_w[:ue],
                            prominence_tau=self.cfg.v2_reveal_prominence_tau,
                            v3_median_ref=bool(self.cfg.v3_mode),
                        )
                        if adv_on_ep:
                            p_c = int(f_w.shape[1])
                            if ue > 0:
                                w_cur_q = compute_patch_flow_warps(
                                    f_w[ue - 1 : ue],
                                    f_cur64[qi : qi + 1],
                                    beta=self.cfg.v2_advect_beta,
                                    id_bias=self.cfg.v2_advect_id_bias,
                                    nbr_mask=nbr_m,
                                )[0]
                                adv_list.append(
                                    accumulate_apex_advection(w_flow_ep, apex_sel_q.tolist(), ue - 1, w_cur_q, k_ap)
                                )
                            else:
                                adv_list.append(
                                    torch.eye(p_c, device=dev, dtype=torch.float32).unsqueeze(0).repeat(k_ap, 1, 1)
                                )
                        prev_ap_q = torch.clamp(apex_sel_q - 1, min=int(sp))
                        f_ap_hr_q = _gather_ep_hr256(apex_sel_q)
                        f_prev_ap_hr_q = _gather_ep_hr256(prev_ap_q) if f_ap_hr_q is not None else None
                        rt, rm = self.head.encode_v2_pyramid_reveal_tokens(
                            f_w_reg[:ue],
                            write_idx[:ue],
                            d_w[:ue],
                            apex_sel_q,
                            swap_sel_q,
                            sp,
                            f_apex_hr=f_ap_hr_q,
                            f_prev_apex_hr=f_prev_ap_hr_q,
                        )
                    else:
                        v_sel, sp = select_v2_hybrid_reveal_indices(
                            write_idx[:ue],
                            pvis_w[:ue],
                            kin["vis_motion_rms"][:ue],
                            b_reset_w[:ue] if b_reset_w is not None else None,
                            cutoff_excl_frame=c_f,
                            num_keyframes=self.cfg.reveal_keyframes,
                            uniform_anchors=self.cfg.v2_uniform_anchors,
                            flush_thresh=self.cfg.boundary_flush_threshold,
                            front_hist_w=f_w[:ue],
                            is_demo_w=d_w[:ue],
                            apex_peaks=self.cfg.v2_apex_peaks,
                        )
                        prev_sel_q = torch.clamp(v_sel - 1, min=int(sp))
                        f_sel_hr_q = _gather_ep_hr256(v_sel)
                        f_prev_hr_q = _gather_ep_hr256(prev_sel_q) if f_sel_hr_q is not None else None
                        rt, rm = self.head.encode_v2_reveal_tokens(
                            f_w_reg[:ue],
                            write_idx[:ue],
                            d_w[:ue],
                            v_sel,
                            sp,
                            f_sel_hr=f_sel_hr_q,
                            f_prev_hr=f_prev_hr_q,
                        )
                    phys_l.append(
                        self.head.compute_v2_phys_scalars(
                            d_w[sp:ue], kin["vis_motion_rms"][sp:ue], kin["grip_spike"][sp:ue]
                        )
                    )
                else:
                    v_sel = self.head.reveal_selector.select_vlm_surprise_indices(
                        write_idx, d_w, sal_w, pvis_w, b_reset_w, cutoff_excl_frame=c_f
                    )
                    f_vlm = f_w[v_sel] if int(v_sel.numel()) > 0 else f_w[:0]
                    rt, rm = self.head.reveal_selector.encode_vlm_surprise_tokens(f_vlm)
                rev_tok_list.append(rt)
                rev_mask_list.append(rm)

            u_ends_t = torch.tensor(u_ends, device=dev, dtype=torch.long)
            decay_t = torch.ones((m_steps,), device=dev, dtype=torch.float32)
            f_ctx_16 = pool_spatial_tokens_to_square(f_cur, 16)
            ctx_q = self.head.ctx_q_proj(
                torch.cat([l_pool.unsqueeze(1).to(head_dt), f_ctx_16.to(head_dt)], dim=1)
            )
            gdn_tok, _ = self.head.gdn.forward_interleaved_episode_readout(
                gdn_slots_w,
                kin,
                b_reset_w,
                u_ends=u_ends_t,
                decay_factors=decay_t,
                ctx_tokens=ctx_q,
                force_pytorch=force_pytorch_gdn,
            )
            gdn_vlm = self.head.gdn_to_vlm(gdn_tok.to(head_dt)).to(vlm_dt)

            prev_idx = torch.maximum(
                torch.full_like(q_steps, exec_start),
                q_steps - int(self.cfg.write_stride_exec),
            )
            st_cur = ep_st[q_steps]
            st_prev = ep_st[prev_idx]
            phys_b = torch.stack(phys_l, dim=0) if (self.cfg.v2_mode and phys_l) else None
            st_vlm = self.head.build_state_contact_tokens(st_cur, st_prev, phys_scalars=phys_b).to(vlm_dt)

            gs_tok = torch.cat([gdn_vlm, st_vlm], dim=1)
            has_h_t = torch.tensor(has_hist_l, device=dev, dtype=torch.bool)
            gs_mask = has_h_t.unsqueeze(-1).expand(m_steps, int(gs_tok.shape[1])).clone()
            gs_mask[:, : self.cfg.num_persistent_tokens] = True
            gs_mask[:, -2:] = True  # Current contact kinematics always valid!

            w16 = pool_spatial_tokens_to_square(w_cur, self.cfg.wrist_tokens_curr_pool).to(vlm_dt)
            w16 = w16 + self.head.wrist_view_emb.to(device=dev, dtype=vlm_dt)

            f_cur_vlm = self.head.encode_v2_front_coords(f_cur_reg.to(vlm_dt)) if self.cfg.v2_mode else f_cur.to(vlm_dt)
            front_list.append(f_cur_vlm)
            wrist16_list.append(w16)
            lt_list.append(lt)
            lm_list.append(lm)
            gdn_st_tok_list.append(gs_tok)
            gdn_st_mask_list.append(gs_mask)

            for tk in tgt_lists:
                if tk in ep:
                    tgt_lists[tk].append(ep[tk].to(dev))

        front_b = torch.cat(front_list, dim=0)
        wrist16_b = torch.cat(wrist16_list, dim=0)
        lt_b = torch.cat(lt_list, dim=0)
        lm_b = torch.cat(lm_list, dim=0)
        rev_tok_b = torch.stack(rev_tok_list, dim=0).to(vlm_dt)
        rev_mask_b = torch.stack(rev_mask_list, dim=0)
        gs_tok_b = torch.cat(gdn_st_tok_list, dim=0)
        gs_mask_b = torch.cat(gdn_st_mask_list, dim=0)

        extra_embs = torch.cat([rev_tok_b, gs_tok_b], dim=1)
        extra_mask = torch.cat([rev_mask_b, gs_mask_b], dim=1)

        prefix_tokens, prefix_pad_masks, _ = self.pi05.embed_prefix(
            lt_b,
            lm_b,
            image_embs=[front_b, wrist16_b],
            extra_prefix_embs=extra_embs,
            extra_prefix_mask=extra_mask,
        )
        x0_out = self._run_vlm_prefix_backbone(prefix_tokens, prefix_pad_masks)

        n_f = int(front_b.shape[1])
        n_w = int(wrist16_b.shape[1])
        n_l = int(lt_b.shape[1])
        n_rev = int(rev_tok_b.shape[1])
        n_gs = int(gs_tok_b.shape[1])

        front_vlm = x0_out[:, :n_f, :]
        lang_slice = x0_out[:, n_f + n_w : n_f + n_w + n_l, :]
        lm_f = lm_b.to(lang_slice.dtype).unsqueeze(-1)
        lang_pool = (lang_slice * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)
        rev_vlm = x0_out[:, n_f + n_w + n_l : n_f + n_w + n_l + n_rev, :] if self.cfg.v2_mode else None
        gdn_slice = x0_out[:, -n_gs:, :]
        gm_f = gs_mask_b.to(gdn_slice.dtype).unsqueeze(-1)
        gdn_pool = (gdn_slice * gm_f).sum(dim=1) / gm_f.sum(dim=1).clamp(min=1.0)

        preds = self.head.coda(
            front_vlm,
            lang_pool,
            gdn_pool,
            reveal_vlm=rev_vlm,
            reveal_raw=rev_tok_b if (self.cfg.v2_mode and self.cfg.v2_raw_siglip_reveal) else None,
            reveal_mask=rev_mask_b if self.cfg.v2_mode else None,
            reveal_advect=(torch.stack(adv_list, dim=0) if adv_list else None),
        )
        preds["front_vlm"] = front_vlm
        preds["lang_pool"] = lang_pool
        preds["gdn_pool"] = gdn_pool
        targets = {k: torch.cat(v, dim=0) for k, v in tgt_lists.items() if v}
        return preds, targets

    def compute_loss(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev = preds["xy_pri"].device
        s_side = self.head.coda.front_side
        centers = ((torch.arange(s_side, device=dev, dtype=torch.float32) + 0.5) / float(s_side)) * 2.0 - 1.0
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)
        sigma2 = (1.25 / float(s_side)) ** 2

        tgt_tmpl = targets["template_id"].long()
        has_xy1 = targets["has_xy1"].bool()
        tgt_xy1 = targets["xy1_m11"].float()
        has_xy2 = targets["has_xy2"].bool()
        tgt_xy2 = targets["xy2_m11"].float()
        is_entry = targets.get("is_stage_entry")

        if bool(self.cfg.v3_mode) and is_entry is not None and int(is_entry.numel()) == int(tgt_tmpl.numel()):
            ce_vec = F.cross_entropy(preds["template_logits"], tgt_tmpl, label_smoothing=0.04, reduction="none")
            w_ce = torch.where(
                is_entry.bool(),
                torch.full_like(ce_vec, float(self.cfg.v3_transition_ce_boost)),
                torch.ones_like(ce_vec),
            )
            tmpl_loss = (ce_vec * w_ce).sum() / w_ce.sum().clamp(min=1.0)
        else:
            tmpl_loss = F.cross_entropy(preds["template_logits"], tgt_tmpl, label_smoothing=0.04)
        xy_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        px_err_1 = torch.zeros_like(tgt_tmpl, dtype=torch.float32)

        if bool(has_xy1.any()):
            p1 = preds["xy_pri"][has_xy1]
            t1 = tgt_xy1[has_xy1]
            d2_1 = ((grid_xy - t1.unsqueeze(1)) ** 2).sum(dim=-1)
            tgt_p1 = F.softmax(-0.5 * d2_1 / sigma2, dim=-1)
            if self.cfg.v2_mode and is_entry is not None and int(is_entry.numel()) == int(has_xy1.numel()):
                ent_1 = is_entry.bool()[has_xy1]
                reg1_vec = F.smooth_l1_loss(p1, t1, beta=0.03, reduction="none").mean(dim=-1)
                kl1_vec = F.kl_div(
                    F.log_softmax(preds["hm_logits"][has_xy1, 0, :], dim=-1), tgt_p1, reduction="none"
                ).sum(dim=-1)
                w1 = torch.where(
                    ent_1,
                    torch.ones_like(reg1_vec),
                    torch.full_like(reg1_vec, float(self.cfg.v2_non_entry_xy_weight)),
                )
                w1_sum = w1.sum()
                if float(w1_sum.item()) > 1e-6:
                    reg1 = (reg1_vec * w1).sum() / w1_sum
                    kl1 = (kl1_vec * w1).sum() / w1_sum
                else:
                    reg1 = reg1_vec.mean() * 0.0
                    kl1 = kl1_vec.mean() * 0.0
            else:
                reg1 = F.smooth_l1_loss(p1, t1, beta=0.03)
                kl1 = F.kl_div(F.log_softmax(preds["hm_logits"][has_xy1, 0, :], dim=-1), tgt_p1, reduction="batchmean")
            xy_loss = xy_loss + 2.5 * reg1 + 0.35 * kl1
            with torch.no_grad():
                fin_p = apply_peak_window_to_preds(preds, s_side=s_side, radius=2)
                px_pred_all = norm_xy_to_pixel_xy(fin_p["xy_pri"].detach(), align_corners=False)
                px_tgt_all = norm_xy_to_pixel_xy(tgt_xy1, align_corners=False)
                px_err_1 = torch.linalg.vector_norm(px_pred_all - px_tgt_all, ord=2, dim=-1)

        if bool(has_xy2.any()):
            p2 = preds["xy_sec"][has_xy2]
            t2 = tgt_xy2[has_xy2]
            d2_2 = ((grid_xy - t2.unsqueeze(1)) ** 2).sum(dim=-1)
            tgt_p2 = F.softmax(-0.5 * d2_2 / sigma2, dim=-1)
            if self.cfg.v2_mode and is_entry is not None and int(is_entry.numel()) == int(has_xy2.numel()):
                ent_2 = is_entry.bool()[has_xy2]
                reg2_vec = F.smooth_l1_loss(p2, t2, beta=0.03, reduction="none").mean(dim=-1)
                kl2_vec = F.kl_div(
                    F.log_softmax(preds["hm_logits"][has_xy2, 1, :], dim=-1), tgt_p2, reduction="none"
                ).sum(dim=-1)
                w2 = torch.where(
                    ent_2,
                    torch.ones_like(reg2_vec),
                    torch.full_like(reg2_vec, float(self.cfg.v2_non_entry_xy_weight)),
                )
                w2_sum = w2.sum()
                if float(w2_sum.item()) > 1e-6:
                    reg2 = (reg2_vec * w2).sum() / w2_sum
                    kl2 = (kl2_vec * w2).sum() / w2_sum
                else:
                    reg2 = reg2_vec.mean() * 0.0
                    kl2 = kl2_vec.mean() * 0.0
            else:
                reg2 = F.smooth_l1_loss(p2, t2, beta=0.03)
                kl2 = F.kl_div(F.log_softmax(preds["hm_logits"][has_xy2, 1, :], dim=-1), tgt_p2, reduction="batchmean")
            xy_loss = xy_loss + 1.5 * reg2 + 0.20 * kl2

        cnt_c = targets["completed_count"].clamp(0, self.cfg.max_count_classes - 1).long()
        cnt_r = targets["remaining_count"].clamp(0, self.cfg.max_count_classes - 1).long()
        ord_t = targets["ordinal_id"].clamp(0, self.cfg.max_count_classes - 1).long()
        stg_t = targets["stage_id"].clamp(0, self.cfg.num_stage_classes - 1).long()
        col_t = targets["color_id"].clamp(0, self.cfg.num_color_classes - 1).long()

        aux_loss = (
            0.25 * F.cross_entropy(preds["completed_count_logits"], cnt_c)
            + 0.25 * F.cross_entropy(preds["remaining_count_logits"], cnt_r)
            + 0.15 * F.cross_entropy(preds["ordinal_logits"], ord_t)
            + 0.15 * F.cross_entropy(preds["stage_logits"], stg_t)
            + 0.15 * F.cross_entropy(preds["color_logits"], col_t)
        )
        loss = tmpl_loss + float(self.cfg.xy_loss_weight) * xy_loss + aux_loss
        with torch.no_grad():
            acc = (preds["template_logits"].argmax(dim=-1) == tgt_tmpl).float().mean()
            if self.cfg.v2_mode and is_entry is not None and int(is_entry.numel()) == int(has_xy1.numel()):
                ent_xy = has_xy1 & is_entry.bool()
                m_px = px_err_1[ent_xy].mean() if bool(ent_xy.any()) else (
                    px_err_1[has_xy1].mean() if bool(has_xy1.any()) else torch.tensor(0.0, device=dev)
                )
            else:
                m_px = px_err_1[has_xy1].mean() if bool(has_xy1.any()) else torch.tensor(0.0, device=dev)
        return {
            "loss": loss,
            "tmpl_loss": tmpl_loss.detach(),
            "xy_loss": xy_loss.detach(),
            "tmpl_acc": acc,
            "px_err": m_px,
        }

    def export_payload(self, state_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        dg_lora_sd: Dict[str, torch.Tensor] = {}
        for name, p in self.pi05.named_parameters():
            if "dg_lora_" in name:
                dg_lora_sd[name] = p.detach().cpu()
        return {
            "deltaground_vlm_config": self.cfg.to_dict(),
            "template_vocab": list(self.head.template_vocab),
            "state_stats": state_stats or MME_VLA_NORM_STATS,
            "head_state_dict": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "dg_lora_state_dict": dg_lora_sd,
        }

    def load_payload(self, payload: Dict[str, Any], *, strict: bool = True, allow_slot_interp: bool = False) -> None:
        hsd = dict(payload["head_state_dict"])
        slot_k = "reveal_selector.vlm_hist_slot_emb"
        if allow_slot_interp:
            if slot_k in hsd:
                cur_slot = self.head.reveal_selector.vlm_hist_slot_emb
                src_slot = hsd[slot_k]
                if src_slot.shape != cur_slot.shape and src_slot.ndim == 3 and cur_slot.ndim == 3:
                    hsd[slot_k] = (
                        F.interpolate(
                            src_slot.permute(1, 2, 0).float(),
                            size=int(cur_slot.shape[0]),
                            mode="linear",
                            align_corners=True,
                        )
                        .permute(2, 0, 1)
                        .to(dtype=src_slot.dtype)
                        .contiguous()
                    )
            if self.head.hr16_pos_proj is not None and "hr16_pos_proj.weight" not in hsd:
                hsd["hr16_pos_proj.weight"] = torch.zeros_like(self.head.hr16_pos_proj.weight)
            # `#426` Wave-16 warm-start: keep fresh (zero-init `adv_out`) advection params when absent from source ckpt
            for k_new, v_new in self.head.state_dict().items():
                if k_new.startswith("coda.adv_") and k_new not in hsd:
                    hsd[k_new] = v_new.detach().clone()
        self.head.load_state_dict(hsd, strict=strict)
        own_sd = self.pi05.state_dict()
        for k, v in payload["dg_lora_state_dict"].items():
            if k in own_sd:
                own_sd[k].copy_(v.to(device=own_sd[k].device, dtype=own_sd[k].dtype))


class ContinuousStageEntryLatch:
    """Continuous twin of `StageEntrySubgoalLatch` (`#395(2)` `P2-M11`): latches neutral-arm `16x16` heatmaps & target ROIs!"""

    def __init__(self, twin_to_grounded: Optional[Mapping[str, str]] = None) -> None:
        self._twin_map: Dict[str, str] = dict(twin_to_grounded or {})
        self.reset()

    def reset(self) -> None:
        self._active_canon: str = ""
        self._p1: Optional[torch.Tensor] = None
        self._p2: Optional[torch.Tensor] = None
        self._xy1: Optional[torch.Tensor] = None
        self._xy2: Optional[torch.Tensor] = None
        self._roi1: Optional[torch.Tensor] = None
        self._roi2: Optional[torch.Tensor] = None

    def to_device(self, dev: torch.device) -> None:
        if self._p1 is not None:
            self._p1 = self._p1.to(dev)
        if self._p2 is not None:
            self._p2 = self._p2.to(dev)
        if self._xy1 is not None:
            self._xy1 = self._xy1.to(dev)
        if self._xy2 is not None:
            self._xy2 = self._xy2.to(dev)
        if self._roi1 is not None:
            self._roi1 = self._roi1.to(dev)
        if self._roi2 is not None:
            self._roi2 = self._roi2.to(dev)

    def step(
        self,
        tmpl_str: str,
        p1_c: torch.Tensor,
        p2_c: torch.Tensor,
        xy1_c: torch.Tensor,
        xy2_c: torch.Tensor,
        roi1_c: torch.Tensor,
        roi2_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        canon = self._twin_map.get(tmpl_str, tmpl_str) if tmpl_str else ""
        if self._p1 is None or (canon and canon != self._active_canon):
            self._p1 = p1_c.detach().clone()
            self._p2 = p2_c.detach().clone()
            self._xy1 = xy1_c.detach().clone()
            self._xy2 = xy2_c.detach().clone()
            self._roi1 = roi1_c.detach().clone()
            self._roi2 = roi2_c.detach().clone()
            if canon:
                self._active_canon = canon
        assert (
            self._p1 is not None
            and self._p2 is not None
            and self._xy1 is not None
            and self._xy2 is not None
            and self._roi1 is not None
            and self._roi2 is not None
        )
        return self._p1, self._p2, self._xy1, self._xy2, self._roi1, self._roi2


def build_step_continuous_grounding_packet(
    preds: Mapping[str, torch.Tensor],
    bi: int,
    tmpl_vocab: Sequence[str],
    cont_latch: ContinuousStageEntryLatch,
) -> Dict[str, torch.Tensor]:
    """Shared train=eval builder producing `100%` identical continuous grounding tensors for `ContinuousVLMGroundingBridge`."""
    tid = int(preds["template_logits"][bi].argmax(dim=-1).item())
    t_str = tmpl_vocab[tid] if 0 <= tid < len(tmpl_vocab) else ""
    f_vlm_i = preds["front_vlm"][bi]  # (256, vlm_width)
    v_dt = f_vlm_i.dtype
    p1_c = F.softmax(preds["hm_logits"][bi, 0].float(), dim=-1)
    p2_c = F.softmax(preds["hm_logits"][bi, 1].float(), dim=-1)
    xy1_c = preds["xy_pri"][bi].float()
    xy2_c = preds["xy_sec"][bi].float()
    roi1_c = torch.matmul(p1_c.to(v_dt), f_vlm_i)
    roi2_c = torch.matmul(p2_c.to(v_dt), f_vlm_i)
    p1_l, p2_l, xy1_l, xy2_l, roi1_l, roi2_l = cont_latch.step(
        t_str, p1_c, p2_c, xy1_c, xy2_c, roi1_c, roi2_c
    )
    f8 = pool_spatial_tokens_to_square(f_vlm_i.unsqueeze(0), 64).squeeze(0)
    hm_m = torch.stack([p1_c, p2_c, p1_l.to(p1_c.device), p2_l.to(p2_c.device)], dim=0)
    rev_p = preds["rev_pool"][bi] if preds.get("rev_pool") is not None else torch.zeros_like(preds["lang_pool"][bi])
    roi_v = torch.stack(
        [
            roi1_c,
            roi1_l.to(device=f_vlm_i.device, dtype=v_dt),
            roi2_c,
            roi2_l.to(device=f_vlm_i.device, dtype=v_dt),
            preds["lang_pool"][bi].to(v_dt),
            preds["gdn_pool"][bi].to(v_dt),
            rev_p.to(v_dt),
        ],
        dim=0,
    )
    sem_p = torch.cat(
        [
            F.softmax(preds["stage_logits"][bi].float(), dim=-1),
            F.softmax(preds["color_logits"][bi].float(), dim=-1),
            F.softmax(preds["ordinal_logits"][bi].float(), dim=-1),
            F.softmax(preds["remaining_count_logits"][bi].float(), dim=-1),
        ],
        dim=-1,
    )
    # Pad/trim `sem_p` to canonical `48-D` (`15+9+12+12`) so tiny CPU configs also match `ContinuousVLMGroundingBridge`!
    if int(sem_p.numel()) < 48:
        sem_p = F.pad(sem_p, (0, 48 - int(sem_p.numel())))
    elif int(sem_p.numel()) > 48:
        sem_p = sem_p[:48]
    xy_c = torch.cat([xy1_c, xy1_l.to(xy1_c.device), xy2_c, xy2_l.to(xy2_c.device)], dim=-1)
    return {
        "front_vlm_8x8": f8,
        "hm_maps": hm_m,
        "roi_vecs": roi_v,
        "sem_probs": sem_p,
        "xy_coords": xy_c,
        "active_mask": torch.tensor(True, device=f_vlm_i.device, dtype=torch.bool),
    }


class DeltaGroundVLMEvalAdapter:
    """Online closed-loop `P2-M9: DeltaGround-VLM` subgoal predictor sharing `Pi05Model`'s `PaliGemma-2B` (`#334`)."""

    def __init__(
        self,
        system: DeltaGroundVLMSystem,
        *,
        restore_lora_mode: LoRAMode = "none",
        peak_window_radius: int = 2,
        fold_occlusion_twins: bool = False,
        stage_entry_latch: bool = False,
        min_template_confidence: float = 0.0,
        abstain_ordinal_counting: bool = False,
        abstain_place_order: bool = True,
        abstain_temporal_button: bool = True,
        dynamic_policy_lora_on_subgoal: bool = False,
        non_subgoal_lora_mode: LoRAMode = "none",
        enable_reactive_lora_routing: bool = False,
        tokenizer_path: str | Path | None = None,
        state_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        from pi05.policy import Pi05Tokenizer

        self.system = system.eval()
        self.head = system.head
        self.pi05 = system.pi05
        self.cfg = system.cfg
        self.mem_cfg = self.head.mem_cfg
        self.enable_reactive_lora_routing = bool(enable_reactive_lora_routing)
        self.device = self.pi05.device
        self.restore_lora_mode: LoRAMode = restore_lora_mode
        self.peak_window_radius = int(peak_window_radius)
        self.fold_occlusion_twins = bool(fold_occlusion_twins)
        self.stage_entry_latch = bool(stage_entry_latch)
        self.min_template_confidence = float(min_template_confidence)
        self.abstain_ordinal_counting = bool(abstain_ordinal_counting)
        self.abstain_place_order = bool(abstain_place_order)
        self.abstain_temporal_button = bool(abstain_temporal_button)
        self.dynamic_policy_lora_on_subgoal = bool(dynamic_policy_lora_on_subgoal)
        self.non_subgoal_lora_mode: LoRAMode = non_subgoal_lora_mode

        self.tokenizer = Pi05Tokenizer(
            max_len=self.cfg.max_prompt_len, discrete_state_input=False, tokenizer_path=tokenizer_path
        )
        self.normalizer = ActionStateNormalizer(mode="quantile_delta", stats=state_stats or MME_VLA_NORM_STATS)
        self._prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._twin_pairs = build_occlusion_twin_template_pairs(self.head.template_vocab)
        self._twin_to_grounded = {
            self.head.template_vocab[u_i]: self.head.template_vocab[g_i] for u_i, g_i in self._twin_pairs
        }
        self._stage_latch = StageEntrySubgoalLatch(self._twin_to_grounded)
        self._cont_latch = ContinuousStageEntryLatch(self._twin_to_grounded)
        from .corm import CausalObjectCentricMemory

        self.enable_corm: bool = os.environ.get("PI05_DISABLE_CORM", "0") != "1"
        self.corm = CausalObjectCentricMemory(device=self.device)
        init_mode: LoRAMode = self.non_subgoal_lora_mode if self.dynamic_policy_lora_on_subgoal else self.restore_lora_mode
        set_vlm_lora_mode(self.pi05, init_mode)
        self.reset()

    @property
    def wants_raw_rgb(self) -> bool:
        return bool(self.enable_corm)

    @torch.no_grad()
    def ingest_raw_rgb_stream(
        self,
        front_rgb_seq: Sequence[np.ndarray] | np.ndarray,
        raw_states_seq: Sequence[np.ndarray] | np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        if self.enable_corm:
            self.corm.ingest_raw_stream(
                front_rgb_seq, raw_states_seq, task_goal=task_goal, is_demo=is_demo
            )

    def reset(self) -> None:
        self.stream_total_frames: int = 0
        self.demo_frames_in_session: int = 0
        self.session_start_frame: int = 0
        self.session_exec_start_frame: int = 0
        self._last_was_demo: Optional[bool] = None
        self._last_prompt_str: Optional[str] = None
        self._pending_prompt_jump: float = 0.0
        self._prepared_write_batch: Optional[Tuple[List[int], List[float], List[bool], bool]] = None

        self._prev_w_front: Optional[torch.Tensor] = None
        self._prev_w_state: Optional[torch.Tensor] = None
        self._prev_w_demo: Optional[torch.Tensor] = None
        self._conv_hist: Optional[torch.Tensor] = None

        self._w_steps: List[int] = []
        self._w_is_demo: List[bool] = []
        self._w_front_hist: List[torch.Tensor] = []
        self._w_front_hist_256: List[torch.Tensor] = []
        self._w_flow: List[torch.Tensor] = []
        self._w_gdn_slots: List[torch.Tensor] = []
        self._w_state_norm: List[torch.Tensor] = []
        self._w_vis_rms: List[float] = []
        self._w_grip_spike: List[float] = []
        self._w_b_reset: List[float] = []
        self._w_salience: List[float] = []
        self._w_pure_vis_salience: List[float] = []

        self._committed_u: int = 0
        self._gdn_states = self.head.gdn.init_empty_states(1, self.device)
        self._surp_l1_state = torch.zeros(
            (1, self.cfg.gdn_heads, self.cfg.gdn_d_v, self.cfg.gdn_d_k), device=self.device, dtype=torch.float32
        )
        self._sal_ema: float = 0.0
        self._pvis_ema: float = 0.0
        self._last_pred_subgoal: str = ""
        self._last_loop_depth: int = 1
        self._last_b_reset_max: float = 0.0
        self._stage_latch.reset()
        self._cont_latch.reset()
        if hasattr(self, "corm"):
            self.corm.reset()
        set_vlm_lora_mode(
            self.pi05, self.non_subgoal_lora_mode if self.dynamic_policy_lora_on_subgoal else self.restore_lora_mode
        )

    def _encode_prompt(self, task_goal: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = task_goal.strip().lower()
        if key not in self._prompt_cache:
            v_max = int(self.pi05.paligemma_with_expert.embed_tokens.num_embeddings) - 1
            if v_max >= 4096:
                tok_np, mask_np = self.tokenizer.tokenize(key, norm_state=None, subgoal=None)
                lt = torch.from_numpy(tok_np).unsqueeze(0).to(self.device)
                lm = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
            else:
                ids = [max(1, ord(c) % max(2, v_max)) for c in key[: self.cfg.max_prompt_len]] or [1]
                pad = max(0, self.cfg.max_prompt_len - len(ids))
                lt = torch.tensor([ids + [0] * pad], device=self.device, dtype=torch.long)
                lm = torch.tensor([[True] * len(ids) + [False] * pad], device=self.device, dtype=torch.bool)
            with torch.no_grad():
                le = self.pi05.paligemma_with_expert.embed_language(lt.long().clamp(0, v_max))
                lm_f = lm.to(le.dtype).unsqueeze(-1)
                lp = (le * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)
            self._prompt_cache[key] = (lt, lm, lp.detach())
        return self._prompt_cache[key]

    @torch.no_grad()
    def select_write_indices(
        self,
        num_frames: int,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> List[int]:
        t_in = int(num_frames)
        if t_in <= 0:
            self._prepared_write_batch = None
            return []

        p_key = task_goal.strip().lower()
        if self._last_prompt_str is not None and p_key != self._last_prompt_str:
            self._pending_prompt_jump = 1.0
        self._last_prompt_str = p_key

        thresh = max(0, int(self.cfg.adaptive_demo_threshold))
        short_s = max(1, int(self.cfg.write_stride_demo))
        long_s = max(1, int(self.cfg.long_demo_stride))
        exec_s = max(1, int(self.cfg.write_stride_exec))
        start_long = ((thresh + long_s - 1) // long_s) * long_s
        dense_pref = max(0, int(self.mem_cfg.dense_prefix_frames))

        sel_local: List[int] = []
        sel_stream_steps: List[int] = []
        sel_p_jumps: List[float] = []
        sel_new_sess: List[bool] = []

        for i in range(t_in):
            s_idx = self.stream_total_frames + i
            exec_to_demo = (self._last_was_demo is False) and bool(is_demo)
            demo_to_exec = (self._last_was_demo is True) and (not bool(is_demo))
            self._last_was_demo = bool(is_demo)

            p_jump = self._pending_prompt_jump
            is_new_sess = bool(exec_to_demo or (p_jump > 0.5))
            if is_new_sess:
                self.demo_frames_in_session = 0
                self.session_start_frame = s_idx
                self.session_exec_start_frame = s_idx
                self._stage_latch.reset()
                self._cont_latch.reset()
            elif demo_to_exec:
                self.session_exec_start_frame = s_idx
                self._stage_latch.reset()
                self._cont_latch.reset()

            local_idx = s_idx - self.session_start_frame
            if is_demo:
                self.demo_frames_in_session += 1
                self.session_exec_start_frame = s_idx + 1
                is_final_demo = (
                    (self.demo_frames_in_session == int(total_demo_len))
                    if total_demo_len is not None
                    else (i == (t_in - 1))
                )
                if local_idx < dense_pref:
                    stride_hit = True
                elif local_idx < thresh:
                    stride_hit = (local_idx % short_s) == 0
                else:
                    stride_hit = (local_idx >= start_long) and ((local_idx - start_long) % long_s == 0)
                write_flag = stride_hit or is_final_demo or is_new_sess
            else:
                exec_rel = max(0, s_idx - self.session_exec_start_frame)
                stride_hit = (local_idx < dense_pref) or ((exec_rel % exec_s) == 0)
                write_flag = stride_hit or demo_to_exec or is_new_sess

            if write_flag:
                sel_local.append(i)
                sel_stream_steps.append(s_idx)
                sel_p_jumps.append(1.0 if is_new_sess else p_jump)
                sel_new_sess.append(is_new_sess)
                self._pending_prompt_jump = 0.0

        self.stream_total_frames += t_in
        self._prepared_write_batch = (sel_stream_steps, sel_p_jumps, sel_new_sess, bool(is_demo)) if sel_local else None
        return sel_local

    @torch.no_grad()
    def ingest_selected_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        del task_goal, norm_states
        prep = self._prepared_write_batch
        self._prepared_write_batch = None
        if prep is None:
            return
        sel_stream_steps, sel_p_jumps, sel_new_sess, prep_demo = prep
        is_demo = bool(prep_demo)
        w_cnt = len(sel_stream_steps)
        if w_cnt == 0 or int(front_tokens_256.shape[0]) == 0:
            return

        if sel_new_sess[0]:
            self._prev_w_front = None
            self._prev_w_state = None
            self._prev_w_demo = None
            self._conv_hist = None

        f_in = self.system.project_vis(front_tokens_256.to(self.device))
        f_w = pool_spatial_tokens_to_square(f_in, self.cfg.front_tokens_hist)
        if wrist_tokens_256 is not None:
            w_in = self.system.project_vis(wrist_tokens_256.to(self.device))
            w_w = pool_spatial_tokens_to_square(w_in, self.cfg.wrist_tokens_hist)
        else:
            w_w = torch.zeros((w_cnt, self.cfg.wrist_tokens_hist, f_w.shape[-1]), device=self.device, dtype=f_w.dtype)

        st_mme = self.normalizer.normalize_state(np.asarray(raw_states, dtype=np.float32))
        s_w = torch.from_numpy(np.asarray(st_mme, dtype=np.float32)).to(self.device)
        d_w = torch.full((w_cnt,), bool(is_demo), device=self.device, dtype=torch.bool)
        local_step_t = torch.tensor(
            [st_s - self.session_start_frame for st_s in sel_stream_steps], device=self.device, dtype=torch.long
        )

        kin = self.head.writer.compute_frame_kinematics(
            f_w, s_w, d_w, prev_front=self._prev_w_front, prev_state=self._prev_w_state, prev_is_demo=self._prev_w_demo
        )
        self._prev_w_front = f_w[-1].detach().clone()
        self._prev_w_state = s_w[-1].detach().clone()
        self._prev_w_demo = d_w[-1].detach().clone()

        w_idx_enc = (
            torch.where(d_w, local_step_t, torch.zeros_like(local_step_t))
            if (self.cfg.v2_mode and not self.cfg.v2_keep_exec_clock)
            else local_step_t
        )
        raw_gdn, _ = self.head.writer.encode_frames_raw(f_w, w_w, s_w, w_idx_enc, d_w, kin)
        gdn_slots, self._conv_hist = self.head.writer.apply_slot_conv(raw_gdn, conv_history=self._conv_hist)

        pj_t = torch.tensor(sel_p_jumps, device=self.device, dtype=torch.float32)
        br_t = (
            self.head.gdn.boundary_gate(gdn_slots[:, 0, :], kin, prompt_cos_jump=pj_t)
            if self.head.gdn.boundary_gate is not None
            else torch.zeros((w_cnt,), device=self.device, dtype=torch.float32)
        )
        for bi in range(w_cnt):
            if sel_new_sess[bi] or float(br_t[bi].item()) > float(self.cfg.boundary_flush_threshold):
                self._stage_latch.reset()
                self._cont_latch.reset()
            if float(br_t[bi].item()) > float(self.cfg.boundary_flush_threshold):
                self.session_start_frame = sel_stream_steps[bi]
                if not is_demo:
                    self.session_exec_start_frame = sel_stream_steps[bi]
                if self._conv_hist is not None:
                    self._conv_hist.zero_()

        k1, v1, g1, b1, _ = self.head.gdn.layer1.project_slots(
            gdn_slots.unsqueeze(0),
            vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
            grip_spike=kin["grip_spike"].unsqueeze(0),
            boundary_reset=br_t.unsqueeze(0),
        )
        raw_surp = self.head.gdn.layer1.compute_exact_write_surprise(
            k1, v1, g1, b1, num_write_slots=self.cfg.num_write_slots, initial_state_fp32=self._surp_l1_state
        ).squeeze(0)
        _, self._surp_l1_state, _ = run_dual_backend_gdn(
            k=k1, v=v1, g_fp32=g1, beta_fp32=b1, q=None, initial_state_fp32=self._surp_l1_state, track_occ=False
        )

        if (not is_demo) and len(self._w_is_demo) > 0 and self._w_is_demo[-1]:
            self._sal_ema = 0.0
            self._pvis_ema = 0.0
        beta = float(self.cfg.surprise_ema_beta)
        s_list = raw_surp.detach().float().cpu().tolist()
        g_list = kin["grip_spike"].detach().float().cpu().tolist()
        br_list = br_t.detach().float().cpu().tolist()
        cur_s = float(self._sal_ema)
        cur_p = float(self._pvis_ema)
        for wi in range(w_cnt):
            if br_list[wi] > 0.5:
                cur_s = 0.0
                cur_p = 0.0
            inst_p = float(s_list[wi])
            inst_s = inst_p + 0.5 * (1.0 if g_list[wi] > 0.25 else 0.0) + 0.35 * float(g_list[wi])
            cur_p = beta * cur_p + (1.0 - beta) * inst_p
            cur_s = beta * cur_s + (1.0 - beta) * inst_s

            self._w_steps.append(sel_stream_steps[wi])
            self._w_is_demo.append(bool(is_demo))
            self._w_front_hist.append(f_w[wi].detach())
            if (
                int(self.cfg.reveal_tokens_per_frame) > int(self.cfg.front_tokens_hist)
                or int(self.cfg.v2_pyramid_apex_hr16) > 0
            ):
                self._w_front_hist_256.append(f_in[wi].detach())
            self._w_gdn_slots.append(gdn_slots[wi])
            self._w_state_norm.append(s_w[wi].detach())
            self._w_vis_rms.append(float(kin["vis_motion_rms"][wi].item()))
            self._w_grip_spike.append(float(g_list[wi]))
            self._w_b_reset.append(float(br_list[wi]))
            self._w_salience.append(max(inst_s, cur_s))
            self._w_pure_vis_salience.append(max(inst_p, cur_p))
        self._sal_ema = cur_s
        self._pvis_ema = cur_p

    @torch.no_grad()
    def ingest_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
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
        sel = self.select_write_indices(t_in, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len)
        if not sel:
            return
        idx_t = torch.tensor(sel, device=front_tokens_256.device, dtype=torch.long)
        w_sel = wrist_tokens_256[idx_t.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
        self.ingest_selected_frames(
            front_tokens_256[idx_t],
            w_sel,
            np.asarray(raw_states, dtype=np.float32)[sel],
            np.asarray(norm_states, dtype=np.float32)[sel],
            task_goal=task_goal,
            is_demo=is_demo,
        )

    @torch.no_grad()
    def predict_raw_preds(
        self,
        *,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        task_goal: str,
    ) -> Dict[str, torch.Tensor]:
        vlm_dt = self.pi05.compute_dtype
        head_dt = self.head.gdn_to_vlm[1].weight.dtype  # type: ignore[index]
        q_stream_step = max(0, self.stream_total_frames - 1)
        cutoff_stream = q_stream_step + 1

        w_tot = len(self._w_steps)
        w_steps_t = (
            torch.tensor(self._w_steps, device=self.device, dtype=torch.long)
            if w_tot > 0
            else torch.empty((0,), device=self.device, dtype=torch.long)
        )
        u_end = int(torch.searchsorted(w_steps_t, cutoff_stream, right=False).item()) if w_tot > 0 else 0

        if u_end > self._committed_u:
            seg_slots = torch.stack(self._w_gdn_slots[self._committed_u : u_end], dim=0).unsqueeze(0)
            seg_kin = {
                "vis_motion_rms": torch.tensor(self._w_vis_rms[self._committed_u : u_end], device=self.device, dtype=torch.float32),
                "grip_spike": torch.tensor(self._w_grip_spike[self._committed_u : u_end], device=self.device, dtype=torch.float32),
            }
            seg_br = torch.tensor(self._w_b_reset[self._committed_u : u_end], device=self.device, dtype=torch.float32).unsqueeze(0)
            seg_out = self.head.gdn.forward_segment(
                seg_slots, seg_kin, seg_br, init_states=self._gdn_states, compute_surprise=False
            )
            self._gdn_states = (seg_out["S1"], seg_out["occ1"], seg_out["S2"], seg_out["occ2"])
            self._committed_u = u_end

        S1_p, o1_p, S2_p, o2_p = self._gdn_states
        has_hist = u_end > 0
        self._last_b_reset_max = max(self._w_b_reset[:u_end]) if u_end > 0 else 0.0

        phys_sc: Optional[torch.Tensor] = None
        adv_ev: Optional[torch.Tensor] = None
        if self.cfg.v2_mode:
            if has_hist:
                d_all = torch.tensor(self._w_is_demo[:u_end], device=self.device, dtype=torch.bool)
                br_all = torch.tensor(self._w_b_reset[:u_end], device=self.device, dtype=torch.float32)
                pvis_all = torch.tensor(self._w_pure_vis_salience[:u_end], device=self.device, dtype=torch.float32)
                vrms_all = torch.tensor(self._w_vis_rms[:u_end], device=self.device, dtype=torch.float32)
                grip_all = torch.tensor(self._w_grip_spike[:u_end], device=self.device, dtype=torch.float32)
                f_all = torch.stack(self._w_front_hist[:u_end], dim=0)
                if int(self.cfg.v2_pyramid_apex_hr16) > 0:
                    k_ap = int(self.cfg.v2_pyramid_apex_hr16)
                    k_sw = max(0, int(self.cfg.reveal_keyframes) - k_ap)
                    apex_sel_ev, swap_sel_ev, sp = select_v2_pyramid_reveal_indices(
                        w_steps_t[:u_end],
                        pvis_all,
                        vrms_all,
                        br_all,
                        cutoff_excl_frame=cutoff_stream,
                        k_apex=k_ap,
                        k_swap=k_sw,
                        uniform_anchors=self.cfg.v2_uniform_anchors,
                        flush_thresh=self.cfg.boundary_flush_threshold,
                        front_hist_w=f_all,
                        is_demo_w=d_all,
                        prominence_tau=self.cfg.v2_reveal_prominence_tau,
                        v3_median_ref=bool(self.cfg.v3_mode),
                    )
                    if self.head.coda.adv_on:
                        nbr_m = self.system._advect_nbr_mask(self.device)
                        p_c = int(f_all.shape[1])
                        n_have = len(self._w_flow)
                        if u_end - 1 > n_have:
                            new_w = compute_patch_flow_warps(
                                f_all[n_have : u_end - 1],
                                f_all[n_have + 1 : u_end],
                                beta=self.cfg.v2_advect_beta,
                                id_bias=self.cfg.v2_advect_id_bias,
                                nbr_mask=nbr_m,
                            )
                            self._w_flow.extend(list(new_w.unbind(0)))
                        f_cur64_ev = pool_spatial_tokens_to_square(
                            self.system.project_vis(front_tokens_256.to(self.device)), p_c
                        )
                        w_cur_ev = compute_patch_flow_warps(
                            f_all[u_end - 1 : u_end],
                            f_cur64_ev[:1],
                            beta=self.cfg.v2_advect_beta,
                            id_bias=self.cfg.v2_advect_id_bias,
                            nbr_mask=nbr_m,
                        )[0]
                        w_flow_t = (
                            torch.stack(self._w_flow[: u_end - 1], dim=0)
                            if u_end > 1
                            else torch.empty((0, p_c, p_c), device=self.device, dtype=torch.float32)
                        )
                        adv_ev = accumulate_apex_advection(
                            w_flow_t, apex_sel_ev.tolist(), u_end - 1, w_cur_ev, k_ap
                        ).unsqueeze(0)
                    f_ap_hr_ev: Optional[torch.Tensor] = None
                    f_prev_ap_hr_ev: Optional[torch.Tensor] = None
                    if len(self._w_front_hist_256) >= u_end and int(apex_sel_ev.numel()) > 0:
                        prev_ap_ev = torch.clamp(apex_sel_ev - 1, min=int(sp))
                        f_ap_hr_ev = torch.stack([self._w_front_hist_256[int(vi)] for vi in apex_sel_ev.tolist()], dim=0)
                        f_prev_ap_hr_ev = torch.stack([self._w_front_hist_256[int(pi)] for pi in prev_ap_ev.tolist()], dim=0)
                    rev_tok, rev_mask = self.head.encode_v2_pyramid_reveal_tokens(
                        f_all,
                        w_steps_t[:u_end],
                        d_all,
                        apex_sel_ev,
                        swap_sel_ev,
                        sp,
                        f_apex_hr=f_ap_hr_ev,
                        f_prev_apex_hr=f_prev_ap_hr_ev,
                    )
                else:
                    v_sel, sp = select_v2_hybrid_reveal_indices(
                        w_steps_t[:u_end],
                        pvis_all,
                        vrms_all,
                        br_all,
                        cutoff_excl_frame=cutoff_stream,
                        num_keyframes=self.cfg.reveal_keyframes,
                        uniform_anchors=self.cfg.v2_uniform_anchors,
                        flush_thresh=self.cfg.boundary_flush_threshold,
                        front_hist_w=f_all,
                        is_demo_w=d_all,
                        apex_peaks=self.cfg.v2_apex_peaks,
                    )
                    f_sel_hr_ev: Optional[torch.Tensor] = None
                    f_prev_hr_ev: Optional[torch.Tensor] = None
                    if (
                        int(self.cfg.reveal_tokens_per_frame) > int(self.cfg.front_tokens_hist)
                        and len(self._w_front_hist_256) >= u_end
                        and int(v_sel.numel()) > 0
                    ):
                        prev_sel_ev = torch.clamp(v_sel - 1, min=int(sp))
                        f_sel_hr_ev = torch.stack([self._w_front_hist_256[int(vi)] for vi in v_sel.tolist()], dim=0)
                        f_prev_hr_ev = torch.stack([self._w_front_hist_256[int(pi)] for pi in prev_sel_ev.tolist()], dim=0)
                    rev_tok, rev_mask = self.head.encode_v2_reveal_tokens(
                        f_all,
                        w_steps_t[:u_end],
                        d_all,
                        v_sel,
                        sp,
                        f_sel_hr=f_sel_hr_ev,
                        f_prev_hr=f_prev_hr_ev,
                    )
                phys_sc = self.head.compute_v2_phys_scalars(
                    d_all[sp:u_end], vrms_all[sp:u_end], grip_all[sp:u_end]
                ).unsqueeze(0)
            else:
                f_empty = torch.empty((0, self.cfg.front_tokens_hist, self.cfg.vis_dim), device=self.device, dtype=vlm_dt)
                w_empty = torch.empty((0,), device=self.device, dtype=torch.long)
                d_empty = torch.empty((0,), device=self.device, dtype=torch.bool)
                if int(self.cfg.v2_pyramid_apex_hr16) > 0:
                    rev_tok, rev_mask = self.head.encode_v2_pyramid_reveal_tokens(
                        f_empty, w_empty, d_empty, w_empty, w_empty, 0
                    )
                else:
                    rev_tok, rev_mask = self.head.encode_v2_reveal_tokens(f_empty, w_empty, d_empty, w_empty, 0)
                phys_sc = torch.zeros((1, 4), device=self.device, dtype=torch.float32)
        else:
            if has_hist:
                d_all = torch.tensor(self._w_is_demo[:u_end], device=self.device, dtype=torch.bool)
                br_all = torch.tensor(self._w_b_reset[:u_end], device=self.device, dtype=torch.float32)
                sal_all = torch.tensor(self._w_salience[:u_end], device=self.device, dtype=torch.float32)
                pvis_all = torch.tensor(self._w_pure_vis_salience[:u_end], device=self.device, dtype=torch.float32)
                v_sel = self.head.reveal_selector.select_vlm_surprise_indices(
                    w_steps_t[:u_end], d_all, sal_all, pvis_all, br_all, cutoff_excl_frame=cutoff_stream
                )
                f_vlm = (
                    torch.stack([self._w_front_hist[int(vi)] for vi in v_sel.tolist()], dim=0)
                    if int(v_sel.numel()) > 0
                    else torch.empty((0, self.cfg.front_tokens_hist, self.cfg.vis_dim), device=self.device, dtype=vlm_dt)
                )
            else:
                f_vlm = torch.empty((0, self.cfg.front_tokens_hist, self.cfg.vis_dim), device=self.device, dtype=vlm_dt)
            rev_tok, rev_mask = self.head.reveal_selector.encode_vlm_surprise_tokens(f_vlm)
        rev_tok_b = rev_tok.unsqueeze(0).to(vlm_dt)
        rev_mask_b = rev_mask.unsqueeze(0)

        lt, lm, l_pool = self._encode_prompt(task_goal)
        f_cur = self.system.project_vis(front_tokens_256.to(self.device)).to(vlm_dt)
        w_cur = self.system.project_vis(wrist_tokens_256.to(self.device)).to(vlm_dt)
        w16 = pool_spatial_tokens_to_square(w_cur, self.cfg.wrist_tokens_curr_pool)
        w16 = w16 + self.head.wrist_view_emb.to(device=self.device, dtype=vlm_dt)

        ctx_q = self.head.ctx_q_proj(
            torch.cat([l_pool.unsqueeze(1).to(head_dt), pool_spatial_tokens_to_square(f_cur, 16).to(head_dt)], dim=1)
        )
        gdn_tok, _ = self.head.gdn.readout(ctx_q, S1_p, o1_p, S2_p, o2_p)
        gdn_vlm = self.head.gdn_to_vlm(gdn_tok.to(head_dt)).to(vlm_dt)

        st_mme = self.normalizer.normalize_state(np.asarray(raw_state, dtype=np.float32))
        st_cur = torch.from_numpy(np.asarray(st_mme, dtype=np.float32)).unsqueeze(0).to(self.device)
        st_prev = st_cur
        if u_end > 0:
            target_prev = max(self.session_exec_start_frame, q_stream_step - int(self.cfg.write_stride_exec))
            p_pos = int(torch.searchsorted(w_steps_t[:u_end], target_prev, right=True).item()) - 1
            if p_pos >= 0 and (not self._w_is_demo[p_pos]) and self._w_steps[p_pos] >= self.session_exec_start_frame:
                st_prev = self._w_state_norm[p_pos].unsqueeze(0)
        st_vlm = self.head.build_state_contact_tokens(st_cur, st_prev, phys_scalars=phys_sc).to(vlm_dt)

        gs_tok_b = torch.cat([gdn_vlm, st_vlm], dim=1)
        gs_mask_b = torch.full((1, int(gs_tok_b.shape[1])), has_hist, device=self.device, dtype=torch.bool)
        gs_mask_b[:, : self.cfg.num_persistent_tokens] = True
        gs_mask_b[:, -2:] = True

        extra_embs = torch.cat([rev_tok_b, gs_tok_b], dim=1)
        extra_mask = torch.cat([rev_mask_b, gs_mask_b], dim=1)
        f_cur_vlm = self.head.encode_v2_front_coords(f_cur) if self.cfg.v2_mode else f_cur

        try:
            set_vlm_lora_mode(self.pi05, self.system.vlm_lora_mode)
            prefix_tokens, prefix_pad_masks, _ = self.pi05.embed_prefix(
                lt, lm, image_embs=[f_cur_vlm, w16], extra_prefix_embs=extra_embs, extra_prefix_mask=extra_mask
            )
            x0_out = self.system._run_vlm_prefix_backbone(prefix_tokens, prefix_pad_masks)
        finally:
            set_vlm_lora_mode(self.pi05, self.restore_lora_mode)

        n_f = int(f_cur_vlm.shape[1])
        n_w = int(w16.shape[1])
        n_l = int(lt.shape[1])
        n_rev = int(rev_tok_b.shape[1])
        n_gs = int(gs_tok_b.shape[1])

        front_vlm = x0_out[:, :n_f, :]
        lang_slice = x0_out[:, n_f + n_w : n_f + n_w + n_l, :]
        lm_f = lm.to(lang_slice.dtype).unsqueeze(-1)
        lang_pool = (lang_slice * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)
        rev_vlm = x0_out[:, n_f + n_w + n_l : n_f + n_w + n_l + n_rev, :] if self.cfg.v2_mode else None
        gdn_slice = x0_out[:, -n_gs:, :]
        gm_f = gs_mask_b.to(gdn_slice.dtype).unsqueeze(-1)
        gdn_pool = (gdn_slice * gm_f).sum(dim=1) / gm_f.sum(dim=1).clamp(min=1.0)

        preds = self.head.coda(
            front_vlm,
            lang_pool,
            gdn_pool,
            reveal_vlm=rev_vlm,
            reveal_raw=rev_tok_b if (self.cfg.v2_mode and self.cfg.v2_raw_siglip_reveal) else None,
            reveal_mask=rev_mask_b if self.cfg.v2_mode else None,
            reveal_advect=adv_ev,
        )
        if self.peak_window_radius > 0:
            preds = apply_peak_window_to_preds(preds, s_side=self.head.coda.front_side, radius=self.peak_window_radius)
        if self.fold_occlusion_twins and self._twin_pairs:
            preds = {
                **preds,
                "template_logits": fold_occlusion_twin_logits(preds["template_logits"], self._twin_pairs),
            }
        preds["front_vlm"] = front_vlm
        preds["lang_pool"] = lang_pool
        preds["gdn_pool"] = gdn_pool
        return preds

    @torch.no_grad()
    def predict_continuous_grounding(
        self,
        *,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        task_goal: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """`P2-M11` (`#395(2)`): Zero-ASCII continuous grounding (`max_token_len=64`!), `"base_policy"` on 10 base tasks (`0.0` diff!)."""
        if self.dynamic_policy_lora_on_subgoal and not should_route_to_symbolic_specialist(task_goal):
            target_mode: LoRAMode = (
                "reactive_policy"
                if (self.enable_reactive_lora_routing and should_route_to_reactive_precision(task_goal))
                else self.non_subgoal_lora_mode
            )
            set_vlm_lora_mode(self.pi05, target_mode)
            self._last_pred_subgoal = ""
            return None
        preds = self.predict_raw_preds(
            front_tokens_256=front_tokens_256,
            wrist_tokens_256=wrist_tokens_256,
            raw_state=raw_state,
            task_goal=task_goal,
        )
        pkt_1 = build_step_continuous_grounding_packet(preds, 0, self.head.template_vocab, self._cont_latch)
        # Expand leading batch dim `B = 1` for `ContinuousVLMGroundingBridge`
        pkt_b = {k: v.unsqueeze(0) for k, v in pkt_1.items()}
        if self.dynamic_policy_lora_on_subgoal:
            set_vlm_lora_mode(self.pi05, "policy")
        return pkt_b

    @torch.no_grad()
    def predict_subgoal(
        self,
        *,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        del norm_state, step_idx, exec_start_idx
        if self.dynamic_policy_lora_on_subgoal and not should_route_to_symbolic_specialist(task_goal):
            target_mode: LoRAMode = (
                "reactive_policy"
                if (self.enable_reactive_lora_routing and should_route_to_reactive_precision(task_goal))
                else self.non_subgoal_lora_mode
            )
            set_vlm_lora_mode(self.pi05, target_mode)
            self._last_pred_subgoal = ""
            return None
        if self.abstain_ordinal_counting and should_abstain_by_task_goal(
            task_goal,
            abstain_place_order=self.abstain_place_order,
            abstain_temporal_button=self.abstain_temporal_button,
        ):
            if self.dynamic_policy_lora_on_subgoal:
                set_vlm_lora_mode(self.pi05, self.non_subgoal_lora_mode)
            self._last_pred_subgoal = ""
            return None

        preds = self.predict_raw_preds(
            front_tokens_256=front_tokens_256,
            wrist_tokens_256=wrist_tokens_256,
            raw_state=raw_state,
            task_goal=task_goal,
        )
        if self.min_template_confidence > 0.0:
            probs = F.softmax(preds["template_logits"][0].float(), dim=-1)
            max_prob = float(compute_twin_summed_max_prob(probs, self._twin_pairs).item())
            if max_prob < self.min_template_confidence:
                if self.dynamic_policy_lora_on_subgoal:
                    set_vlm_lora_mode(self.pi05, self.non_subgoal_lora_mode)
                self._last_pred_subgoal = ""
                return None

        tmpl_idx = int(preds["template_logits"][0].argmax(dim=-1).item())
        tmpl_str = self.head.template_vocab[tmpl_idx] if 0 <= tmpl_idx < len(self.head.template_vocab) else ""
        xy1_use, xy2_use = preds["xy_pri"][0], preds["xy_sec"][0]
        if self.enable_corm and tmpl_str:
            tmpl_str = self.corm.guard_stage_template(tmpl_str, self._stage_latch.canon_stage, raw_state, task_goal)
            cand_grounded = self._twin_to_grounded.get(tmpl_str, tmpl_str)
            corm_yx = self.corm.query_target_yx(cand_grounded, task_goal)
            if corm_yx is not None:
                tmpl_str = cand_grounded
                xy1_use, xy2_use = self.corm.refine_norm_xy(tmpl_str, task_goal, xy1_use, xy2_use)
                self._stage_latch.canon_stage = cand_grounded
                self._stage_latch.latched_subgoal = render_grounded_subgoal_from_norm_xy(tmpl_str, xy1_use, xy2_use)
                self._stage_latch.has_bbox = True
                pred_sg = self._stage_latch.latched_subgoal
            elif self.stage_entry_latch:
                pred_sg = self._stage_latch.step(tmpl_str, xy1_use, xy2_use)
            else:
                pred_sg = render_grounded_subgoal_from_norm_xy(tmpl_str, xy1_use, xy2_use)
        elif self.stage_entry_latch:
            pred_sg = self._stage_latch.step(tmpl_str, xy1_use, xy2_use)
        else:
            pred_sg = render_grounded_subgoal_from_norm_xy(tmpl_str, xy1_use, xy2_use)
        self._last_pred_subgoal = pred_sg
        out_sg = pred_sg if pred_sg else None
        if self.dynamic_policy_lora_on_subgoal:
            set_vlm_lora_mode(self.pi05, "policy" if out_sg is not None else self.non_subgoal_lora_mode)
        return out_sg

    def to_device(self, dev: torch.device) -> None:
        self.device = dev
        self.head.to(dev)
        self._prompt_cache.clear()
        self._gdn_states = tuple(t.to(dev) for t in self._gdn_states)  # type: ignore[assignment]
        self._surp_l1_state = self._surp_l1_state.to(dev)
        if self._prev_w_front is not None:
            self._prev_w_front = self._prev_w_front.to(dev)
        if self._prev_w_state is not None:
            self._prev_w_state = self._prev_w_state.to(dev)
        if self._prev_w_demo is not None:
            self._prev_w_demo = self._prev_w_demo.to(dev)
        if self._conv_hist is not None:
            self._conv_hist = self._conv_hist.to(dev)
        self._w_front_hist = [t.to(dev) for t in self._w_front_hist]
        self._w_front_hist_256 = [t.to(dev) for t in self._w_front_hist_256]
        self._w_flow = [t.to(dev) for t in getattr(self, "_w_flow", [])]
        self._w_gdn_slots = [t.to(dev) for t in self._w_gdn_slots]
        self._w_state_norm = [t.to(dev) for t in self._w_state_norm]
        self._cont_latch.to_device(dev)
        if hasattr(self, "corm"):
            self.corm.to_device(dev)

    @torch.no_grad()
    def read_memory(
        self,
        *,
        prefix_hidden: torch.Tensor | None = None,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> Tuple[None, None, Dict[str, Any]]:
        del prefix_hidden, front_tokens_256, wrist_tokens_256, raw_state, norm_state, task_goal, step_idx, exec_start_idx
        return None, None, {
            "loop_depth": 1,
            "mem_active_tokens": 0,
            "pred_subgoal": self._last_pred_subgoal,
            "boundary_reset_max": self._last_b_reset_max,
        }


class DecoupledStagePointerVLMAdapter:
    """`#376` Action A: Zero-Retraining Decoupled `v1` Stage (`98.8%` transitions, `0` timeouts) + `v2b` Neutral-Arm `<y, x>` Pointer (`0 MiB` extra VLM!)."""

    def __init__(
        self,
        stage_adapter: DeltaGroundVLMEvalAdapter,
        pointer_adapter: DeltaGroundVLMEvalAdapter,
    ) -> None:
        self.stage_adapter = stage_adapter
        self.pointer_adapter = pointer_adapter
        self.device = stage_adapter.device
        self._twin_to_grounded = dict(stage_adapter._twin_to_grounded)
        self._stage_latch = StageEntrySubgoalLatch(self._twin_to_grounded)
        self._last_pred_subgoal: str = ""
        self._last_b_reset_max: float = 0.0
        self._pending_sub_idx: Optional[Tuple[List[int], List[int]]] = None

    def to_device(self, dev: torch.device) -> None:
        self.device = dev
        self.stage_adapter.to_device(dev)
        self.pointer_adapter.to_device(dev)

    def reset(self) -> None:
        self.stage_adapter.reset()
        self.pointer_adapter.reset()
        self._stage_latch.reset()
        self._last_pred_subgoal = ""
        self._last_b_reset_max = 0.0
        self._pending_sub_idx = None

    @property
    def wants_raw_rgb(self) -> bool:
        return bool(getattr(self.stage_adapter, "wants_raw_rgb", False))

    @torch.no_grad()
    def ingest_raw_rgb_stream(
        self,
        front_rgb_seq: Sequence[np.ndarray] | np.ndarray,
        raw_states_seq: Sequence[np.ndarray] | np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        if hasattr(self.stage_adapter, "ingest_raw_rgb_stream"):
            self.stage_adapter.ingest_raw_rgb_stream(
                front_rgb_seq, raw_states_seq, task_goal=task_goal, is_demo=is_demo
            )

    @torch.no_grad()
    def select_write_indices(
        self,
        num_frames: int,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> List[int]:
        prev_sess_s = self.stage_adapter.session_start_frame
        prev_exec_s = self.stage_adapter.session_exec_start_frame
        sel_s = self.stage_adapter.select_write_indices(
            num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        sel_p = self.pointer_adapter.select_write_indices(
            num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        if (
            self.stage_adapter.session_start_frame != prev_sess_s
            or self.stage_adapter.session_exec_start_frame != prev_exec_s
        ):
            self._stage_latch.reset()
        union = sorted(set(sel_s) | set(sel_p))
        pos = {val: i for i, val in enumerate(union)}
        self._pending_sub_idx = ([pos[v] for v in sel_s], [pos[v] for v in sel_p]) if union else None
        return union

    @torch.no_grad()
    def ingest_selected_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        sub = self._pending_sub_idx
        self._pending_sub_idx = None
        prev_sess_s = self.stage_adapter.session_start_frame
        prev_exec_s = self.stage_adapter.session_exec_start_frame
        prev_w_len = len(self.stage_adapter._w_b_reset)
        rs_np = np.asarray(raw_states, dtype=np.float32)
        ns_np = np.asarray(norm_states, dtype=np.float32)

        if sub is None:
            self.stage_adapter.ingest_selected_frames(
                front_tokens_256, wrist_tokens_256, rs_np, ns_np, task_goal=task_goal, is_demo=is_demo
            )
            self.pointer_adapter.ingest_selected_frames(
                front_tokens_256, wrist_tokens_256, rs_np, ns_np, task_goal=task_goal, is_demo=is_demo
            )
        else:
            idx_s, idx_p = sub
            if idx_s:
                ts_s = torch.tensor(idx_s, device=front_tokens_256.device, dtype=torch.long)
                ws_s = wrist_tokens_256[ts_s.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
                self.stage_adapter.ingest_selected_frames(
                    front_tokens_256[ts_s], ws_s, rs_np[idx_s], ns_np[idx_s], task_goal=task_goal, is_demo=is_demo
                )
            if idx_p:
                ts_p = torch.tensor(idx_p, device=front_tokens_256.device, dtype=torch.long)
                ws_p = wrist_tokens_256[ts_p.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
                self.pointer_adapter.ingest_selected_frames(
                    front_tokens_256[ts_p], ws_p, rs_np[idx_p], ns_np[idx_p], task_goal=task_goal, is_demo=is_demo
                )

        if (
            self.stage_adapter.session_start_frame != prev_sess_s
            or self.stage_adapter.session_exec_start_frame != prev_exec_s
            or any(r > float(self.stage_adapter.cfg.boundary_flush_threshold) for r in self.stage_adapter._w_b_reset[prev_w_len:])
        ):
            self._stage_latch.reset()

    @torch.no_grad()
    def ingest_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
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
        sel = self.select_write_indices(t_in, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len)
        if not sel:
            return
        idx_t = torch.tensor(sel, device=front_tokens_256.device, dtype=torch.long)
        w_sel = wrist_tokens_256[idx_t.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
        self.ingest_selected_frames(
            front_tokens_256[idx_t],
            w_sel,
            np.asarray(raw_states, dtype=np.float32)[sel],
            np.asarray(norm_states, dtype=np.float32)[sel],
            task_goal=task_goal,
            is_demo=is_demo,
        )

    @torch.no_grad()
    def predict_subgoal(
        self,
        *,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        del norm_state, step_idx, exec_start_idx
        st_preds = self.stage_adapter.predict_raw_preds(
            front_tokens_256=front_tokens_256,
            wrist_tokens_256=wrist_tokens_256,
            raw_state=raw_state,
            task_goal=task_goal,
        )
        self._last_b_reset_max = float(self.stage_adapter._last_b_reset_max)
        vocab_s = self.stage_adapter.head.template_vocab
        t_idx = int(st_preds["template_logits"][0].argmax(dim=-1).item())
        tmpl_str = vocab_s[t_idx] if 0 <= t_idx < len(vocab_s) else ""
        if not tmpl_str:
            self._stage_latch.reset()
            self._last_pred_subgoal = ""
            return None

        if self.stage_adapter.enable_corm:
            tmpl_str = self.stage_adapter.corm.guard_stage_template(
                tmpl_str, self._stage_latch.canon_stage, raw_state, task_goal
            )
            cand_grounded = self._twin_to_grounded.get(tmpl_str, tmpl_str)
            corm_yx = self.stage_adapter.corm.query_target_yx(cand_grounded, task_goal)
            if corm_yx is not None:
                tmpl_str = cand_grounded
                xy1, xy2 = self.stage_adapter.corm.refine_norm_xy(
                    tmpl_str, task_goal, st_preds["xy_pri"][0], st_preds["xy_sec"][0]
                )
                self._stage_latch.canon_stage = cand_grounded
                self._stage_latch.latched_subgoal = render_grounded_subgoal_from_norm_xy(tmpl_str, xy1, xy2)
                self._stage_latch.has_bbox = True
                self._last_pred_subgoal = self._stage_latch.latched_subgoal
                return self._last_pred_subgoal

        canon = self._twin_to_grounded.get(tmpl_str, tmpl_str)
        need_fresh_xy = ("<bbox>" in tmpl_str) and (canon != self._stage_latch.canon_stage or not self._stage_latch.has_bbox)
        if need_fresh_xy:
            pt_preds = self.pointer_adapter.predict_raw_preds(
                front_tokens_256=front_tokens_256,
                wrist_tokens_256=wrist_tokens_256,
                raw_state=raw_state,
                task_goal=task_goal,
            )
            xy1, xy2 = pt_preds["xy_pri"][0], pt_preds["xy_sec"][0]
        else:
            xy1, xy2 = st_preds["xy_pri"][0], st_preds["xy_sec"][0]

        pred_sg = self._stage_latch.step(tmpl_str, xy1, xy2)
        self._last_pred_subgoal = pred_sg
        return pred_sg if pred_sg else None

    @torch.no_grad()
    def read_memory(
        self,
        *,
        prefix_hidden: torch.Tensor | None = None,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> Tuple[None, None, Dict[str, Any]]:
        del prefix_hidden, front_tokens_256, wrist_tokens_256, raw_state, norm_state, task_goal, step_idx, exec_start_idx
        return None, None, {
            "loop_depth": 1,
            "mem_active_tokens": 0,
            "pred_subgoal": self._last_pred_subgoal,
            "boundary_reset_max": self._last_b_reset_max,
        }


@torch.no_grad()
def precompute_deltaground_vlm_subgoals(
    deltaground_vlm_ckpt: str | Path,
    pi05_ckpt_dir: str | Path,
    cache_dir: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    tasks: Sequence[str] | None = None,
    device: str | torch.device = "cuda",
    peak_window_radius: int = 2,
    min_template_confidence: float = 0.60,
    abstain_ordinal_counting: bool = True,
    abstain_place_order: bool = True,
    abstain_temporal_button: bool = True,
    chunk_batch_size: int = 32,
    force_pytorch_gdn: bool = False,
) -> Tuple[Dict[str, List[Dict[int, str]]], Dict[str, Any]]:
    """Precompute `P2-M9: DeltaGround-VLM` subgoals with `#342`/`#343`/`#345` train=eval selective abstention (`< 16 GB` RSS)."""
    raw = torch.load(Path(deltaground_vlm_ckpt), map_location="cpu", weights_only=False)
    payload: Dict[str, Any] = raw.get("ema_deltaground_vlm") or raw.get("deltaground_vlm") or raw

    if output_path is not None and Path(output_path).exists():
        cached = torch.load(Path(output_path), map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and "pred_subgoals_by_task" in cached:
            return cached["pred_subgoals_by_task"], payload

    dev = torch.device(device)
    cache_root = Path(cache_dir)
    prec = "bfloat16" if dev.type == "cuda" else "float32"
    pi05_base, _ = load_pi05_checkpoint(pi05_ckpt_dir, device=dev, precision=prec)
    cfg = DeltaGroundVLMConfig.from_dict(payload["deltaground_vlm_config"])
    tmpl_vocab: List[str] = list(payload["template_vocab"])
    twin_pairs = build_occlusion_twin_template_pairs(tmpl_vocab)
    twin_to_grounded = {tmpl_vocab[u_i]: tmpl_vocab[g_i] for u_i, g_i in twin_pairs}
    system = DeltaGroundVLMSystem(cfg, pi05_base, template_vocab=tmpl_vocab, enable_grad_ckpt=False)
    system.load_payload(payload, strict=True)
    system.eval()

    normalizer = ActionStateNormalizer(mode="quantile_delta", stats=payload.get("state_stats") or MME_VLA_NORM_STATS)
    target_tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
    pred_by_task: Dict[str, List[Dict[int, str]]] = {}

    for t_idx, task in enumerate(target_tasks):
        sh = load_pi05_shards(cache_root, tasks=(task,), mmap=True)[task]
        n_eps = int(sh["epis_idx"].shape[0])
        prompts_list: List[str] = [str(p) for p in sh["prompts"]]
        ep_offs = sh["ep_offsets"].to(torch.long).numpy()
        ex_offs = sh["exec_ep_offsets"].to(torch.long).numpy()
        ex_starts = sh["exec_start_idx"].to(torch.long).numpy()
        rel_all = sh["exec_rel_step_idx"].to(torch.long)
        raw_st_all = sh["state"].to(torch.float32).contiguous().clone().numpy()
        st_norm_all = torch.from_numpy(normalizer.normalize_state(raw_st_all))
        is_demo_all = sh["is_demo"].to(torch.bool).contiguous().clone()
        lt_all = sh["lang_tokens"].to(torch.long).contiguous().clone()
        lm_all = sh["lang_masks"].to(torch.bool).contiguous().clone()

        task_list: List[Dict[int, str]] = []
        n_active_sg = 0
        n_tot_sg = 0
        for ep_i in range(n_eps):
            s = int(ep_offs[ep_i])
            e = int(ep_offs[ep_i + 1])
            es = int(ex_offs[ep_i])
            ee = int(ex_offs[ep_i + 1])
            q_all = rel_all[es:ee]
            m_tot = int(q_all.numel())
            if m_tot == 0:
                task_list.append({})
                continue
            ep_goal = prompts_list[ep_i] if ep_i < len(prompts_list) else ""
            if abstain_ordinal_counting and should_abstain_by_task_goal(
                ep_goal,
                abstain_place_order=abstain_place_order,
                abstain_temporal_button=abstain_temporal_button,
            ):
                n_tot_sg += m_tot
                task_list.append({int(q_st): "" for q_st in q_all.tolist()})
                continue

            ep_f = sh["front_emb"][s:e].to(torch.bfloat16).contiguous()
            ep_w = sh["wrist_emb"][s:e].to(torch.bfloat16).contiguous()
            ep_st = st_norm_all[s:e]
            ep_dm = is_demo_all[s:e]
            ep_map: Dict[int, str] = {}
            ep_latch = StageEntrySubgoalLatch(twin_to_grounded) if cfg.v2_mode else None

            # Chunk `m_tot` query steps in slices of `chunk_batch_size` (`32`) so `T=1000` episodes never spike VRAM!
            for c0 in range(0, m_tot, max(1, int(chunk_batch_size))):
                c1 = min(m_tot, c0 + max(1, int(chunk_batch_size)))
                q_sub = q_all[c0:c1]
                ep_in = {
                    "exec_start_idx": int(ex_starts[ep_i]),
                    "lang_tokens": lt_all[ep_i],
                    "lang_masks": lm_all[ep_i],
                    "front_emb": ep_f,
                    "wrist_emb": ep_w,
                    "state_norm": ep_st,
                    "is_demo": ep_dm,
                    "chunk_step_indices": q_sub,
                    "exec_front_256": sh["exec_front_256"][es + c0 : es + c1].to(torch.bfloat16).contiguous(),
                    "exec_wrist_256": sh["exec_wrist_256"][es + c0 : es + c1].to(torch.bfloat16).contiguous(),
                }
                preds, _ = system.forward_episodes(
                    [ep_in], force_pytorch_gdn=force_pytorch_gdn or (dev.type != "cuda")
                )
                fin = apply_peak_window_to_preds(preds, s_side=system.head.coda.front_side, radius=peak_window_radius)
                t_logits = (
                    fold_occlusion_twin_logits(fin["template_logits"], twin_pairs)
                    if (cfg.v2_mode and twin_pairs)
                    else fin["template_logits"]
                )
                probs = F.softmax(fin["template_logits"].float(), dim=-1)
                max_probs = compute_twin_summed_max_prob(probs, twin_pairs)
                t_ids = t_logits.argmax(dim=-1)
                max_p_cpu = max_probs.detach().cpu().tolist()
                t_id_cpu = t_ids.detach().cpu().tolist()
                xy1_cpu = fin["xy_pri"].detach().cpu()
                xy2_cpu = fin["xy_sec"].detach().cpu()
                for qi, q_st in enumerate(q_sub.tolist()):
                    n_tot_sg += 1
                    tid = int(t_id_cpu[qi])
                    t_s = tmpl_vocab[tid] if 0 <= tid < len(tmpl_vocab) else ""
                    if min_template_confidence > 0.0 and float(max_p_cpu[qi]) < float(min_template_confidence):
                        ep_map[int(q_st)] = ""
                    else:
                        sg_s = (
                            ep_latch.step(t_s, xy1_cpu[qi], xy2_cpu[qi])
                            if ep_latch is not None
                            else render_grounded_subgoal_from_norm_xy(t_s, xy1_cpu[qi], xy2_cpu[qi])
                        )
                        ep_map[int(q_st)] = sg_s
                        if sg_s:
                            n_active_sg += 1
            task_list.append(ep_map)

        pred_by_task[task] = task_list
        del sh, raw_st_all, st_norm_all, is_demo_all, lt_all, lm_all, rel_all
        gc.collect()
        _drop_shard_page_cache(cache_root, task)
        act_pct = 100.0 * n_active_sg / max(1, n_tot_sg)
        print(
            f"[DG-VLM Precompute] {t_idx + 1}/{len(target_tasks)} {task}: {n_eps} eps (active_subgoal={act_pct:.1f}%)",
            flush=True,
        )

    del system, pi05_base
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(".pt.tmp")
        torch.save({"pred_subgoals_by_task": pred_by_task}, tmp_p)
        tmp_p.replace(out_p)

    return pred_by_task, payload


@torch.no_grad()
def extract_specialist_hr256_cache(
    dataset_root: str | Path,
    base_cache_dir: str | Path,
    hr_output_dir: str | Path,
    *,
    siglip_pkl: str | Path = "/work/gj26/b20098/checkpoints/pi05_init/pi05_vision_encoder/siglip_params.pkl",
    tasks: Sequence[str] = (
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
        "PickHighlight",
        "VideoRepick",
    ),
    device: str = "cuda",
    vision_batch_size: int = 512,
    num_workers: int = 16,
) -> None:
    """`#395(1)` Step 1a: `< 30s` GH200 extractor writing unpooled `256`-token (`16x16`) `bfloat16` front features (`< 12 GB` RAM)."""
    from concurrent.futures import ThreadPoolExecutor
    import io
    import json
    import time
    from PIL import Image
    import pyarrow.parquet as pq
    from pi05_data.vision_tower import Pi05SiglipVisionTower, preprocess_pi05_rgb
    from robomme_data.constants import TASK_TO_LEROBOT_BLOCK

    ds_root = Path(dataset_root)
    cache_root = Path(base_cache_dir)
    out_root = Path(hr_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_dir = ds_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])

    dev = torch.device(device)
    tower = Pi05SiglipVisionTower(compute_dtype=torch.bfloat16).to(dev).eval()
    tower.load_from_siglip_pickle(Path(siglip_pkl))
    tower.requires_grad_(False)

    def _load_selected_front_rgb(args_tup: Tuple[str, List[int]]) -> np.ndarray:
        pq_path, sel_rows = args_tup
        tbl = pq.read_table(pq_path, columns=["image"])
        col = tbl["image"].to_pylist()
        imgs: List[np.ndarray] = []
        for ri in sel_rows:
            raw = col[ri]["bytes"] if isinstance(col[ri], dict) else col[ri]
            with Image.open(io.BytesIO(raw)) as im:
                imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        return np.stack(imgs, axis=0)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for t_i, task in enumerate(tasks):
            out_file = out_root / f"{task}.pt"
            if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                print(f"[HR256 Cache] {t_i + 1}/{len(tasks)} {task}: already exists ({out_file.stat().st_size / (1024**3):.2f} GiB)", flush=True)
                continue

            t_task = time.perf_counter()
            sh = load_pi05_shards(cache_root, tasks=(task,), mmap=True)[task]
            ep_offs = sh["ep_offsets"].to(torch.long).numpy()
            ex_starts = sh["exec_start_idx"].to(torch.long).numpy()
            n_eps = int(sh["epis_idx"].shape[0])
            blk = TASK_TO_LEROBOT_BLOCK[task]

            sel_rows_per_ep: List[List[int]] = []
            w_offs = [0]
            for ep_l in range(n_eps):
                tot_f = int(ep_offs[ep_l + 1] - ep_offs[ep_l])
                demo_end = min(max(0, int(ex_starts[ep_l])), tot_f)
                # Store 100% of demo frames + first 8 frames + stride-2 exec frames -> 100% cache hit!
                idx_s = set(range(0, demo_end)) | set(range(0, min(tot_f, 8))) | set(range(demo_end, tot_f, 2))
                s_list = sorted(idx_s)
                sel_rows_per_ep.append(s_list)
                w_offs.append(w_offs[-1] + len(s_list))

            tot_w = w_offs[-1]
            wf_256 = torch.empty((tot_w, 256, 1152), dtype=torch.bfloat16)
            w_rel = torch.empty((tot_w,), dtype=torch.int32)

            work_args: List[Tuple[str, List[int]]] = []
            for ep_l in range(n_eps):
                g_ep = blk * 100 + ep_l
                pq_p = str(ds_root / data_tpl.format(episode_chunk=g_ep // chunks_size, episode_index=g_ep))
                work_args.append((pq_p, sel_rows_per_ep[ep_l]))

            rgb_list = list(pool.map(_load_selected_front_rgb, work_args))
            rgb_cat = np.concatenate(rgb_list, axis=0)
            del rgb_list

            for ep_l, s_l in enumerate(sel_rows_per_ep):
                w_rel[w_offs[ep_l] : w_offs[ep_l + 1]] = torch.tensor(s_l, dtype=torch.int32)

            bs = max(1, int(vision_batch_size))
            for c0 in range(0, tot_w, bs):
                c1 = min(tot_w, c0 + bs)
                px = preprocess_pi05_rgb(rgb_cat[c0:c1], device=str(dev))
                enc = tower(px, return_pre_head=True)
                wf_256[c0:c1].copy_(enc.cpu())

            del rgb_cat, sh
            tmp_f = out_file.with_suffix(".pt.tmp")
            torch.save(
                {
                    "write_front_256": wf_256,
                    "write_ep_offsets": torch.tensor(w_offs, dtype=torch.long),
                    "write_rel_idx": w_rel,
                },
                tmp_f,
            )
            tmp_f.replace(out_file)
            del wf_256, w_rel
            gc.collect()
            _drop_shard_page_cache(cache_root, task)
            _drop_shard_page_cache(out_root, task)
            print(
                f"[HR256 Cache] {t_i + 1}/{len(tasks)} {task}: {tot_w} frames in {time.perf_counter() - t_task:.1f}s "
                f"({out_file.stat().st_size / (1024**3):.2f} GiB)",
                flush=True,
            )

    del tower
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[HR256 Cache] Done all {len(tasks)} specialist tasks in {time.perf_counter() - t0:.1f}s", flush=True)


@torch.no_grad()
def precompute_continuous_vlm_grounding(
    deltaground_vlm_ckpt: str | Path,
    pi05_ckpt_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    hr_cache_dir: Optional[str | Path] = None,
    tasks: Sequence[str] = (
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
        "PickHighlight",
        "VideoRepick",
    ),
    device: str | torch.device = "cuda",
    peak_window_radius: int = 2,
    chunk_batch_size: int = 24,
    force_pytorch_gdn: bool = False,
) -> Dict[str, Any]:
    """`#395(2)` Precompute `P2-M11` continuous VLM grounding shards (`100%` train==eval `ContinuousStageEntryLatch`, `< 12 GB` RAM)."""
    raw = torch.load(Path(deltaground_vlm_ckpt), map_location="cpu", weights_only=False)
    payload: Dict[str, Any] = raw.get("ema_deltaground_vlm") or raw.get("deltaground_vlm") or raw

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if all((out_root / f"{t}.pt").exists() for t in tasks):
        print(f"[P2-M11 Precompute] All {len(tasks)} continuous grounding shards already exist at {out_root}", flush=True)
        return payload

    dev = torch.device(device)
    cache_root = Path(cache_dir)
    hr_root = Path(hr_cache_dir) if hr_cache_dir is not None else None
    prec = "bfloat16" if dev.type == "cuda" else "float32"
    pi05_base, _ = load_pi05_checkpoint(pi05_ckpt_dir, device=dev, precision=prec)
    cfg = DeltaGroundVLMConfig.from_dict(payload["deltaground_vlm_config"])
    tmpl_vocab: List[str] = list(payload["template_vocab"])
    twin_pairs = build_occlusion_twin_template_pairs(tmpl_vocab)
    twin_to_grounded = {tmpl_vocab[u_i]: tmpl_vocab[g_i] for u_i, g_i in twin_pairs}
    system = DeltaGroundVLMSystem(cfg, pi05_base, template_vocab=tmpl_vocab, enable_grad_ckpt=False)
    system.load_payload(payload, strict=True)
    system.eval()

    normalizer = ActionStateNormalizer(mode="quantile_delta", stats=payload.get("state_stats") or MME_VLA_NORM_STATS)

    for t_idx, task in enumerate(tasks):
        out_f = out_root / f"{task}.pt"
        if out_f.exists():
            continue
        sh = load_pi05_shards(cache_root, tasks=(task,), mmap=True)[task]
        hr_sh = (
            torch.load(hr_root / f"{task}.pt", map_location="cpu", mmap=True, weights_only=False)
            if (hr_root is not None and (hr_root / f"{task}.pt").exists())
            else None
        )
        n_eps = int(sh["epis_idx"].shape[0])
        ep_offs = sh["ep_offsets"].to(torch.long).numpy()
        ex_offs = sh["exec_ep_offsets"].to(torch.long).numpy()
        ex_starts = sh["exec_start_idx"].to(torch.long).numpy()
        rel_all = sh["exec_rel_step_idx"].to(torch.long)
        tot_exec = int(ex_offs[-1])
        raw_st_all = sh["state"].to(torch.float32).contiguous().clone().numpy()
        st_norm_all = torch.from_numpy(normalizer.normalize_state(raw_st_all))
        is_demo_all = sh["is_demo"].to(torch.bool).contiguous().clone()
        lt_all = sh["lang_tokens"].to(torch.long).contiguous().clone()
        lm_all = sh["lang_masks"].to(torch.bool).contiguous().clone()

        v_dim = int(cfg.vlm_width)
        f8_out = torch.zeros((tot_exec, 64, v_dim), dtype=torch.bfloat16)
        hm_out = torch.zeros((tot_exec, 4, 256), dtype=torch.bfloat16)
        roi_out = torch.zeros((tot_exec, 7, v_dim), dtype=torch.bfloat16)
        sem_out = torch.zeros((tot_exec, 48), dtype=torch.bfloat16)
        xy_out = torch.zeros((tot_exec, 8), dtype=torch.float32)

        for ep_i in range(n_eps):
            s = int(ep_offs[ep_i])
            e = int(ep_offs[ep_i + 1])
            es = int(ex_offs[ep_i])
            ee = int(ex_offs[ep_i + 1])
            q_all = rel_all[es:ee]
            m_tot = int(q_all.numel())
            if m_tot == 0:
                continue

            ep_f = sh["front_emb"][s:e].to(torch.bfloat16).contiguous()
            ep_w = sh["wrist_emb"][s:e].to(torch.bfloat16).contiguous()
            ep_st = st_norm_all[s:e]
            ep_dm = is_demo_all[s:e]
            hr_extra: Dict[str, Any] = {}
            if hr_sh is not None:
                ws = int(hr_sh["write_ep_offsets"][ep_i])
                we = int(hr_sh["write_ep_offsets"][ep_i + 1])
                hr_extra = {
                    "target_write_front_256": hr_sh["write_front_256"][ws:we].to(torch.bfloat16).contiguous(),
                    "target_write_abs_idx": hr_sh["write_rel_idx"][ws:we].to(torch.long).contiguous(),
                }

            ep_cont_latch = ContinuousStageEntryLatch(twin_to_grounded)
            for c0 in range(0, m_tot, max(1, int(chunk_batch_size))):
                c1 = min(m_tot, c0 + max(1, int(chunk_batch_size)))
                q_sub = q_all[c0:c1]
                ep_in = {
                    "exec_start_idx": int(ex_starts[ep_i]),
                    "lang_tokens": lt_all[ep_i],
                    "lang_masks": lm_all[ep_i],
                    "front_emb": ep_f,
                    "wrist_emb": ep_w,
                    "state_norm": ep_st,
                    "is_demo": ep_dm,
                    "chunk_step_indices": q_sub,
                    "exec_front_256": sh["exec_front_256"][es + c0 : es + c1].to(torch.bfloat16).contiguous(),
                    "exec_wrist_256": sh["exec_wrist_256"][es + c0 : es + c1].to(torch.bfloat16).contiguous(),
                    **hr_extra,
                }
                preds, _ = system.forward_episodes(
                    [ep_in], force_pytorch_gdn=force_pytorch_gdn or (dev.type != "cuda")
                )
                fin = apply_peak_window_to_preds(preds, s_side=system.head.coda.front_side, radius=peak_window_radius)
                if cfg.v2_mode and twin_pairs:
                    fin = {
                        **fin,
                        "template_logits": fold_occlusion_twin_logits(fin["template_logits"], twin_pairs),
                    }
                for qi in range(int(q_sub.numel())):
                    pkt = build_step_continuous_grounding_packet(fin, qi, tmpl_vocab, ep_cont_latch)
                    row = es + c0 + qi
                    f8_out[row].copy_(pkt["front_vlm_8x8"].to(torch.bfloat16).cpu())
                    hm_out[row].copy_(pkt["hm_maps"].to(torch.bfloat16).cpu())
                    roi_out[row].copy_(pkt["roi_vecs"].to(torch.bfloat16).cpu())
                    sem_out[row].copy_(pkt["sem_probs"].to(torch.bfloat16).cpu())
                    xy_out[row].copy_(pkt["xy_coords"].to(torch.float32).cpu())

        tmp_f = out_f.with_suffix(".pt.tmp")
        torch.save(
            {
                "front_vlm_8x8": f8_out,
                "hm_maps": hm_out,
                "roi_vecs": roi_out,
                "sem_probs": sem_out,
                "xy_coords": xy_out,
            },
            tmp_f,
        )
        tmp_f.replace(out_f)
        del sh, hr_sh, raw_st_all, st_norm_all, is_demo_all, lt_all, lm_all, rel_all
        del f8_out, hm_out, roi_out, sem_out, xy_out
        gc.collect()
        _drop_shard_page_cache(cache_root, task)
        if hr_root is not None:
            _drop_shard_page_cache(hr_root, task)
        _drop_shard_page_cache(out_root, task)
        print(
            f"[P2-M11 Precompute] {t_idx + 1}/{len(tasks)} {task}: {tot_exec} exec steps ({out_f.stat().st_size / (1024**3):.2f} GiB)",
            flush=True,
        )

    del system, pi05_base
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return payload
