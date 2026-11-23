"""PyTorch `Pi05Model` with 18-Layer `ActionExpert` Memory Modulation, Prefix KV Caching, LoRA, & Mem-CFG.

Exact PyTorch port of `mme_vla_suite/models/integration/history_pi0.py` (`HistoryPi0` with `pi05=True`)
and `openpi/models/pi0.py`.

Key APIs exposed for `@pi05-trainer` (`src/pi05_mem/`) and `@pi05-bridge` (`scripts/eval_pi05.py`):
- `encode_images(images)` -> `[B, 256, 2048]` SigLIP patch embeddings
- `prefill_prefix_kv(lang_tokens, lang_masks, *, images=..., image_embs=..., image_masks=...)` -> `PrefixKVCache`
  (accepts EITHER raw images OR pre-extracted `[B, 256, 2048]` SigLIP tokens for ultra-fast GH200 Stage A!)
- `denoise_step_from_kv(kv_cache, x_t, timestep, *, mem_tokens=..., mem_mask=...)` -> `v_t: [B, H, 32]`
- `compute_loss_from_kv(kv_cache, actions, *, noise=..., time=..., mem_tokens=..., mem_mask=...)` -> `[B, H, D]`
- `sample_actions_from_kv(kv_cache, *, noise=..., num_steps=10, mem_tokens=..., mem_mask=..., mem_cfg_scale=1.0)`
- `forward_joint(...)` (end-to-end joint VLM + ActionExpert + Memory forward pass for Stage B co-adaptation)
- `enable_memory_modulation(mode="modul_residual")` & `inject_vlm_lora(rank=32, alpha=32.0)`
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MemModMode, PerceptualMemConfig, Pi05Config
from .gemma_expert import (
    PaliGemmaWithActionExpert,
    PrefixKVCache,
    inject_lora_into_stream,
)
from .perceptual_mem import PerceptualMemoryEncoder
from .siglip import SigLIPVisionTower


def posemb_sincos(
    pos: Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Matches `openpi/models/pi0.py:posemb_sincos` (`float32` frequencies & sin/cos)."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim={embedding_dim} must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, dtype=torch.float32, device=pos.device)
    period = min_period * ((max_period / min_period) ** fraction)
    scaling = (2.0 * math.pi) / period
    sin_input = pos.float().unsqueeze(-1) * scaling.unsqueeze(0)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def make_attn_mask(input_mask: Tensor, ar_mask: Tensor) -> Tensor:
    """Matches `openpi/models/pi0.py:make_attn_mask` (`bool[B, N, N]`)."""
    if ar_mask.ndim == 1:
        ar_mask = ar_mask.unsqueeze(0).expand_as(input_mask)
    cumsum = torch.cumsum(ar_mask.long(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] & input_mask[:, :, None]
    return attn_mask & valid_mask


class Pi05Model(nn.Module):
    """Complete `π0.5` model (`SigLIP-400M` + `PaliGemma-2B` + `ActionExpert-300M` + `HistoryBlock` hooks)."""

    def __init__(self, config: Pi05Config | None = None) -> None:
        super().__init__()
        self.config = config or Pi05Config()

        self.vision_tower = SigLIPVisionTower(self.config)
        self.paligemma_with_expert = PaliGemmaWithActionExpert(self.config)

        self.action_in_proj = nn.Linear(self.config.action_dim, self.config.expert_width)
        self.time_mlp_in = nn.Linear(self.config.expert_width, self.config.expert_width)
        self.time_mlp_out = nn.Linear(self.config.expert_width, self.config.expert_width)
        self.action_out_proj = nn.Linear(self.config.expert_width, self.config.action_dim)

        self.mem_encoder: PerceptualMemoryEncoder | None = None
        if self.config.has_perceptual_mem_encoder:
            pm_cfg = self.config.perceptual_mem or PerceptualMemConfig()
            self.mem_encoder = PerceptualMemoryEncoder(pm_cfg)

        self.apply_precision(self.config.precision)

    @property
    def compute_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.precision == "bfloat16" else torch.float32

    @property
    def device(self) -> torch.device:
        return self.action_in_proj.weight.device

    def apply_precision(self, precision: Literal["bfloat16", "float32"] = "bfloat16") -> Pi05Model:
        """Cast heavy transformer matrices to `bfloat16` while keeping sensitive norms/projections in `fp32`."""
        self.config.precision = precision
        if precision == "float32":
            self.to(torch.float32)
            return self
        if precision != "bfloat16":
            raise ValueError(f"Unsupported precision={precision!r}")

        self.to(torch.bfloat16)
        # Keep patch/pos embeddings, 1D RMSNorm scale vectors, and action/time IO projections in float32
        self.vision_tower.patch_embedding.to(torch.float32)
        self.vision_tower.pos_embedding.data = self.vision_tower.pos_embedding.data.float()
        for name, param in self.named_parameters():
            if name.endswith("vlm.input_layernorm.weight") or name.endswith("vlm.post_attention_layernorm.weight"):
                param.data = param.data.float()
            elif name.endswith("vlm_norm.weight") or name.endswith("mem_rms_norm.weight"):
                param.data = param.data.float()
        self.action_in_proj.to(torch.float32)
        self.action_out_proj.to(torch.float32)
        self.time_mlp_in.to(torch.float32)
        self.time_mlp_out.to(torch.float32)
        return self

    def set_gradient_checkpointing(self, enabled: bool = True, *, include_vision: bool = False) -> None:
        self.paligemma_with_expert.gradient_checkpointing = enabled
        self.vision_tower.gradient_checkpointing = enabled and include_vision

    def enable_memory_modulation(self, mode: MemModMode = "modul_residual") -> list[nn.Parameter]:
        """Enable 18-layer `MemoryAttention` + `MemoryRMSNorm` hooks and return their parameters."""
        self.config.mem_mod_mode = mode
        self.paligemma_with_expert.enable_memory_modulation(mode)
        return self.memory_hook_parameters()

    def memory_hook_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for block in self.paligemma_with_expert.layers:
            if block.mem_attn is not None:
                params.extend(block.mem_attn.parameters())
            if block.mem_rms_norm_ffn is not None:
                params.extend(block.mem_rms_norm_ffn.parameters())
        return params

    def memory_gate_parameters(self) -> list[nn.Parameter]:
        """Return the 18-layer `mem_rms_norm_ffn` FiLM gate parameters (for 10x gate LR in Stage A)."""
        params: list[nn.Parameter] = []
        for block in self.paligemma_with_expert.layers:
            if block.mem_rms_norm_ffn is not None:
                params.extend(block.mem_rms_norm_ffn.parameters())
        return params

    def action_expert_parameters(self, *, include_io_Heads: bool = True) -> list[nn.Parameter]:
        """Return `ActionExpert-300M` stream parameters (excluding `mem_attn` hooks)."""
        params: list[nn.Parameter] = []
        for block in self.paligemma_with_expert.layers:
            params.extend(block.expert.parameters())
        params.extend(self.paligemma_with_expert.expert_norm.parameters())
        if include_io_Heads:
            params.extend(self.action_in_proj.parameters())
            params.extend(self.action_out_proj.parameters())
            params.extend(self.time_mlp_in.parameters())
            params.extend(self.time_mlp_out.parameters())
        return params

    def inject_vlm_lora(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        *,
        include_mlp: bool = True,
        include_expert: bool = False,
    ) -> list[nn.Parameter]:
        """Freeze base VLM weights and inject step-0 `0.0`-diff LoRA into `PaliGemma-2B`."""
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.paligemma_with_expert.embed_tokens.weight.requires_grad = False
        for p in self.paligemma_with_expert.vlm_norm.parameters():
            p.requires_grad = False
        for block in self.paligemma_with_expert.layers:
            for p in block.vlm.parameters():
                p.requires_grad = False

        lora_params = inject_lora_into_stream(
            self.paligemma_with_expert.layers,
            stream="vlm",
            rank=rank,
            alpha=alpha,
            include_mlp=include_mlp,
        )
        if include_expert:
            for block in self.paligemma_with_expert.layers:
                for p in block.expert.parameters():
                    p.requires_grad = False
            lora_params.extend(
                inject_lora_into_stream(
                    self.paligemma_with_expert.layers,
                    stream="expert",
                    rank=rank,
                    alpha=alpha,
                    include_mlp=include_mlp,
                )
            )
        return lora_params

    def freeze_backbone(self) -> None:
        """Freeze all base `π0.5` weights (`SigLIP` + `PaliGemma-2B` + `ActionExpert-300M` + IO heads)."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.memory_hook_parameters():
            p.requires_grad = True

    def encode_images(self, images: Tensor, *, exact_fp32_attn: bool = False) -> Tensor:
        """Encode `[N, 224, 224, 3]` or `[N, 3, 224, 224]` in `[-1, 1]` -> `[N, 256, 2048]`."""
        return self.vision_tower(images, exact_fp32_attn=exact_fp32_attn)

    def embed_prefix(
        self,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        *,
        images: Sequence[Tensor] | None = None,
        image_embs: Sequence[Tensor] | None = None,
        image_masks: Sequence[Tensor] | None = None,
        extra_prefix_embs: Tensor | None = None,
        extra_prefix_mask: Tensor | None = None,
        exact_fp32_attn: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build VLM prefix tokens `[images..., lang_tokens, extra_prefix_embs?]` matching `HistoryPi0.embed_prefix`."""
        if (images is None) == (image_embs is None):
            raise ValueError("Provide exactly one of `images` (raw RGB) or `image_embs` (cached SigLIP tokens)")

        bsz = lang_tokens.shape[0]
        dev = lang_tokens.device
        target_dtype = self.compute_dtype

        if image_embs is None:
            assert images is not None
            # Batch all camera views through SigLIP in one fused forward pass when shapes match
            if len(images) > 1 and all(img.shape == images[0].shape for img in images):
                stacked = torch.cat(list(images), dim=0)
                encoded_all = self.vision_tower(stacked, exact_fp32_attn=exact_fp32_attn)
                encoded_views = list(encoded_all.chunk(len(images), dim=0))
            else:
                encoded_views = [self.vision_tower(img, exact_fp32_attn=exact_fp32_attn) for img in images]
        else:
            encoded_views = list(image_embs)

        if image_masks is None:
            view_masks = [torch.ones((bsz,), dtype=torch.bool, device=dev) for _ in encoded_views]
        else:
            view_masks = [m.to(device=dev, dtype=torch.bool) for m in image_masks]

        token_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        for emb, v_mask in zip(encoded_views, view_masks, strict=True):
            num_tok = emb.shape[1]
            token_parts.append(emb.to(target_dtype))
            mask_parts.append(v_mask[:, None].expand(bsz, num_tok))

        lang_emb = self.paligemma_with_expert.embed_language(lang_tokens.long()).to(target_dtype)
        token_parts.append(lang_emb)
        mask_parts.append(lang_masks.to(device=dev, dtype=torch.bool))

        if extra_prefix_embs is not None:
            token_parts.append(extra_prefix_embs.to(device=dev, dtype=target_dtype))
            if extra_prefix_mask is None:
                extra_prefix_mask = torch.ones(
                    (bsz, int(extra_prefix_embs.shape[1])), dtype=torch.bool, device=dev
                )
            mask_parts.append(extra_prefix_mask.to(device=dev, dtype=torch.bool))

        prefix_tokens = torch.cat(token_parts, dim=1)
        prefix_pad_masks = torch.cat(mask_parts, dim=1)
        prefix_ar_mask = torch.zeros((prefix_tokens.shape[1],), dtype=torch.bool, device=dev)
        return prefix_tokens, prefix_pad_masks, prefix_ar_mask

    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build ActionExpert suffix tokens & AdaRMS time conditioning matching `HistoryPi0.embed_suffix`."""
        bsz, horizon, act_dim = noisy_actions.shape
        if act_dim < self.config.action_dim:
            noisy_actions = F.pad(noisy_actions, (0, self.config.action_dim - act_dim))

        target_dtype = self.compute_dtype
        act_tokens = F.linear(
            noisy_actions.float(),
            self.action_in_proj.weight.float(),
            self.action_in_proj.bias.float(),
        ).to(target_dtype)

        t_emb = posemb_sincos(
            timestep,
            self.config.expert_width,
            min_period=self.config.time_min_period,
            max_period=self.config.time_max_period,
        )
        t_emb = F.silu(
            F.linear(t_emb, self.time_mlp_in.weight.float(), self.time_mlp_in.bias.float())
        )
        t_emb = F.silu(
            F.linear(t_emb, self.time_mlp_out.weight.float(), self.time_mlp_out.bias.float())
        )
        adarms_cond = t_emb.to(target_dtype)

        suffix_pad_masks = torch.ones((bsz, horizon), dtype=torch.bool, device=noisy_actions.device)
        suffix_ar_mask = torch.zeros((horizon,), dtype=torch.bool, device=noisy_actions.device)
        suffix_ar_mask[0] = True
        return act_tokens, suffix_pad_masks, suffix_ar_mask, adarms_cond

    def prefill_prefix_kv(
        self,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        *,
        images: Sequence[Tensor] | None = None,
        image_embs: Sequence[Tensor] | None = None,
        image_masks: Sequence[Tensor] | None = None,
        extra_prefix_embs: Tensor | None = None,
        extra_prefix_mask: Tensor | None = None,
        return_prefix_hidden: bool = False,
        exact_fp32_attn: bool = False,
    ) -> PrefixKVCache | tuple[PrefixKVCache, Tensor]:
        """Prefill 18-layer VLM prefix KV cache (`no_grad`-friendly for Stage A & ODE inference)."""
        prefix_tokens, prefix_pad_masks, prefix_ar_mask = self.embed_prefix(
            lang_tokens,
            lang_masks,
            images=images,
            image_embs=image_embs,
            image_masks=image_masks,
            extra_prefix_embs=extra_prefix_embs,
            extra_prefix_mask=extra_prefix_mask,
            exact_fp32_attn=exact_fp32_attn,
        )
        prefix_attn_mask = make_attn_mask(prefix_pad_masks, prefix_ar_mask)
        prefix_hidden, kv_cache = self.paligemma_with_expert.prefill_prefix(
            prefix_tokens,
            prefix_pad_masks,
            prefix_attn_mask,
            exact_fp32_attn=exact_fp32_attn,
        )
        if return_prefix_hidden:
            return kv_cache, prefix_hidden
        return kv_cache

    def denoise_step_from_kv(
        self,
        kv_cache: PrefixKVCache,
        x_t: Tensor,
        timestep: Tensor,
        *,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_kv_cache: list[tuple[Tensor, Tensor, Tensor, int] | None] | None = None,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        """Predict flow velocity `v_t: [B, action_horizon, action_dim]` given cached prefix KV."""
        suffix_tokens, suffix_pad_masks, suffix_ar_mask, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_pad_masks, suffix_ar_mask)
        suffix_out = self.paligemma_with_expert.forward_suffix_with_kv(
            suffix_tokens,
            suffix_pad_masks,
            suffix_attn_mask,
            adarms_cond,
            kv_cache,
            mem_tokens=mem_tokens,
            mem_mask=mem_mask,
            mem_kv_cache=mem_kv_cache,
            exact_fp32_attn=exact_fp32_attn,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        return F.linear(
            suffix_out.float(),
            self.action_out_proj.weight.float(),
            self.action_out_proj.bias.float(),
        )

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        dist = torch.distributions.Beta(
            torch.tensor(self.config.time_beta_alpha, dtype=torch.float32),
            torch.tensor(self.config.time_beta_beta, dtype=torch.float32),
        )
        return dist.sample((batch_size,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001

    def sample_noise(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)

    def compute_loss_from_kv(
        self,
        kv_cache: PrefixKVCache,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        loss_dims: int = 32,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        """Compute per-element squared flow-matching error `[B, H, loss_dims]` from cached prefix KV.

        Note: Upstream `HistoryPi0.compute_loss` pads 8-DoF actions with 24 zeros to 32 dims and
        averages MSE across all `loss_dims=32` dims (`loss.mean(dim=-1)`). Both `loss_dims=32` and
        `loss_dims=8` are supported.
        """
        bsz, horizon, act_dim = actions.shape
        if act_dim < self.config.action_dim:
            actions_32 = F.pad(actions.float(), (0, self.config.action_dim - act_dim))
        else:
            actions_32 = actions[:, :, : self.config.action_dim].float()

        if noise is None:
            noise = self.sample_noise(bsz, actions.device)
        if time is None:
            time = self.sample_time(bsz, actions.device)

        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions_32
        u_t = noise - actions_32

        v_t = self.denoise_step_from_kv(
            kv_cache,
            x_t,
            time,
            mem_tokens=mem_tokens,
            mem_mask=mem_mask,
            exact_fp32_attn=exact_fp32_attn,
        )
        return F.mse_loss(v_t[:, :horizon, :loss_dims], u_t[:, :horizon, :loss_dims], reduction="none")

    def forward_joint(
        self,
        actions: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        *,
        images: Sequence[Tensor] | None = None,
        image_embs: Sequence[Tensor] | None = None,
        image_masks: Sequence[Tensor] | None = None,
        extra_prefix_embs: Tensor | None = None,
        extra_prefix_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        loss_dims: int = 32,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        """Full joint VLM + ActionExpert + Memory forward pass for Stage-B co-adaptation (`[B, H, loss_dims]`)."""
        bsz, horizon, act_dim = actions.shape
        if act_dim < self.config.action_dim:
            actions_32 = F.pad(actions.float(), (0, self.config.action_dim - act_dim))
        else:
            actions_32 = actions[:, :, : self.config.action_dim].float()

        if noise is None:
            noise = self.sample_noise(bsz, actions.device)
        if time is None:
            time = self.sample_time(bsz, actions.device)

        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions_32
        u_t = noise - actions_32

        prefix_tokens, prefix_pad_masks, prefix_ar_mask = self.embed_prefix(
            lang_tokens,
            lang_masks,
            images=images,
            image_embs=image_embs,
            image_masks=image_masks,
            extra_prefix_embs=extra_prefix_embs,
            extra_prefix_mask=extra_prefix_mask,
            exact_fp32_attn=exact_fp32_attn,
        )
        suffix_tokens, suffix_pad_masks, suffix_ar_mask, adarms_cond = self.embed_suffix(x_t, time)

        input_mask = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        full_attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.long(), dim=1) - 1

        _, suffix_out = self.paligemma_with_expert.forward_joint(
            prefix_tokens,
            suffix_tokens,
            positions,
            full_attn_mask,
            adarms_cond,
            mem_tokens=mem_tokens,
            mem_mask=mem_mask,
            exact_fp32_attn=exact_fp32_attn,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        v_t = F.linear(
            suffix_out.float(),
            self.action_out_proj.weight.float(),
            self.action_out_proj.bias.float(),
        )
        return F.mse_loss(v_t[:, :horizon, :loss_dims], u_t[:, :horizon, :loss_dims], reduction="none")

    @torch.no_grad()
    def sample_actions_from_kv(
        self,
        kv_cache: PrefixKVCache,
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_cfg_scale: float = 1.0,
        noise_temp: float = 1.0,
        antithetic: bool = False,
        best_of_k: int = 1,
        return_all_candidates: bool = False,
        ode_solver: str = "euler",
        generator: torch.Generator | None = None,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        """Flow-ODE integration from `t=1.0` to `t=0.0` (`euler`, `midpoint`, `heun`, `best_of_k` medoid, `Mem-CFG`)."""
        steps = num_steps if num_steps is not None else self.config.num_inference_steps
        bsz = kv_cache.pad_masks.shape[0]
        dev = kv_cache.pad_masks.device
        k_cand = max(1, int(best_of_k))

        if k_cand > 1:
            base_noise = (
                noise.float()
                if (noise is not None and noise.shape[0] == bsz * k_cand)
                else self.sample_noise(bsz * k_cand, dev, generator=generator)
            )
            if abs(noise_temp - 1.0) > 1e-8:
                base_noise = base_noise * float(noise_temp)
            x_t = base_noise
            eff_bsz = bsz * k_cand
            kv_base = kv_cache.repeat_interleave(k_cand)
            mem_tokens_base = mem_tokens.repeat_interleave(k_cand, dim=0) if mem_tokens is not None else None
            mem_mask_base = mem_mask.repeat_interleave(k_cand, dim=0) if mem_mask is not None else None
        elif antithetic:
            base_noise = noise.float() if noise is not None else self.sample_noise(bsz, dev, generator=generator)
            if abs(noise_temp - 1.0) > 1e-8:
                base_noise = base_noise * float(noise_temp)
            x_t = torch.cat([base_noise, -base_noise], dim=0)
            eff_bsz = bsz * 2
            kv_base = kv_cache.repeat_interleave(2)
            mem_tokens_base = mem_tokens.repeat_interleave(2, dim=0) if mem_tokens is not None else None
            mem_mask_base = mem_mask.repeat_interleave(2, dim=0) if mem_mask is not None else None
        else:
            base_noise = noise.float() if noise is not None else self.sample_noise(bsz, dev, generator=generator)
            if abs(noise_temp - 1.0) > 1e-8:
                base_noise = base_noise * float(noise_temp)
            x_t = base_noise
            eff_bsz = bsz
            kv_base = kv_cache
            mem_tokens_base = mem_tokens
            mem_mask_base = mem_mask

        use_mem_cfg = (
            abs(mem_cfg_scale - 1.0) > 1e-6
            and mem_tokens_base is not None
            and self.config.mem_mod_mode != "none"
        )
        if use_mem_cfg:
            assert mem_tokens_base is not None
            kv_run = kv_base.repeat_interleave(2)
            s_mem = mem_tokens_base.shape[1]
            cond_mask = (
                mem_mask_base.to(device=dev, dtype=torch.bool)
                if mem_mask_base is not None
                else torch.ones((eff_bsz, s_mem), dtype=torch.bool, device=dev)
            )
            uncond_mask = torch.zeros((eff_bsz, s_mem), dtype=torch.bool, device=dev)
            mem_tokens_run = torch.cat([mem_tokens_base, mem_tokens_base], dim=0)
            mem_mask_run = torch.cat([cond_mask, uncond_mask], dim=0)
        else:
            kv_run = kv_base
            mem_tokens_run = mem_tokens_base
            mem_mask_run = mem_mask_base

        mem_kv_cache = (
            self.paligemma_with_expert.precompute_memory_kv(
                mem_tokens_run, mem_mask_run, dtype=self.compute_dtype
            )
            if (mem_tokens_run is not None and self.config.mem_mod_mode != "none")
            else None
        )

        def _eval_vel(cur_x: Tensor, t_scalar: float) -> Tensor:
            if use_mem_cfg:
                t_vec = torch.full((eff_bsz * 2,), t_scalar, dtype=torch.float32, device=dev)
                x_in = torch.cat([cur_x, cur_x], dim=0)
                v_both = self.denoise_step_from_kv(
                    kv_run,
                    x_in,
                    t_vec,
                    mem_tokens=mem_tokens_run,
                    mem_mask=mem_mask_run,
                    mem_kv_cache=mem_kv_cache,
                    exact_fp32_attn=exact_fp32_attn,
                )
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                return v_uncond + mem_cfg_scale * (v_cond - v_uncond)
            t_vec = torch.full((eff_bsz,), t_scalar, dtype=torch.float32, device=dev)
            return self.denoise_step_from_kv(
                kv_run,
                cur_x,
                t_vec,
                mem_tokens=mem_tokens_run,
                mem_mask=mem_mask_run,
                mem_kv_cache=mem_kv_cache,
                exact_fp32_attn=exact_fp32_attn,
            )

        solver = ode_solver.strip().lower()
        if solver == "euler":
            dt = -1.0 / float(steps)
            for step_i in range(steps):
                t_val = 1.0 + step_i * dt
                x_t = x_t + dt * _eval_vel(x_t, t_val)
        elif solver == "midpoint":
            macro_steps = max(1, steps // 2)
            h = -1.0 / float(macro_steps)
            for step_i in range(macro_steps):
                t_val = 1.0 + step_i * h
                k1 = _eval_vel(x_t, t_val)
                x_mid = x_t + 0.5 * h * k1
                k2 = _eval_vel(x_mid, t_val + 0.5 * h)
                x_t = x_t + h * k2
        elif solver == "heun":
            macro_steps = max(1, steps // 2)
            h = -1.0 / float(macro_steps)
            for step_i in range(macro_steps):
                t_val = 1.0 + step_i * h
                k1 = _eval_vel(x_t, t_val)
                x_pred = x_t + h * k1
                t_next = max(1e-3, t_val + h)
                k2 = _eval_vel(x_pred, t_next)
                x_t = x_t + 0.5 * h * (k1 + k2)
        else:
            raise ValueError(f"Unknown ode_solver={ode_solver!r}; expected 'euler', 'midpoint', or 'heun'")

        if k_cand > 1:
            x_cand = x_t.view(bsz, k_cand, self.config.action_horizon, self.config.action_dim)
            if return_all_candidates:
                return x_cand
            x_7 = x_cand[:, :, :, : min(7, self.config.real_action_dim)]
            pw_l1 = (x_7.unsqueeze(2) - x_7.unsqueeze(1)).abs().mean(dim=(-2, -1))
            best_idx = pw_l1.sum(dim=-1).argmin(dim=-1)
            return x_cand[torch.arange(bsz, device=dev), best_idx]
        if antithetic:
            x_pos, x_neg = x_t.chunk(2, dim=0)
            return 0.5 * (x_pos + x_neg)
        return x_t

    @torch.no_grad()
    def sample_actions(
        self,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        *,
        images: Sequence[Tensor] | None = None,
        image_embs: Sequence[Tensor] | None = None,
        image_masks: Sequence[Tensor] | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        kv_cache = self.prefill_prefix_kv(
            lang_tokens,
            lang_masks,
            images=images,
            image_embs=image_embs,
            image_masks=image_masks,
            exact_fp32_attn=exact_fp32_attn,
        )
        assert isinstance(kv_cache, PrefixKVCache)
        return self.sample_actions_from_kv(
            kv_cache,
            noise=noise,
            num_steps=num_steps,
            mem_tokens=mem_tokens,
            mem_mask=mem_mask,
            mem_cfg_scale=mem_cfg_scale,
            generator=generator,
            exact_fp32_attn=exact_fp32_attn,
        )


def save_pi05_checkpoint(
    model: Pi05Model,
    output_dir: str | Path,
    *,
    norm_stats: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save_json(out / "config.json")
    state_dict = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"))
    if norm_stats is not None:
        (out / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2) + "\n")
    if extra_meta is not None:
        (out / "conversion_meta.json").write_text(json.dumps(extra_meta, indent=2) + "\n")
    return out


def load_pi05_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    precision: Literal["bfloat16", "float32"] | None = None,
    mem_mod_mode: MemModMode | None = None,
    strict: bool = True,
) -> tuple[Pi05Model, dict[str, Any] | None]:
    ckpt = Path(checkpoint_dir)
    config = Pi05Config.load_json(ckpt / "config.json")
    if precision is not None:
        config.precision = precision
    if mem_mod_mode is not None:
        config.mem_mod_mode = mem_mod_mode

    model = Pi05Model(config)
    sd = load_file(str(ckpt / "model.safetensors"), device="cpu")
    model.load_state_dict(sd, strict=strict)
    model.apply_precision(config.precision)
    model.to(device)
    model.eval()

    norm_stats: dict[str, Any] | None = None
    norm_path = ckpt / "norm_stats.json"
    if norm_path.exists():
        raw_ns = json.loads(norm_path.read_text())
        norm_stats = raw_ns.get("norm_stats", raw_ns)
    return model, norm_stats
