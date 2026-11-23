"""Frozen SigLIP (`google/siglip-base-patch16-224`) multi-camera spatial & text encoder.

Used identically by:
1. Offline feature extraction on Miyabi GH200 (`robomme_data.extract_siglip_cache`)
2. Online headless ManiSkill3/SAPIEN evaluation on local RTX 5080 (`src/robomme_eval/`)

Guarantees 100% identical image resize/pad/[-1, 1] scaling and spatial adaptive pooling:
- `front_emb`: `(T, 64, 768)` (`8x8` spatial grid pooled from `14x14` SigLIP patch tokens)
- `wrist_emb`: `(T, 16, 768)` (`4x4` spatial grid pooled from `14x14` SigLIP patch tokens)
- `text_token_emb`: `(B, 64, 768)` + `text_pool_emb`: `(B, 768)` + `text_attn_mask`: `(B, 64)`
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"


def preprocess_siglip_rgb(
    images: np.ndarray | torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
    device: torch.device | str | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert RGB image(s) to SigLIP `[-1, 1]` `(N, 3, H, W)` tensor with aspect-ratio-preserving pad.

    For RoboMME `(256, 256, 3)` `uint8` frames, aspect ratio is 1:1 (`pad=0`), so this reduces to
    exact antialiased bilinear downsampling `256x256 -> 224x224` followed by `x * 2.0 - 1.0`.
    """
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(images))
    else:
        x = images

    single = x.ndim == 3
    if single:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D RGB input, got shape {tuple(x.shape)}")

    # Convert NHWC -> NCHW if channels are last
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got shape {tuple(x.shape)}")

    if device is not None:
        x = x.to(device=device, non_blocking=True)

    if x.dtype == torch.uint8:
        x = x.to(torch.float32) / 255.0
    else:
        x = x.to(torch.float32)
        if x.max() > 1.0 + 1e-4:
            x = x / 255.0

    _, _, h, w = x.shape
    target_h, target_w = image_size
    if (h, w) != (target_h, target_w):
        # Aspect-preserving resize + zero pad (in [0, 1] space before [-1, 1] shift; pad=0 when h==w)
        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        if (new_h, new_w) != (target_h, target_w):
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    x = x * 2.0 - 1.0
    return x.to(dtype=out_dtype)


def pool_siglip_patch_tokens(
    patch_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Pool `(N, 196, D)` (`14x14`) SigLIP patch tokens to `(N, gh * gw, D)` spatial grid tokens."""
    n, num_patches, d = patch_tokens.shape
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        raise ValueError(f"Expected square patch count, got {num_patches}")
    gh, gw = grid_hw
    if (gh, gw) == (side, side):
        return patch_tokens
    fmap = patch_tokens.reshape(n, side, side, d).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(fmap, (gh, gw))
    return pooled.permute(0, 2, 3, 1).reshape(n, gh * gw, d)


class SiglipFeatureEncoder(nn.Module):
    """Frozen `google/siglip-base-patch16-224` encoder for multi-camera spatial & text features."""

    def __init__(
        self,
        model_id: str = SIGLIP_MODEL_ID,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        front_grid: tuple[int, int] = (8, 8),
        wrist_grid: tuple[int, int] = (4, 4),
        max_text_len: int = 64,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device_obj = torch.device(device)
        self.dtype = dtype
        self.front_grid = front_grid
        self.wrist_grid = wrist_grid
        self.max_text_len = max_text_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device_obj)

        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @torch.inference_mode()
    def encode_views(
        self,
        front_rgb: np.ndarray | torch.Tensor,
        wrist_rgb: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 512,
        return_cpu: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `front_rgb` and `wrist_rgb` into `(N, 64, 768)` and `(N, 16, 768)` tensors.

        Fuses front and wrist images into a single batched ViT forward pass per chunk for maximum
        GPU occupancy during offline extraction and minimal latency (`~1 ms`) during online rollout.
        """
        f_arr = torch.from_numpy(np.ascontiguousarray(front_rgb)) if isinstance(front_rgb, np.ndarray) else front_rgb
        w_arr = torch.from_numpy(np.ascontiguousarray(wrist_rgb)) if isinstance(wrist_rgb, np.ndarray) else wrist_rgb
        if f_arr.ndim == 3:
            f_arr = f_arr.unsqueeze(0)
        if w_arr.ndim == 3:
            w_arr = w_arr.unsqueeze(0)
        if f_arr.shape[0] != w_arr.shape[0]:
            raise ValueError(f"Mismatched batch sizes: front={f_arr.shape[0]} vs wrist={w_arr.shape[0]}")

        total_n = f_arr.shape[0]
        front_chunks: list[torch.Tensor] = []
        wrist_chunks: list[torch.Tensor] = []

        # Each step processes `step_n` pairs = `2 * step_n` images through `vision_model`
        pair_bs = max(1, batch_size // 2)
        for start in range(0, total_n, pair_bs):
            end = min(total_n, start + pair_bs)
            f_px = preprocess_siglip_rgb(f_arr[start:end], device=self.device_obj, out_dtype=self.dtype)
            w_px = preprocess_siglip_rgb(w_arr[start:end], device=self.device_obj, out_dtype=self.dtype)
            cat_px = torch.cat([f_px, w_px], dim=0)  # (2 * m, 3, 224, 224)

            v_out = self.model.vision_model(pixel_values=cat_px)
            lhs = v_out.last_hidden_state  # (2 * m, 196, 768)
            m = end - start
            f_tok = pool_siglip_patch_tokens(lhs[:m], self.front_grid).to(self.dtype)
            w_tok = pool_siglip_patch_tokens(lhs[m:], self.wrist_grid).to(self.dtype)

            if return_cpu:
                f_tok = f_tok.cpu()
                w_tok = w_tok.cpu()
            front_chunks.append(f_tok)
            wrist_chunks.append(w_tok)

        front_out = torch.cat(front_chunks, dim=0) if len(front_chunks) > 1 else front_chunks[0]
        wrist_out = torch.cat(wrist_chunks, dim=0) if len(wrist_chunks) > 1 else wrist_chunks[0]
        return front_out, wrist_out

    @torch.inference_mode()
    def encode_texts(
        self,
        prompts: Sequence[str],
        *,
        return_cpu: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode language prompts with caching.

        Note: SigLIP's SentencePiece tokenizer uses `pad_token_id == eos_token_id == 1` and trains
        `text_model` with fixed `max_length=64` without attention masking. We run `text_model(input_ids)`
        exactly as trained, and construct `text_attn_mask` marking all content tokens up to and
        including the first EOS token so downstream cross-attention can optionally mask trailing pad slots.
        """
        missing = [p for p in set(prompts) if p not in self._text_cache]
        if missing:
            tok_out = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.max_text_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok_out["input_ids"].to(self.device_obj)
            t_out = self.model.text_model(input_ids=input_ids)
            tok_emb = t_out.last_hidden_state.to(self.dtype).cpu()   # (U, 64, 768)
            pool_emb = t_out.pooler_output.to(self.dtype).cpu()      # (U, 768)

            ids_cpu = tok_out["input_ids"]
            pad_id = self.tokenizer.pad_token_id
            for idx, prompt_str in enumerate(missing):
                row_ids = ids_cpu[idx]
                pad_positions = (row_ids == pad_id).nonzero(as_tuple=False).flatten()
                if pad_positions.numel() > 0:
                    # Keep tokens up to and including the first EOS/pad token
                    valid_len = int(pad_positions[0].item()) + 1
                else:
                    valid_len = int(row_ids.numel())
                mask = torch.zeros_like(row_ids, dtype=torch.bool)
                mask[:valid_len] = True
                self._text_cache[prompt_str] = (tok_emb[idx], pool_emb[idx], mask)

        tok_list = [self._text_cache[p][0] for p in prompts]
        pool_list = [self._text_cache[p][1] for p in prompts]
        mask_list = [self._text_cache[p][2] for p in prompts]

        tok_batch = torch.stack(tok_list, dim=0)
        pool_batch = torch.stack(pool_list, dim=0)
        mask_batch = torch.stack(mask_list, dim=0)
        if not return_cpu:
            tok_batch = tok_batch.to(self.device_obj, non_blocking=True)
            pool_batch = pool_batch.to(self.device_obj, non_blocking=True)
            mask_batch = mask_batch.to(self.device_obj, non_blocking=True)
        return {
            "text_token_emb": tok_batch,
            "text_pool_emb": pool_batch,
            "text_attn_mask": mask_batch,
        }
