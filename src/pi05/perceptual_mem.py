"""Exact PyTorch port of MME-VLA's `PerceptualMemory` (`perceptual-framesamp-modul`, `44.51%` SR).

Ports:
- `mme_vla_suite/shared/posemb_3d.py` (`PosEmb3D`)
- `mme_vla_suite/models/representation/mem_encoder.py` (`FeatureEncoder`)
- `mme_vla_suite/models/representation/percep_mem.py` (`PerceptualMemory`)
- `mme_vla_suite/shared/mem_buffer.py` (`MemoryBuffer` frame sampling)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import PerceptualMemConfig
from .siglip import pool_vision_tokens


class PosEmb3D:
    """Exact port of `mme_vla_suite/shared/posemb_3d.py`."""

    def __init__(self, dim: int = 768, temporal_base: float = 10_000.0, spatial_base: float = 1_000.0) -> None:
        if dim % 6 != 0:
            raise ValueError(f"dim={dim} must be divisible by 6")
        self.dim = dim
        width = dim // 6
        omega = np.arange(width, dtype=np.float32) / float(width - 1)
        self.temporal_omega = (1.0 / (temporal_base**omega)).astype(np.float32)
        self.spatial_omega = (1.0 / (spatial_base**omega)).astype(np.float32)

        self.spatial_pe = {
            8: self._compute_spatial_pe(8, scale=2, offset=1),
            4: self._compute_spatial_pe(4, scale=4, offset=2),
            2: self._compute_spatial_pe(2, scale=8, offset=4),
        }
        self.temporal_pe = self._compute_temporal_pe(4096)

    def _compute_spatial_pe(self, grid: int, scale: int, offset: int) -> np.ndarray:
        y, x = np.mgrid[:grid, :grid]
        yf = (scale * y + offset).reshape(-1).astype(np.float32)
        xf = (scale * x + offset).reshape(-1).astype(np.float32)
        ye = np.outer(yf, self.spatial_omega)
        xe = np.outer(xf, self.spatial_omega)
        return np.concatenate([np.sin(ye), np.cos(ye), np.sin(xe), np.cos(xe)], axis=-1).astype(np.float32)

    def _compute_temporal_pe(self, max_length: int) -> np.ndarray:
        pos = np.arange(max_length, dtype=np.float32)
        te = np.outer(pos, self.temporal_omega)
        return np.concatenate([np.sin(te), np.cos(te)], axis=-1).astype(np.float32)

    def __call__(self, pos: np.ndarray, spatial_size: int) -> np.ndarray:
        pos_arr = np.asarray(pos, dtype=np.int64)
        max_len = int(pos_arr.max()) + 1 if pos_arr.size > 0 else 1
        if max_len > self.temporal_pe.shape[0]:
            self.temporal_pe = self._compute_temporal_pe(max_len * 2)
        sp = self.spatial_pe[spatial_size]  # (hw, 4*w)
        tp = self.temporal_pe[pos_arr]  # (T, 2*w)
        hw = spatial_size * spatial_size
        tp_rep = np.repeat(tp[:, None, :], hw, axis=1)
        sp_rep = np.broadcast_to(sp[None, :, :], (pos_arr.shape[0], hw, sp.shape[-1]))
        return np.concatenate([tp_rep, sp_rep], axis=-1).astype(np.float32)


class PerceptualMemoryEncoder(nn.Module):
    """Exact port of `PerceptualMemory` + `FeatureEncoder` (`percep_mem.py`, `mem_encoder.py`)."""

    def __init__(self, config: PerceptualMemConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.img_input_dim
        if config.use_state_emb:
            self.state_proj: nn.Linear | None = nn.Linear(config.state_input_dim, config.state_hidden_dim)
            input_dim += config.state_hidden_dim
        else:
            self.state_proj = None

        if config.use_pos_emb:
            self.pos_proj: nn.Linear | None = nn.Linear(config.pos_input_dim, config.pos_hidden_dim)
            input_dim += config.pos_hidden_dim
        else:
            self.pos_proj = None

        self.encoder_static = nn.Linear(input_dim, config.memory_token_dim)

    def forward(
        self,
        static_image_emb: Tensor,
        static_pos_emb: Tensor | None = None,
        static_state_emb: Tensor | None = None,
    ) -> Tensor:
        w_dtype = self.encoder_static.weight.dtype
        parts = [static_image_emb.to(w_dtype)]
        if self.config.use_pos_emb and self.pos_proj is not None and static_pos_emb is not None:
            parts.append(F.silu(self.pos_proj(static_pos_emb.to(w_dtype))))
        if self.config.use_state_emb and self.state_proj is not None and static_state_emb is not None:
            parts.append(F.silu(self.state_proj(static_state_emb.to(w_dtype))))
        cat = torch.cat(parts, dim=-1)
        return self.encoder_static(cat)


class FrameSamplingBuffer:
    """Online episode buffer matching `mme_vla_suite/shared/mem_buffer.py` (`prepare_frame_sampling`)."""

    def __init__(self, config: PerceptualMemConfig | None = None) -> None:
        self.config = config or PerceptualMemConfig()
        self.spatial_size = int(math.isqrt(self.config.token_per_image))
        self.max_frames = self.config.budget // (self.config.token_per_image * self.config.num_views)
        self.pos_embedder = PosEmb3D(dim=self.config.pos_input_dim)
        self.clear()

    def clear(self) -> None:
        self._img_embs: list[np.ndarray] = []  # each (V, P, 2048) float32
        self._states: list[np.ndarray] = []  # each (8,) float32

    @property
    def num_frames(self) -> int:
        return len(self._img_embs)

    def add_encoded_frames(self, tokens_256: Tensor, states: np.ndarray) -> None:
        """Append `(T, V, 256, 2048)` or `(T, 256, 2048)` SigLIP tokens + raw `(T, 8)` states."""
        if tokens_256.ndim == 3:
            tokens_256 = tokens_256.unsqueeze(1)
        pooled = pool_vision_tokens(
            tokens_256, target_tokens=self.config.token_per_image, pool_type=self.config.pool_type
        )
        pooled_np = pooled.detach().float().cpu().numpy()
        states_np = np.asarray(states, dtype=np.float32)
        for i in range(pooled_np.shape[0]):
            self._img_embs.append(pooled_np[i])
            self._states.append(states_np[i])

    def prepare_inputs(self, norm_state_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(static_image_emb, static_pos_emb, static_state_emb, static_mask)` matching `mem_buffer.py`."""
        step_idx = len(self._img_embs) - 1
        if step_idx < 0:
            raise RuntimeError("FrameSamplingBuffer is empty; call add_encoded_frames before prepare_inputs")

        if step_idx < self.max_frames:
            indices = list(range(step_idx + 1))
        else:
            indices = np.linspace(0, step_idx, self.max_frames, dtype=np.int32).tolist()

        v = self.config.num_views
        p = self.config.token_per_image
        sampled_img = np.stack([self._img_embs[idx] for idx in indices], axis=0)  # (L, V, P, 2048)
        pos_indices = np.concatenate(
            [np.arange(idx * v, (idx + 1) * v, dtype=np.int64) for idx in indices], axis=0
        )
        sampled_pos = self.pos_embedder(pos_indices, self.spatial_size).reshape(
            len(indices), v, p, self.config.pos_input_dim
        )
        sampled_state = np.stack([self._states[idx] for idx in indices], axis=0)  # (L, 8)
        if norm_state_fn is not None:
            sampled_state = norm_state_fn(sampled_state)
        frame_mask = np.ones((len(indices),), dtype=np.bool_)

        if len(indices) < self.max_frames:
            pad_n = self.max_frames - len(indices)
            sampled_img = np.pad(sampled_img, ((0, pad_n), (0, 0), (0, 0), (0, 0)))
            sampled_pos = np.pad(sampled_pos, ((0, pad_n), (0, 0), (0, 0), (0, 0)))
            sampled_state = np.pad(sampled_state, ((0, pad_n), (0, 0)))
            frame_mask = np.pad(frame_mask, ((0, pad_n),), constant_values=False)

        img_emb = sampled_img.reshape(self.config.budget, self.config.img_input_dim).astype(np.float32)
        pos_emb = sampled_pos.reshape(self.config.budget, self.config.pos_input_dim).astype(np.float32)
        state_emb = np.repeat(sampled_state, v * p, axis=0).astype(np.float32)
        mask = np.repeat(frame_mask, v * p)
        return img_emb, pos_emb, state_emb, mask
