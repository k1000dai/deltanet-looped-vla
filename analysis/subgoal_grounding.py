#!/usr/bin/env python3
"""Verified RoboMME Grounded Subgoal `<row_y, col_x>` Parser & Affine Coordinate Utilities.

CRITICAL ROBOMME COORDINATE CONVENTION (verified in
`external/robomme_benchmark/src/robomme/robomme_env/utils/segmentation_utils.py` lines 74-103, 211):
  1. RoboMME renders front/wrist RGB and segmentation masks at native `256 x 256` (`H=256, W=256`).
  2. Object centers are computed from `coords = np.argwhere(mask)` (`(row_y, col_x)`):
       - `center_y = int(coords[:, 0].mean())`  -> vertical row index `y in [0, 255]` (`0` = top, `255` = bottom)
       - `center_x = int(coords[:, 1].mean())`  -> horizontal col index `x in [0, 255]` (`0` = left, `255` = right)
     and formatted into `grounded_subgoal` strings as **`f"<{center_y}, {center_x}>"` (`<row_y, col_x>`)**!
  3. Downstream PyTorch `F.grid_sample` / `F.affine_grid` and `SpatialSoftmax` use canonical
     **`(x_horiz, y_vert)`** order (`coord[..., 0] = x_col`, `coord[..., 1] = y_row`) in `[-1, 1]^2`.

This module converts `<row_y, col_x>` strings into canonical `(x_col, y_row)` pixel and `[-1, 1]^2`
normalized coordinates (default `align_corners=False`, matching `vision_encoder.py:preprocess_siglip_rgb`
and `F.grid_sample(..., align_corners=False)`), handles 0-, 1-, and 2-coordinate subgoals (e.g. `MoveCube`
`"hook the cube at <y1, x1> to the target at <y2, x2>..."`), provides exact inverse affine target
transformations for `U3` sub-grid random crops (`F.grid_sample`), and extracts count/stage/color metadata
for `U4` auxiliary grounding and Cycle 2A/3 mechanistic probes (`RESEARCH_PLAN.md` v2.3 §3.7 & §4.3).
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ROBOMME_IMAGE_HW: Tuple[int, int] = (256, 256)

# Matches `<y, x>` integer (or float) coordinate pairs inside RoboMME `grounded_subgoal` strings.
_COORD_PAIR_RE = re.compile(r"<\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*>")

# Word and numeric ordinals appearing in PickXtimes, SwingXtimes, VideoRepick, ButtonUnmaskSwap, etc.
_ORDINAL_MAP: Dict[str, int] = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}
_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINAL_MAP.keys()) + r")\b", re.IGNORECASE)

STAGE_VOCAB: Tuple[str, ...] = (
    "none_or_complete",   # 0: "", "complete", "no record"
    "static",             # 1: "static", "remain static"
    "press_button",       # 2: "press the ... button ..."
    "pick_container",     # 3: "pick up the container ..."
    "put_container",      # 4: "put down the container", "put it down"
    "pick_cube",          # 5: "pick up the ... cube ..."
    "place_target",       # 6: "place/drop/put ... onto ... target ..."
    "drop_table",         # 7: "drop/place/put ... onto/on [the] table"
    "bin_fill",           # 8: "put it into the bin ..."
    "pick_peg",           # 9: "pick up the peg ..."
    "insert_peg",         # 10: "insert the peg ..."
    "hook_or_push",       # 11: "hook the cube ..." / "close the gripper and push ..."
    "move_direction",     # 12: PatternLock "move up/down/left/right/..."
    "swing_or_prepare",   # 13: SwingXtimes "move to the top of ..."
    "other",              # 14: fallback (e.g. occluded task-name placeholder)
)
STAGE_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(STAGE_VOCAB)}

COLOR_VOCAB: Tuple[str, ...] = (
    "none",    # 0
    "red",     # 1
    "green",   # 2
    "blue",    # 3
    "yellow",  # 4
    "purple",  # 5
    "orange",  # 6
    "cyan",    # 7
    "pink",    # 8
)
COLOR_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(COLOR_VOCAB)}
_COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_VOCAB[1:]) + r")\b", re.IGNORECASE)

_SENTINEL_EMPTY = frozenset({"", "no record", "none", "null"})


# =============================================================================
# 1. Pixel <-> Normalized Coordinate Conversions (`(x_col, y_row)` Canonical)
# =============================================================================


def pixel_xy_to_norm_xy(
    xy_px: torch.Tensor,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
    align_corners: bool = False,
) -> torch.Tensor:
    """Convert canonical `(x_col, y_row)` pixel coordinates to `[-1, 1]^2` normalized coordinates.

    With `align_corners=False` (default, matching `F.interpolate`/`F.grid_sample` in SigLIP preprocessing):
      `x_norm = ((x_col + 0.5) / W) * 2 - 1`,  `y_norm = ((y_row + 0.5) / H) * 2 - 1`.
    Notice this is strictly invariant under `256x256 -> 224x224` or `8x8` / `14x14` patch pooling!
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image_hw={image_hw}")
    xy = xy_px.float()
    out = torch.empty_like(xy)
    if align_corners:
        out[..., 0] = (xy[..., 0] / max(W - 1, 1)) * 2.0 - 1.0
        out[..., 1] = (xy[..., 1] / max(H - 1, 1)) * 2.0 - 1.0
    else:
        out[..., 0] = ((xy[..., 0] + 0.5) / float(W)) * 2.0 - 1.0
        out[..., 1] = ((xy[..., 1] + 0.5) / float(H)) * 2.0 - 1.0
    return out


def pixel_yx_to_norm_xy(
    yx_px: torch.Tensor,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
    align_corners: bool = False,
) -> torch.Tensor:
    """Convert raw RoboMME `(row_y, col_x)` pixel coordinates directly to canonical `(x_norm, y_norm)`."""
    xy_px = torch.stack([yx_px[..., 1], yx_px[..., 0]], dim=-1)
    return pixel_xy_to_norm_xy(xy_px, image_hw=image_hw, align_corners=align_corners)


def norm_xy_to_pixel_xy(
    xy_norm: torch.Tensor,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
    align_corners: bool = False,
) -> torch.Tensor:
    """Invert `pixel_xy_to_norm_xy`: convert `(x_norm, y_norm)` in `[-1, 1]^2` back to `(x_col, y_row)` px."""
    H, W = int(image_hw[0]), int(image_hw[1])
    xy = xy_norm.float()
    out = torch.empty_like(xy)
    if align_corners:
        out[..., 0] = (xy[..., 0] + 1.0) * 0.5 * max(W - 1, 1)
        out[..., 1] = (xy[..., 1] + 1.0) * 0.5 * max(H - 1, 1)
    else:
        out[..., 0] = (xy[..., 0] + 1.0) * 0.5 * float(W) - 0.5
        out[..., 1] = (xy[..., 1] + 1.0) * 0.5 * float(H) - 0.5
    return out


# =============================================================================
# 2. Grounded Subgoal `<row_y, col_x>` String Parser
# =============================================================================


def parse_single_grounded_subgoal(
    subgoal: Optional[str],
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
) -> List[Tuple[float, float]]:
    """Extract all valid `(x_col, y_row)` pixel pairs from a single RoboMME `grounded_subgoal` string.

    Each `<A, B>` token in RoboMME is `<row_y, col_x>` (`segmentation_utils.py` line 102); this function
    swaps the axis order immediately into canonical `(x_col=B, y_row=A)` and validates bounds `[0, W) x [0, H)`.
    """
    if not subgoal:
        return []
    s_strip = subgoal.strip()
    if s_strip.lower() in _SENTINEL_EMPTY or s_strip.lower() == "complete":
        return []

    H, W = int(image_hw[0]), int(image_hw[1])
    coords_xy: List[Tuple[float, float]] = []
    for m in _COORD_PAIR_RE.finditer(s_strip):
        row_y = float(m.group(1))
        col_x = float(m.group(2))
        if 0.0 <= col_x < float(W) and 0.0 <= row_y < float(H):
            coords_xy.append((col_x, row_y))
    return coords_xy


def resolve_episode_subgoals(subgoals: Sequence[str]) -> List[str]:
    """Resolve trailing `"complete"` strings within a SINGLE episode to the last active non-empty subgoal.

    Matches `robomme_h5_utils.py:resolve_subgoal` while keeping `"NO RECORD"` / `""` untouched.
    """
    resolved: List[str] = []
    last_valid: str = ""
    for raw in subgoals:
        s = (raw or "").strip()
        s_low = s.lower()
        if s_low == "complete":
            resolved.append(last_valid if last_valid else "")
        else:
            if s_low not in _SENTINEL_EMPTY:
                last_valid = s
            resolved.append(s)
    return resolved


def parse_grounded_subgoals(
    subgoals: Union[Sequence[str], Sequence[Sequence[str]]],
    *,
    ep_offsets: Optional[Union[Sequence[int], torch.Tensor, np.ndarray]] = None,
    resolve_complete: bool = False,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
    align_corners: bool = False,
    primary_mode: str = "first",
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Vectorized parser for RoboMME `grounded_subgoals` (flat list OR nested per-episode `list[list[str]]`).

    Args:
        subgoals: Either a flat `Sequence[str]` of length `N` (optionally segmented by `ep_offsets`),
            or a nested `Sequence[Sequence[str]]` as stored in `{TaskName}.pt["grounded_subgoals"]`.
        ep_offsets: Optional `(E + 1,)` episode boundary offsets when `subgoals` is a flat sequence.
        resolve_complete: If True, `"complete"` steps inherit the episode's most recent active subgoal
            (never crossing episode boundaries). Defaults to `False` so terminal `"complete"` steps
            have `has_xy = False` unless explicitly requested.
        image_hw: Native `(H, W)` pixel resolution (`(256, 256)` in RoboMME).
        align_corners: Normalization convention passed to `pixel_xy_to_norm_xy` (default `False`).
        primary_mode: `"first"` (default: immediate manipulation target `<y1, x1>`) or `"last"`
            (destination target `<y2, x2>` on 2-coord `MoveCube` subgoals).

    Returns:
        Dictionary containing:
          - `xy_norm`: `(N, 2)` float32 canonical `(x_norm, y_norm)` in `[-1, 1]^2` (`0.0` if `!has_xy`)
          - `xy_px`: `(N, 2)` float32 canonical `(x_col, y_row)` in `[0, W) x [0, H)` (`0.0` if `!has_xy`)
          - `yx_px_raw`: `(N, 2)` float32 raw `<row_y, col_x>` as written in RoboMME strings
          - `has_xy`: `(N,)` bool mask (`True` iff >= 1 valid `<y, x>` coordinate exists)
          - `secondary_xy_norm`: `(N, 2)` float32 second coordinate (`MoveCube` destination `<y2, x2>`)
          - `secondary_xy_px`: `(N, 2)` float32 second coordinate in `(x_col, y_row)` pixels
          - `has_secondary_xy`: `(N,)` bool mask (`True` iff >= 2 valid coordinates exist)
          - `num_coords`: `(N,)` int64 count of parsed coordinates per step
          - `clean_text`: `List[str]` with `<y, x>` normalized to `<target>`
    """
    if primary_mode not in ("first", "last"):
        raise ValueError(f"primary_mode must be 'first' or 'last', got {primary_mode!r}")

    # Flatten nested `list[list[str]]` per-episode inputs cleanly while respecting episode boundaries
    flat_strings: List[str] = []
    if len(subgoals) > 0 and not isinstance(subgoals[0], str):
        for ep_seq in subgoals:  # type: ignore[assignment]
            ep_list = [str(x) for x in ep_seq]
            flat_strings.extend(resolve_episode_subgoals(ep_list) if resolve_complete else ep_list)
    else:
        str_list = [str(x) for x in subgoals]  # type: ignore[arg-type]
        if resolve_complete:
            if ep_offsets is not None:
                offs = [int(x) for x in (ep_offsets.tolist() if hasattr(ep_offsets, "tolist") else ep_offsets)]
                for i in range(len(offs) - 1):
                    flat_strings.extend(resolve_episode_subgoals(str_list[offs[i] : offs[i + 1]]))
            else:
                flat_strings = resolve_episode_subgoals(str_list)
        else:
            flat_strings = str_list

    N = len(flat_strings)
    xy_px = torch.zeros((N, 2), dtype=torch.float32)
    yx_px_raw = torch.zeros((N, 2), dtype=torch.float32)
    has_xy = torch.zeros((N,), dtype=torch.bool)

    sec_xy_px = torch.zeros((N, 2), dtype=torch.float32)
    has_sec_xy = torch.zeros((N,), dtype=torch.bool)
    num_coords = torch.zeros((N,), dtype=torch.long)
    clean_text: List[str] = []

    for i, s in enumerate(flat_strings):
        pairs = parse_single_grounded_subgoal(s, image_hw=image_hw)
        k = len(pairs)
        num_coords[i] = k
        if k >= 1:
            p_xy = pairs[0] if primary_mode == "first" else pairs[-1]
            xy_px[i, 0] = p_xy[0]
            xy_px[i, 1] = p_xy[1]
            yx_px_raw[i, 0] = p_xy[1]
            yx_px_raw[i, 1] = p_xy[0]
            has_xy[i] = True
        if k >= 2:
            s_xy = pairs[1] if primary_mode == "first" else pairs[0]
            sec_xy_px[i, 0] = s_xy[0]
            sec_xy_px[i, 1] = s_xy[1]
            has_sec_xy[i] = True
        clean_text.append(_COORD_PAIR_RE.sub("<target>", s.strip()))

    xy_norm = pixel_xy_to_norm_xy(xy_px, image_hw=image_hw, align_corners=align_corners)
    xy_norm = torch.where(has_xy.unsqueeze(-1), xy_norm, torch.zeros_like(xy_norm))

    sec_xy_norm = pixel_xy_to_norm_xy(sec_xy_px, image_hw=image_hw, align_corners=align_corners)
    sec_xy_norm = torch.where(has_sec_xy.unsqueeze(-1), sec_xy_norm, torch.zeros_like(sec_xy_norm))

    return {
        "xy_norm": xy_norm,
        "xy_px": xy_px,
        "yx_px_raw": yx_px_raw,
        "has_xy": has_xy,
        "secondary_xy_norm": sec_xy_norm,
        "secondary_xy_px": sec_xy_px,
        "has_secondary_xy": has_sec_xy,
        "num_coords": num_coords,
        "clean_text": clean_text,
    }


# =============================================================================
# 3. `U3` Sub-Grid Affine Augmentation & Exact `U4` Inverse Target Transform
# =============================================================================


def sample_subgrid_crop_affine(
    batch_size: int,
    *,
    scale_range: Tuple[float, float] = (0.90, 1.00),
    max_extra_shift: float = 0.04,
    isotropic: bool = False,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample `(B, 2, 3)` `F.affine_grid` matrices `theta` for `U3` sub-grid random crop augmentation.

    Recall PyTorch `F.affine_grid(theta, ...)` maps output normalized coords `p_out in [-1, 1]^2`
    to source normalized coords `p_src = A @ p_out + t`:
      `x_src = s_x * x_out + t_x`,  `y_src = s_y * y_out + t_y`.
    Here `s_x, s_y in [scale_min, scale_max]` (`s < 1` zooms in slightly) and `t_x, t_y` are sampled
    within `[-(1 - s + max_extra_shift), +(1 - s + max_extra_shift)]`.
    """
    s_min, s_max = float(scale_range[0]), float(scale_range[1])
    if not (0.1 <= s_min <= s_max <= 1.5):
        raise ValueError(f"Invalid scale_range={scale_range}")

    dev = device if device is not None else torch.device("cpu")
    u_scale = torch.rand((batch_size, 1 if isotropic else 2), device=dev, generator=generator, dtype=torch.float32)
    scales = s_min + (s_max - s_min) * u_scale
    if isotropic:
        scales = scales.expand(batch_size, 2)

    max_shift = (1.0 - scales).clamp(min=0.0) + float(max_extra_shift)
    u_shift = (
        torch.rand((batch_size, 2), device=dev, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    )
    shifts = u_shift * max_shift

    theta = torch.zeros((batch_size, 2, 3), device=dev, dtype=torch.float32)
    theta[:, 0, 0] = scales[:, 0]
    theta[:, 1, 1] = scales[:, 1]
    theta[:, 0, 2] = shifts[:, 0]
    theta[:, 1, 2] = shifts[:, 1]
    return theta


def apply_subgrid_affine_to_tokens(
    spatial_tokens: torch.Tensor,
    theta: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    *,
    align_corners: bool = False,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Apply sub-grid bilinear `F.grid_sample` crop to `(B, N_tok, D)` SigLIP patch tokens BEFORE 2D pos emb.

    Supports both `8x8` (`N_tok = 64`, Cycle 2A) and `14x14` (`N_tok = 196`, Cycle 3 `U1`) grids.
    """
    if spatial_tokens.ndim != 3:
        raise ValueError(f"Expected (B, N_tok, D) spatial_tokens, got shape {tuple(spatial_tokens.shape)}")
    B, N_tok, D = spatial_tokens.shape
    if grid_hw is None:
        side = int(round(N_tok**0.5))
        if side * side != N_tok:
            raise ValueError(f"Cannot infer square grid_hw from N_tok={N_tok}; pass grid_hw=(H, W)")
        H_g, W_g = side, side
    else:
        H_g, W_g = int(grid_hw[0]), int(grid_hw[1])
        if H_g * W_g != N_tok:
            raise ValueError(f"grid_hw={grid_hw} does not match N_tok={N_tok}")

    orig_dtype = spatial_tokens.dtype
    feat = spatial_tokens.float().transpose(1, 2).reshape(B, D, H_g, W_g)
    if theta.ndim == 2:
        theta = theta.unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(theta.float(), size=[B, D, H_g, W_g], align_corners=align_corners)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.reshape(B, D, N_tok).transpose(1, 2).contiguous().to(orig_dtype)


def transform_coords_under_affine(
    xy_src_norm: torch.Tensor,
    has_xy: Optional[torch.Tensor],
    theta: torch.Tensor,
    *,
    in_bounds_limit: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform source `(x_norm, y_norm)` targets through the EXACT INVERSE of `F.affine_grid(theta)`.

    Because `F.affine_grid` maps output coords `p_out -> p_src = A @ p_out + t`, a point located at
    `p_src` in the uncropped source image lands in the cropped output image at:
        `p_out = A^{-1} @ (p_src - t)`.
    Using forward `A @ p_src + t` would shift targets in the OPPOSITE direction of the cropped image!

    Args:
        xy_src_norm: `(B, 2)` or `(B, ..., 2)` source normalized coordinates in `[-1, 1]^2`.
        has_xy: Optional boolean mask of shape `xy_src_norm.shape[:-1]`.
        theta: `(B, 2, 3)` or `(2, 3)` `F.affine_grid` matrix `[A | t]`.
        in_bounds_limit: Threshold on `|x_out|, |y_out|` (default `1.0`) to keep `valid_mask = True`;
            targets cropped outside `[-in_bounds_limit, +in_bounds_limit]^2` are masked out!

    Returns:
        `(xy_out_norm, valid_after_crop)`:
          - `xy_out_norm`: same shape as `xy_src_norm` (`float32`), zeroed where `~valid_after_crop`
          - `valid_after_crop`: bool tensor of shape `xy_src_norm.shape[:-1]`
    """
    pts = xy_src_norm.float()
    B = pts.shape[0]
    th = theta.float()
    if th.ndim == 2:
        th = th.unsqueeze(0).expand(B, -1, -1)
    if th.shape != (B, 2, 3):
        raise ValueError(f"Expected theta shape ({B}, 2, 3), got {tuple(th.shape)}")

    # Broadcast `[A | t]` across any intermediate dimensions `(B, ..., 2)`
    expand_shape = (B,) + (1,) * (pts.ndim - 2)
    a00 = th[:, 0, 0].view(*expand_shape)
    a01 = th[:, 0, 1].view(*expand_shape)
    tx = th[:, 0, 2].view(*expand_shape)
    a10 = th[:, 1, 0].view(*expand_shape)
    a11 = th[:, 1, 1].view(*expand_shape)
    ty = th[:, 1, 2].view(*expand_shape)

    dx = pts[..., 0] - tx
    dy = pts[..., 1] - ty
    det = a00 * a11 - a01 * a10
    non_singular = det.abs() > 1e-8
    safe_det = torch.where(non_singular, det, torch.ones_like(det))

    x_out = (a11 * dx - a01 * dy) / safe_det
    y_out = (-a10 * dx + a00 * dy) / safe_det
    xy_out = torch.stack([x_out, y_out], dim=-1)

    in_bounds = non_singular & (x_out.abs() <= float(in_bounds_limit)) & (y_out.abs() <= float(in_bounds_limit))
    if has_xy is not None:
        valid = has_xy.bool() & in_bounds
    else:
        valid = in_bounds

    xy_out = torch.where(valid.unsqueeze(-1), xy_out, torch.zeros_like(xy_out))
    return xy_out, valid


# =============================================================================
# 4. 2D SpatialSoftmax Helper (`align_corners=False` Exact Match with Parser)
# =============================================================================


def spatial_softmax_2d(
    patch_logits: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    *,
    temperature: float = 1.0,
    align_corners: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute expected `(x_norm, y_norm)` in `[-1, 1]^2` from patch logits `(..., N_tok)` or `(..., H_g, W_g)`.

    Uses the exact same `align_corners=False` patch-center convention `((c + 0.5) / W_g) * 2 - 1` as
    `pixel_xy_to_norm_xy` and `F.grid_sample`, ensuring zero discretization bias across `256 -> 224 -> 14x14 / 8x8`!
    Returns `(xy_pred_norm, attn_probs_flat)`.
    """
    if patch_logits.ndim >= 2 and grid_hw is None and patch_logits.shape[-1] in (64, 196, 256):
        N_tok = patch_logits.shape[-1]
        side = int(round(N_tok**0.5))
        H_g, W_g = side, side
        flat_logits = patch_logits.float()
    elif grid_hw is not None:
        H_g, W_g = int(grid_hw[0]), int(grid_hw[1])
        if patch_logits.shape[-2:] == (H_g, W_g):
            flat_logits = patch_logits.float().reshape(*patch_logits.shape[:-2], H_g * W_g)
        elif patch_logits.shape[-1] == H_g * W_g:
            flat_logits = patch_logits.float()
        else:
            raise ValueError(f"patch_logits shape {tuple(patch_logits.shape)} incompatible with grid_hw={grid_hw}")
    else:
        raise ValueError(f"Specify grid_hw=(H, W) for patch_logits shape {tuple(patch_logits.shape)}")

    dev = flat_logits.device
    cols = torch.arange(W_g, device=dev, dtype=torch.float32)
    rows = torch.arange(H_g, device=dev, dtype=torch.float32)
    if align_corners:
        x_centers = (cols / max(W_g - 1, 1)) * 2.0 - 1.0
        y_centers = (rows / max(H_g - 1, 1)) * 2.0 - 1.0
    else:
        x_centers = ((cols + 0.5) / float(W_g)) * 2.0 - 1.0
        y_centers = ((rows + 0.5) / float(H_g)) * 2.0 - 1.0

    yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
    grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (H_g * W_g, 2)

    probs = F.softmax(flat_logits / max(float(temperature), 1e-6), dim=-1)
    xy_pred = torch.matmul(probs, grid_xy)
    return xy_pred, probs


# =============================================================================
# 5. Subgoal Stage, Ordinal Count, Color, & Transition Metadata Extractor
# =============================================================================


def classify_subgoal_stage(text: Optional[str]) -> str:
    """Map a RoboMME `simple_subgoal` or `grounded_subgoal` string to a canonical `STAGE_VOCAB` name."""
    if not text:
        return "none_or_complete"
    s = text.strip().lower()
    if s in _SENTINEL_EMPTY or s == "complete":
        return "none_or_complete"
    if s in ("static", "remain static"):
        return "static"
    if "button" in s and "press" in s:
        return "press_button"
    if "move to the top of" in s:
        return "swing_or_prepare"
    if "container" in s and "pick" in s:
        return "pick_container"
    if ("container" in s and "put" in s) or s == "put it down":
        return "put_container"
    if "into the bin" in s:
        return "bin_fill"
    if "insert the peg" in s:
        return "insert_peg"
    if "peg" in s and "pick" in s:
        return "pick_peg"
    if s.startswith("hook ") or ("push the cube" in s):
        return "hook_or_push"
    if s.startswith("move ") and not s.startswith("move to"):
        return "move_direction"
    if "table" in s and any(v in s for v in ("drop", "place", "put")):
        return "drop_table"
    if "target" in s and any(v in s for v in ("place", "drop", "put")):
        return "place_target"
    if "cube" in s and "pick" in s:
        return "pick_cube"
    return "other"


def parse_ordinal_count(text: Optional[str]) -> int:
    """Extract 1-based ordinal index (`1..10`) from subgoal text (`'first'`, `'for the 2nd time'`, etc.), else `0`."""
    if not text:
        return 0
    matches = _ORDINAL_RE.findall(text)
    if not matches:
        return 0
    # Prefer the last ordinal in the clause (e.g., "for the third time" -> 3)
    return _ORDINAL_MAP.get(matches[-1].lower(), 0)


def parse_color_id(text: Optional[str]) -> int:
    """Extract canonical `COLOR_VOCAB` id (`1..8`) if a target cube color is mentioned, else `0`."""
    if not text:
        return 0
    m = _COLOR_RE.search(text)
    if not m:
        return 0
    return COLOR_TO_ID.get(m.group(1).lower(), 0)


def extract_subgoal_metadata(
    simple_subgoals: Union[Sequence[str], Sequence[Sequence[str]]],
    *,
    ep_offsets: Optional[Union[Sequence[int], torch.Tensor, np.ndarray]] = None,
    exec_start_idx: Optional[Union[Sequence[int], torch.Tensor, np.ndarray]] = None,
) -> Dict[str, torch.Tensor]:
    """Extract per-frame count, stage, color, and subgoal-transition metadata for Probes 1-3 & `U4`.

    Handles both nested per-episode `Sequence[Sequence[str]]` (as in `{TaskName}.pt`) and flat `Sequence[str]`.

    Returns dict of `(N,)` tensors:
      - `ordinal_index`: `int64` (`1..10` when explicit ordinal word/number present, else `0`)
      - `stage_id`: `int64` index into `STAGE_VOCAB`
      - `color_id`: `int64` index into `COLOR_VOCAB`
      - `is_transition`: `bool` (`True` when active subgoal string changes within an episode)
      - `completed_transitions`: `int64` cumulative count of subgoal transitions in execution phase
      - `completed_picks`: `int64` cumulative count of distinct pick segments (`pick_cube`/`pick_container`/`pick_peg`)
        entered during execution (`0, 1, 2, ...` — works uniformly across `PickXtimes`, `VideoRepick`, `BinFill`, etc.)
    """
    episodes: List[List[str]] = []
    if len(simple_subgoals) > 0 and not isinstance(simple_subgoals[0], str):
        episodes = [[str(x) for x in ep] for ep in simple_subgoals]  # type: ignore[assignment]
    else:
        flat = [str(x) for x in simple_subgoals]  # type: ignore[arg-type]
        if ep_offsets is not None:
            offs = [int(x) for x in (ep_offsets.tolist() if hasattr(ep_offsets, "tolist") else ep_offsets)]
            episodes = [flat[offs[e] : offs[e + 1]] for e in range(len(offs) - 1)]
        else:
            episodes = [flat]

    exec_starts: List[int]
    if exec_start_idx is not None:
        raw_es = exec_start_idx.tolist() if hasattr(exec_start_idx, "tolist") else list(exec_start_idx)
        exec_starts = [int(x) for x in raw_es]
    else:
        exec_starts = [0] * len(episodes)

    total_n = sum(len(ep) for ep in episodes)
    ordinal_index = torch.zeros((total_n,), dtype=torch.long)
    stage_id = torch.zeros((total_n,), dtype=torch.long)
    color_id = torch.zeros((total_n,), dtype=torch.long)
    is_transition = torch.zeros((total_n,), dtype=torch.bool)
    completed_transitions = torch.zeros((total_n,), dtype=torch.long)
    completed_picks = torch.zeros((total_n,), dtype=torch.long)

    cursor = 0
    pick_stage_ids = {
        STAGE_TO_ID["pick_cube"],
        STAGE_TO_ID["pick_container"],
        STAGE_TO_ID["pick_peg"],
    }
    for ep_i, ep_strs in enumerate(episodes):
        ex_start = exec_starts[ep_i] if ep_i < len(exec_starts) else 0
        prev_norm_str: Optional[str] = None
        trans_cnt = 0
        pick_cnt = 0
        in_pick_segment = False

        for local_t, s in enumerate(ep_strs):
            idx = cursor + local_t
            s_clean = s.strip()
            s_low = s_clean.lower()
            st_name = classify_subgoal_stage(s_clean)
            st_id = STAGE_TO_ID[st_name]

            stage_id[idx] = st_id
            ordinal_index[idx] = parse_ordinal_count(s_clean)
            color_id[idx] = parse_color_id(s_clean)

            if local_t >= ex_start:
                is_non_empty = s_low not in _SENTINEL_EMPTY and s_low != "complete"
                if prev_norm_str is not None and is_non_empty and s_low != prev_norm_str:
                    is_transition[idx] = True
                    trans_cnt += 1
                if is_non_empty:
                    prev_norm_str = s_low

                if st_id in pick_stage_ids:
                    if not in_pick_segment:
                        pick_cnt += 1
                        in_pick_segment = True
                else:
                    in_pick_segment = False

                completed_transitions[idx] = trans_cnt
                completed_picks[idx] = pick_cnt

        cursor += len(ep_strs)

    return {
        "ordinal_index": ordinal_index,
        "stage_id": stage_id,
        "color_id": color_id,
        "is_transition": is_transition,
        "completed_transitions": completed_transitions,
        "completed_picks": completed_picks,
    }


# =============================================================================
# 6. Self-Contained CPU Unit Test Suite (`python -m analysis.subgoal_grounding`)
# =============================================================================


def run_unit_tests() -> None:
    """Thorough CPU unit tests verifying axis order, multi-coord parsing, and exact `F.grid_sample` math."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Exact reproduction of `segmentation_utils.py` (`np.argwhere(mask)` -> `<center_y, center_x>`)
    mask = np.zeros((256, 256), dtype=bool)
    # Place an object near TOP-RIGHT of image: rows `y in [20..28]`, cols `x in [200..212]`
    mask[20:29, 200:213] = True
    coords = np.argwhere(mask)
    center_y = int(coords[:, 0].mean())  # 24 (near top -> y_norm < 0)
    center_x = int(coords[:, 1].mean())  # 206 (near right -> x_norm > 0)
    assert (center_y, center_x) == (24, 206)
    sim_subgoal = f"pick up the red cube at <{center_y}, {center_x}> for the third time"

    parsed = parse_grounded_subgoals([sim_subgoal])
    assert bool(parsed["has_xy"][0]) is True
    # Canonical `xy_px` MUST be `(col_x=206, row_y=24)`, NEVER `(24, 206)`!
    assert torch.allclose(parsed["xy_px"][0], torch.tensor([206.0, 24.0]))
    assert torch.allclose(parsed["yx_px_raw"][0], torch.tensor([24.0, 206.0]))
    # Top-right means `x_norm > +0.5` and `y_norm < -0.7`!
    xy_n = parsed["xy_norm"][0]
    assert float(xy_n[0]) > 0.5 and float(xy_n[1]) < -0.7, f"Axis flip detected: xy_norm={xy_n.tolist()}"

    # Round-trip pixel <-> norm invariance
    rt_px = norm_xy_to_pixel_xy(parsed["xy_norm"])
    assert torch.allclose(rt_px[0], torch.tensor([206.0, 24.0]), atol=1e-5)

    # 2. Zeropatch-bias identity between `256x256` pixel centers and `8x8` / `14x14` SpatialSoftmax patch centers
    for grid_side in (8, 16):
        patch_size = 256 // grid_side
        r_idx, c_idx = 2, grid_side - 2
        ys, xs = np.meshgrid(
            np.arange(r_idx * patch_size, (r_idx + 1) * patch_size),
            np.arange(c_idx * patch_size, (c_idx + 1) * patch_size),
            indexing="ij",
        )
        mean_xy_px = torch.tensor([[float(xs.mean()), float(ys.mean())]])
        target_norm = pixel_xy_to_norm_xy(mean_xy_px, image_hw=(256, 256), align_corners=False)

        logits = torch.full((1, grid_side, grid_side), -1e9)
        logits[0, r_idx, c_idx] = 100.0
        pred_norm, _ = spatial_softmax_2d(logits, grid_hw=(grid_side, grid_side), align_corners=False)
        assert torch.allclose(pred_norm, target_norm, atol=1e-6), (
            f"Patch center mismatch on {grid_side}x{grid_side}: {pred_norm} vs {target_norm}"
        )

    # 3. Multi-coord (`MoveCube`), 0-coord (`static`, `NO RECORD`, `put down`), & per-episode `complete`
    ep1 = [
        "NO RECORD",
        "hook the cube at <140, 80> to the target at <110, 210> with the peg",
        "static",
        "complete",
    ]
    ep2 = [
        "complete",  # Must NOT leak from ep1!
        "press the second button at <64, 32> to stop",
    ]
    p_raw = parse_grounded_subgoals([ep1, ep2], resolve_complete=False)
    assert p_raw["has_xy"].tolist() == [False, True, False, False, False, True]
    assert p_raw["has_secondary_xy"].tolist() == [False, True, False, False, False, False]
    assert p_raw["num_coords"].tolist() == [0, 2, 0, 0, 0, 1]
    assert torch.allclose(p_raw["xy_px"][1], torch.tensor([80.0, 140.0]))
    assert torch.allclose(p_raw["secondary_xy_px"][1], torch.tensor([210.0, 110.0]))

    p_res = parse_grounded_subgoals([ep1, ep2], resolve_complete=True)
    # In ep1, `complete` follows `static` (which has no `<y, x>`), while ep2[0] `complete` has no prior valid step
    assert p_res["has_xy"].tolist() == [False, True, False, False, False, True]
    ep3 = [["pick up the blue cube at <120, 50>", "complete"]]
    p_res3 = parse_grounded_subgoals(ep3, resolve_complete=True)
    assert p_res3["has_xy"].tolist() == [True, True]
    assert torch.allclose(p_res3["xy_px"][1], torch.tensor([50.0, 120.0]))

    # 4. CRITICAL TEST: `apply_subgrid_affine_to_tokens` + `transform_coords_under_affine` exact alignment!
    # Create sharp Gaussian peaks on a 64x64 grid (4096 tokens), warp with random scale + shift + rotation/shear,
    # decode new peak location via `spatial_softmax_2d`, and verify `transform_coords_under_affine` matches to < 0.5 px!
    B_test = 16
    H_g, W_g = 64, 64
    gen = torch.Generator().manual_seed(123)
    # Keep peaks in interior `[-0.55, 0.55]^2` so random crops keep most inside `[-1, 1]^2`
    src_xy = (torch.rand((B_test, 2), generator=gen) * 1.1) - 0.55

    cols = ((torch.arange(W_g, dtype=torch.float32) + 0.5) / W_g) * 2.0 - 1.0
    rows = ((torch.arange(H_g, dtype=torch.float32) + 0.5) / H_g) * 2.0 - 1.0
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    grid_pts = torch.stack([xx, yy], dim=-1).view(1, H_g * W_g, 2)  # (1, 4096, 2)

    # Use smooth Gaussian log-potentials (`sigma = 3` grid cells) so bilinear `grid_sample` +
    # `spatial_softmax_2d(temperature=1.0)` recovers exact sub-cell continuous center-of-mass
    # without discrete 4.0px cell-center argmax quantization!
    sigma2 = (6.0 / H_g) ** 2
    dist2 = ((grid_pts - src_xy.unsqueeze(1)) ** 2).sum(dim=-1, keepdim=True)  # (B, 4096, 1)
    log_potentials = -0.5 * dist2 / sigma2  # (B, 4096, 1)

    theta = sample_subgrid_crop_affine(
        B_test, scale_range=(0.90, 0.98), max_extra_shift=0.05, generator=gen
    )
    # Also inject slight off-diagonal shear/rotation on half the batch to test full 2x2 matrix inversion!
    theta[8:, 0, 1] = 0.04
    theta[8:, 1, 0] = -0.03

    warped_logits = apply_subgrid_affine_to_tokens(
        log_potentials, theta, grid_hw=(H_g, H_g), padding_mode="border"
    )  # (B, 4096, 1)
    expected_xy, valid_crop = transform_coords_under_affine(
        src_xy, torch.ones((B_test,), dtype=torch.bool), theta, in_bounds_limit=0.80
    )
    assert int(valid_crop.sum().item()) >= 10

    measured_xy, _ = spatial_softmax_2d(
        warped_logits.squeeze(-1), grid_hw=(H_g, H_g), temperature=1.0, align_corners=False
    )
    err_norm = (measured_xy[valid_crop] - expected_xy[valid_crop]).norm(dim=-1)
    err_px_256 = err_norm * 128.0
    max_err_px = float(err_px_256.max().item())
    assert max_err_px < 0.15, f"Affine coordinate transform mismatch: max_err_px={max_err_px:.4f}"

    # Sanity-check that the buggy forward transform `A @ p + t` would give huge (> 10 px) error!
    buggy_forward_xy = torch.einsum("bij,bj->bi", theta[:, :, :2], src_xy) + theta[:, :, 2]
    buggy_err_px = float(((measured_xy[valid_crop] - buggy_forward_xy[valid_crop]).norm(dim=-1) * 128.0).max().item())
    assert buggy_err_px > 10.0, f"Expected buggy forward affine to fail loudly, got {buggy_err_px:.4f}px"

    # Also test explicit out-of-bounds crop masking: large shift pushes center outside `[-1, 1]^2`
    theta_oob = torch.tensor([[[0.5, 0.0, 0.85], [0.0, 0.5, 0.0]]])  # p_src=(0,0) -> x_out = -1.7 (OOB!)
    xy_oob, val_oob = transform_coords_under_affine(
        torch.zeros((1, 2)), torch.ones((1,), dtype=torch.bool), theta_oob
    )
    assert bool(val_oob[0]) is False
    assert torch.all(xy_oob == 0.0)

    # 5. Metadata extraction (`ordinal_index`, `stage_id`, `color_id`, `completed_picks`)
    meta = extract_subgoal_metadata(
        [
            [
                "NO RECORD",                                        # demo frame 0
                "pick up the green cube for the first time",        # exec 1 -> pick 1
                "pick up the green cube for the first time",        # exec 2 -> pick 1
                "put down",                                         # exec 3 -> other/trans
                "pick up the green cube for the second time",       # exec 4 -> pick 2
                "place the cube onto the target",                   # exec 5 -> place_target
            ]
        ],
        exec_start_idx=[1],
    )
    assert meta["ordinal_index"].tolist() == [0, 1, 1, 0, 2, 0]
    assert meta["color_id"].tolist() == [0, COLOR_TO_ID["green"], COLOR_TO_ID["green"], 0, COLOR_TO_ID["green"], 0]
    assert meta["completed_picks"].tolist() == [0, 1, 1, 1, 2, 2]
    assert meta["is_transition"].tolist() == [False, False, False, True, True, True]

    print(
        "[PASS] analysis/subgoal_grounding.py: ALL unit tests passed "
        f"(axis `<row_y, col_x> -> (x_col, y_row)`, 0/1/2-coord edge cases, SpatialSoftmax 0.0px bias, "
        f"and F.grid_sample affine peak tracking max_err={max_err_px:.4f} px on 256x256)!"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify RoboMME grounded subgoal parser & affine math.")
    parser.add_argument("--self-test", action="store_true", default=True, help="Run CPU unit test suite.")
    _ = parser.parse_args()
    run_unit_tests()


if __name__ == "__main__":
    _main()
