"""`CORM` (`src/pi05_mem/corm.py`) — Causal Object-Centric Reveal & Transport Memory for `DeltaMem-π0.5 v3`.

Strictly zero-oracle, strictly causal (`t' <= t`), `k = 3` multi-session safe working memory operating
exclusively on non-privileged public policy inputs `(front_rgb, raw_state, task_goal, is_demo)`:

  1. **Container Reveal & Seat-Occupancy Swap Transport (`Permanence`):**
     - Captures pre-drop (`t < t_drop`) revealed cube chroma (`red`, `green`, `blue`) vs. post-drop settled
       container appearance (`f_occ`), plus any empty grey containers (`is_wood(f_rev) & ~is_wood(f_occ)`),
       eliminating Phase-2's `||F_t - F_0||_2 == 0` reveal-inversion bug.
     - Tracks `swap_flat_two_lane` container swaps via simultaneous multi-seat wood-tabletop exposure
       (`n_vac >= 2`), including back-to-back shared-container swaps (`swap_frames_count >= 46` valley split),
       while remaining 100% invariant to single-container execution pickups (`n_vac == 1`) and moving robot arms
       (whose metallic/white links have `is_wood == False`).
  2. **Transient Illumination Memory (`PickHighlight`):**
     - Anchors unoccluded tabletop cubes `(y, x, color)` at session start and integrates persistent white-area
       brightness jumps (`mean(F_t - F_0)`) across the transient highlight window, masking completed execution picks.
  3. **Proprioceptive-Visual Demo Grasp & Swap Transport (`VideoRepick`):**
     - Segments demo cube grasps via physical Panda finger blockage (`0.010 <= min_grip <= 0.028` m vs. `0.040` m open),
       identifies the grasped tabletop cube via wood-tabletop exposure during lift (`is_wood` jumps `0 -> 1`, rejecting
       arm-occlusion shadows where `is_wood == 0`), and transports its identity across post-grasp swaps (`n_vac >= 2`).
  4. **Chronological Demo Target-Pad Contact, Button Timestamping, & Pad-Swap Transport (`VideoPlaceButton`, `VideoPlaceOrder`):**
     - Distinguishes demo cube placements (`min_grip >= 0.010` m) from demo button presses (`min_grip < 0.008` m),
       identifies which purple target pad received each placement at the arm-retracted midpoint `(rise_k + fall_{k+1}) // 2`,
       and transports pad identities across any end-of-demo target-pad swaps (`n_vac >= 2`).
  5. **Continuous `6 x 64 x 64` Evidence Maps + Sub-Pixel `<row_y, col_x>` Refinement & Proprioceptive Stage Guard:**
     - Emits both `6 x 64 x 64` spatial maps for `VLMGroundedPointerCoda` (`DeltaGround-VLM v3`) and sub-pixel
       `[0, 255]^2` `<row_y, col_x>` target coordinates, resetting strictly on non-privileged `EpisodeBoundaryForgetGate`
       signals (`exec -> demo`, `task_goal` change, or full-tabletop teleportation cut; NEVER `step_idx == 0`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from analysis.subgoal_grounding import ROBOMME_IMAGE_HW, pixel_xy_to_norm_xy

_ORDINAL_MAP: Dict[str, int] = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}


def extract_goal_target_colors(task_goal: str) -> List[str]:
    """Extract ordered target cube colors from public `task_goal` string (`1` or `2` colors)."""
    g = (task_goal or "").strip().lower()
    out: List[str] = []
    for part in g.split("finally"):
        for col in ("red", "green", "blue"):
            if f"{col} cube" in part or f"{col} block" in part:
                out.append(col)
                break
    if not out:
        for col in ("red", "green", "blue"):
            if col in g:
                out.append(col)
    return out


def is_wood_tabletop_mask(frame: torch.Tensor) -> torch.Tensor:
    """Return `(H, W)` bool mask of exposed warm-wood tabletop (`~[191, 114, 71]`).

    Disjoint from all RoboMME objects:
      - Red cube `[231, 5, 5]`: `G < 50` (rejected by `G > 72`).
      - Green/Blue cubes, Purple pads, Grey/White containers, White/Dark robot arm: `R - B <= 25` (rejected by `R - B > 55`).
    """
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return ((r - b) > 55.0) & (g > 72.0) & ((r - g) < 120.0)


def _detect_local_peaks_2d(
    score_map: torch.Tensor,  # (H, W) float32
    *,
    thresh: float,
    min_dist: float,
    max_peaks: int = 20,
    y_bounds: Tuple[int, int] = (46, 218),
    x_bounds: Tuple[int, int] = (20, 236),
) -> List[Tuple[float, float, float]]:
    """Greedy 2D NMS returning `[(y, x, val), ...]` with `5x5` local center-of-mass sub-pixel refinement."""
    H, W = int(score_map.shape[0]), int(score_map.shape[1])
    work = score_map.clone()
    work[: max(0, y_bounds[0]), :] = 0.0
    work[min(H, y_bounds[1]) :, :] = 0.0
    work[:, : max(0, x_bounds[0])] = 0.0
    work[:, min(W, x_bounds[1]) :] = 0.0

    yy, xx = torch.meshgrid(
        torch.arange(H, device=score_map.device, dtype=torch.float32),
        torch.arange(W, device=score_map.device, dtype=torch.float32),
        indexing="ij",
    )
    r2 = float(min_dist) ** 2
    peaks: List[Tuple[float, float, float]] = []
    for _ in range(max_peaks):
        val = float(work.max().item())
        if val < float(thresh):
            break
        idx = int(work.argmax().item())
        py, px = idx // W, idx % W
        y0, y1 = max(0, py - 2), min(H, py + 3)
        x0, x1 = max(0, px - 2), min(W, px + 3)
        patch_w = F.relu(score_map[y0:y1, x0:x1])
        w_sum = float(patch_w.sum().item())
        if w_sum > 1e-6:
            sub_y = float((patch_w * yy[y0:y1, x0:x1]).sum().item() / w_sum)
            sub_x = float((patch_w * xx[y0:y1, x0:x1]).sum().item() / w_sum)
        else:
            sub_y, sub_x = float(py), float(px)
        peaks.append((sub_y, sub_x, val))
        work = work.masked_fill(((yy - sub_y).pow(2) + (xx - sub_x).pow(2)) < r2, 0.0)
    return peaks


def _classify_rgb_color(rgb: torch.Tensor) -> str:
    """Classify a `3`-vector `[R, G, B]` into `'red' | 'green' | 'blue'`."""
    r, g, b = float(rgb[0].item()), float(rgb[1].item()), float(rgb[2].item())
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


class CausalObjectCentricMemory:
    """Zero-oracle causal spatiotemporal object memory (`CORM`) over `(front_rgb, raw_state, task_goal)`."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.reset()

    def to_device(self, dev: str | torch.device) -> None:
        self.device = torch.device(dev)
        self._demo_frames = [t.to(self.device) for t in self._demo_frames]
        self._exec_early_frames = [t.to(self.device) for t in self._exec_early_frames]
        if self._last_frame is not None:
            self._last_frame = self._last_frame.to(self.device)
        if self._ev_maps_64 is not None:
            self._ev_maps_64 = self._ev_maps_64.to(self.device)
        if getattr(self, "_hl_wood0", None) is not None:
            self._hl_wood0 = self._hl_wood0.to(self.device)  # type: ignore[union-attr]
        if getattr(self, "_hl_nearest_m", None) is not None:
            self._hl_nearest_m = self._hl_nearest_m.to(self.device)  # type: ignore[union-attr]
        if getattr(self, "_hl_d2", None) is not None:
            self._hl_d2 = self._hl_d2.to(self.device)  # type: ignore[union-attr]

    def reset(self) -> None:
        """Clear all episode state (called on explicit `force_reset` or non-privileged boundary detection)."""
        self._last_goal: Optional[str] = None
        self._last_is_demo: Optional[bool] = None
        self._last_frame: Optional[torch.Tensor] = None
        self._last_state: Optional[np.ndarray] = None
        self._demo_finalized: bool = False

        self._demo_frames: List[torch.Tensor] = []
        self._demo_states: List[np.ndarray] = []
        self._exec_frames_count: int = 0
        self._exec_early_frames: List[torch.Tensor] = []

        # 1) Permanence state (`VideoUnmask`, `ButtonUnmask`, `VideoUnmaskSwap`, `ButtonUnmaskSwap`)
        self.perm_ready: bool = False
        self.perm_t_settle: int = 0
        self.perm_seats: List[Tuple[float, float]] = []
        self.perm_colors: List[str] = []
        self._perm_base_wood: List[float] = []
        self._perm_in_swap: bool = False
        self._perm_swap_count: int = 0
        self._perm_prev_n_vac: int = 0
        self._perm_vac_accum: List[float] = []
        self._perm_exec_pick_started: bool = False

        # 2) Highlight state (`PickHighlight`)
        self.hl_cubes: List[Tuple[float, float, str]] = []
        self.hl_scores: List[float] = []
        self.hl_picked: List[bool] = []
        self.hl_ready: bool = False
        self._hl_wood0: Optional[torch.Tensor] = None
        self._hl_nearest_m: Optional[torch.Tensor] = None
        self._hl_d2: Optional[torch.Tensor] = None

        # 3) Demo Repick state (`VideoRepick`)
        self.repick_ready: bool = False
        self.repick_yx: Optional[Tuple[float, float]] = None
        self.repick_color: Optional[str] = None

        # 4) Demo PlaceButton / PlaceOrder state (`VideoPlaceButton`, `VideoPlaceOrder`)
        self.place_ready: bool = False
        self.place_cube_yx: Optional[Tuple[float, float]] = None
        self.place_target_yx: Optional[Tuple[float, float]] = None

        # Single-button t=0 cap coordinate (`ButtonUnmask`, `PickHighlight`, `VideoRepick`)
        self.single_button_yx: Optional[Tuple[float, float]] = None

        # Execution gripper pick-place cycle tracking (100% proprioceptive `raw_state[7]`)
        self._exec_grip_closed: bool = False
        self._exec_grip_min: float = 0.04
        self._exec_grip_hold_frames: int = 0
        self._exec_completed_object_picks: int = 0
        self._exec_completed_button_presses: int = 0
        self._exec_grip_zero_streak: int = 0
        self._exec_frames_since_release: int = -1
        self._exec_active_grasp_yx: Optional[Tuple[float, float]] = None

        self._ev_maps_64: Optional[torch.Tensor] = None

    def _reset_episode_working_memory(self, goal_clean: str, is_demo: bool) -> None:
        self.reset()
        self._last_goal = goal_clean
        self._last_is_demo = bool(is_demo)

    def _check_non_privileged_boundary(
        self,
        ft: torch.Tensor,
        st: np.ndarray,
        goal_clean: str,
        is_demo: bool,
        explicit_boundary: bool = False,
    ) -> bool:
        """Detect episode boundaries strictly without `step_idx == 0` (`k = 3` un-reset streaming safe!)."""
        if explicit_boundary:
            return True
        if self._last_is_demo is False and bool(is_demo):
            return True
        if self._last_goal is not None and goal_clean != self._last_goal:
            return True
        if (
            not is_demo
            and self._last_is_demo is False
            and self._exec_frames_count >= 16
            and self._last_frame is not None
            and self._last_state is not None
        ):
            dq_max = float(np.max(np.abs(st[:7] - self._last_state[:7])))
            if dq_max > 0.35:
                px_jump = float(
                    ((ft[70:210, 30:226, :] - self._last_frame[70:210, 30:226, :]).abs().mean(dim=-1) > 28.0)
                    .float()
                    .mean()
                    .item()
                )
                if px_jump > 0.35:
                    return True
        return False

    @torch.no_grad()
    def ingest_raw_stream(
        self,
        front_rgb_seq: Sequence[np.ndarray] | np.ndarray,
        raw_states_seq: Sequence[np.ndarray] | np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
        explicit_boundary: bool = False,
    ) -> None:
        """Ingest `1` or `T` raw `uint8` `front_rgb` frames + `8-D` `raw_states` causally."""
        if isinstance(front_rgb_seq, np.ndarray) and front_rgb_seq.ndim == 3:
            rgb_np = front_rgb_seq[None, ...]
        else:
            if len(front_rgb_seq) == 0:
                return
            rgb_np = np.stack([np.asarray(x, dtype=np.uint8) for x in front_rgb_seq], axis=0)

        st_np = np.asarray(raw_states_seq, dtype=np.float32)
        if st_np.ndim == 1:
            st_np = st_np[None, :]

        goal_clean = " ".join((task_goal or "").strip().lower().split())
        total_t = int(rgb_np.shape[0])
        for start_i in range(0, total_t, 32):
            end_i = min(start_i + 32, total_t)
            t_slice = torch.from_numpy(np.ascontiguousarray(rgb_np[start_i:end_i])).to(
                device=self.device, dtype=torch.float32
            )
            for j in range(end_i - start_i):
                i = start_i + j
                ft = t_slice[j]
                st = st_np[min(i, len(st_np) - 1)]
                if self._check_non_privileged_boundary(
                    ft, st, goal_clean, is_demo=bool(is_demo), explicit_boundary=(explicit_boundary and i == 0)
                ):
                    self._reset_episode_working_memory(goal_clean, bool(is_demo))

                if self._last_goal is None:
                    self._last_goal = goal_clean

                if self._last_is_demo is True and (not bool(is_demo)) and (not self._demo_finalized):
                    self._finalize_demo_analyses(ft, goal_clean)
                    self._demo_finalized = True

                self._last_is_demo = bool(is_demo)
                if is_demo:
                    self._step_demo_frame(ft, st, goal_clean)
                else:
                    self._step_exec_frame(ft, st, goal_clean)

                self._last_frame = ft.clone()
                self._last_state = st.copy()
            del t_slice

        self._ev_maps_64 = None

    # -------------------------------------------------------------------------
    # Internal Causal Trackers
    # -------------------------------------------------------------------------

    def _is_permanence_goal(self, goal_clean: str) -> bool:
        return ("container hiding" in goal_clean) or ("container that hides" in goal_clean)

    def _is_highlight_goal(self, goal_clean: str) -> bool:
        return "highlight" in goal_clean

    def _is_repick_goal(self, goal_clean: str) -> bool:
        return "same block" in goal_clean

    def _is_place_ref_goal(self, goal_clean: str) -> bool:
        return ("button was pressed" in goal_clean) or ("target it was previously placed on" in goal_clean)

    @staticmethod
    def _step_wood_vacation_swap(
        ft: torch.Tensor,
        seats: Sequence[Tuple[float, float]],
        base_wood: Sequence[float],
        labels: List[str],
        *,
        in_swap: bool,
        swap_count: int,
        prev_n_vac: int,
        vac_accum: List[float],
        wood_jump_thresh: float = 0.40,
        patch_half: int = 6,
    ) -> Tuple[bool, int, int]:
        """Advance simultaneous 2-seat wood-tabletop exposure swap state machine by 1 frame."""
        if len(seats) < 2:
            return in_swap, swap_count, prev_n_vac
        H, W, _ = ft.shape
        wood_mask = is_wood_tabletop_mask(ft)
        vac_scores: List[float] = []
        for k, (sy, sx) in enumerate(seats):
            iy, ix = int(round(sy)), int(round(sx))
            wf = float(
                wood_mask[
                    max(0, iy - patch_half) : min(H, iy + patch_half + 1),
                    max(0, ix - patch_half) : min(W, ix + patch_half + 1),
                ]
                .float()
                .mean()
                .item()
            )
            vac_scores.append(1.0 if (wf - float(base_wood[k])) > wood_jump_thresh else 0.0)

        n_vac = sum(1 for v in vac_scores if v > 0.5)
        if not in_swap:
            if n_vac >= 2:
                in_swap = True
                swap_count = 1
                for k in range(len(seats)):
                    vac_accum[k] = vac_scores[k]
        else:
            swap_count += 1
            for k in range(len(seats)):
                vac_accum[k] += vac_scores[k]
            valley_split = swap_count >= 45 and n_vac <= 1 and n_vac < prev_n_vac
            if (n_vac == 0 and swap_count >= 10) or valley_split:
                ranked = sorted(range(len(seats)), key=lambda k: vac_accum[k], reverse=True)
                if vac_accum[ranked[0]] >= 6.0 and vac_accum[ranked[1]] >= 6.0:
                    i1, i2 = ranked[0], ranked[1]
                    labels[i1], labels[i2] = labels[i2], labels[i1]
                in_swap = False
                swap_count = 0
                for k in range(len(seats)):
                    vac_accum[k] = 0.0
        return in_swap, swap_count, n_vac

    @torch.no_grad()
    def _init_permanence_from_window(self, frames: List[torch.Tensor]) -> bool:
        """Initialize container seats (`red`/`green`/`blue` + empty grey containers) from `t=0..31` lift -> `t=32` drop."""
        T = len(frames)
        if T < 36:
            return False
        stack = torch.stack(frames[: min(T, 40)], dim=0)
        H, W, _ = stack[0].shape
        d_step = (stack[1:, 65:200, 40:216, :] - stack[:-1, 65:200, 40:216, :]).abs().mean(dim=(-1, -2, -3))
        d_sub = d_step[15:36] if int(d_step.numel()) >= 36 else d_step
        off_t = 15 if int(d_step.numel()) >= 36 else 0
        if float(d_sub.max().item()) < 0.40:
            return False
        t_drop = off_t + int(d_sub.argmax().item()) + 1
        if T < t_drop + 3:
            return False

        f_rev = stack[:t_drop].mean(dim=0)
        self.perm_t_settle = t_drop + 2
        f_occ = frames[self.perm_t_settle]

        wood_rev = is_wood_tabletop_mask(f_rev)
        wood_occ = is_wood_tabletop_mask(f_occ)
        not_wood_rev = (~wood_rev).float()

        r, g, b = f_rev[:, :, 0], f_rev[:, :, 1], f_rev[:, :, 2]
        ro, go, bo = f_occ[:, :, 0], f_occ[:, :, 1], f_occ[:, :, 2]
        ch_map = {
            "red": (F.relu(r - torch.maximum(g, b)) - F.relu(ro - torch.maximum(go, bo))) * not_wood_rev,
            "green": (F.relu(g - torch.maximum(r, b)) - F.relu(go - torch.maximum(ro, bo))) * not_wood_rev,
            "blue": (F.relu(b - torch.maximum(r, g)) - F.relu(bo - torch.maximum(ro, go))) * not_wood_rev,
        }
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.perm_seats = []
        self.perm_colors = []
        for cname in ("red", "green", "blue"):
            ec = F.relu(ch_map[cname] - 18.0).unsqueeze(0).unsqueeze(0)
            ec_s = F.avg_pool2d(ec, kernel_size=7, stride=1, padding=3)[0, 0]
            ec_s[:60, :] = 0.0
            ec_s[212:, :] = 0.0
            if float(ec_s.max().item()) > 2.0:
                idx = int(ec_s.argmax().item())
                yc = float(idx // W) - 3.0
                xc = float(idx % W)
                self.perm_seats.append((yc, xc))
                self.perm_colors.append(cname)

        # Detect ALL empty grey containers (`wood_rev == True` exposed wood at `f_rev` -> `wood_occ == False` grey bin at `f_occ`!)
        occ_achrom = ((f_occ.max(dim=-1).values - f_occ.min(dim=-1).values) < 22.0) & (f_occ.mean(dim=-1) > 95.0)
        wood_to_bin = (wood_rev & (~wood_occ) & occ_achrom).float()
        wb_s = F.avg_pool2d(wood_to_bin.unsqueeze(0).unsqueeze(0), kernel_size=15, stride=1, padding=7)[0, 0]
        wb_s[:62, :] = 0.0
        wb_s[208:, :] = 0.0
        wb_s[:, :24] = 0.0
        wb_s[:, 232:] = 0.0
        for _ in range(3):
            for sy, sx in self.perm_seats:
                wb_s = wb_s.masked_fill(((yy - sy).pow(2) + (xx - sx).pow(2)) < (24.0**2), 0.0)
            if float(wb_s.max().item()) > 0.45:
                idx_e = int(wb_s.argmax().item())
                self.perm_seats.append((float(idx_e // W), float(idx_e % W)))
                self.perm_colors.append("empty")
            else:
                break

        self._perm_base_wood = []
        for sy, sx in self.perm_seats:
            iy, ix = int(round(sy)), int(round(sx))
            bw = float(wood_occ[max(0, iy - 6) : min(H, iy + 7), max(0, ix - 6) : min(W, ix + 7)].float().mean().item())
            self._perm_base_wood.append(bw)
        self._perm_vac_accum = [0.0] * len(self.perm_seats)
        self._perm_in_swap = False
        self._perm_swap_count = 0
        self._perm_prev_n_vac = 0
        self.perm_ready = len(self.perm_seats) > 0
        return self.perm_ready

    @torch.no_grad()
    def _step_demo_frame(self, ft: torch.Tensor, st: np.ndarray, goal_clean: str) -> None:
        if not (
            self._is_permanence_goal(goal_clean)
            or self._is_repick_goal(goal_clean)
            or self._is_place_ref_goal(goal_clean)
        ):
            return

        if self._is_permanence_goal(goal_clean):
            if not self.perm_ready:
                self._demo_frames.append(ft.clone())
                T = len(self._demo_frames)
                if self._init_permanence_from_window(self._demo_frames):
                    for tr in range(self.perm_t_settle + 1, T):
                        self._perm_in_swap, self._perm_swap_count, self._perm_prev_n_vac = self._step_wood_vacation_swap(
                            self._demo_frames[tr],
                            self.perm_seats,
                            self._perm_base_wood,
                            self.perm_colors,
                            in_swap=self._perm_in_swap,
                            swap_count=self._perm_swap_count,
                            prev_n_vac=self._perm_prev_n_vac,
                            vac_accum=self._perm_vac_accum,
                        )
                    self._demo_frames = [self._demo_frames[0]]
            else:
                self._perm_in_swap, self._perm_swap_count, self._perm_prev_n_vac = self._step_wood_vacation_swap(
                    ft,
                    self.perm_seats,
                    self._perm_base_wood,
                    self.perm_colors,
                    in_swap=self._perm_in_swap,
                    swap_count=self._perm_swap_count,
                    prev_n_vac=self._perm_prev_n_vac,
                    vac_accum=self._perm_vac_accum,
                )
            return

        # Buffer on CPU in uint8 (`0 MiB` GPU VRAM during 1,100-frame demos; exact 0.0-diff integer values!)
        self._demo_frames.append(ft.to(dtype=torch.uint8, device="cpu"))
        self._demo_states.append(st.copy())

    @torch.no_grad()
    def _finalize_demo_analyses(self, f_exec0: torch.Tensor, goal_clean: str) -> None:
        """Run causal summary over completed demo stream (`<= T_demo`) upon entering execution."""
        T_demo = len(self._demo_frames)
        if T_demo == 0:
            return

        if self._is_permanence_goal(goal_clean) and self.perm_ready and self._perm_in_swap:
            if self._perm_swap_count >= 10:
                ranked = sorted(range(len(self.perm_seats)), key=lambda k: self._perm_vac_accum[k], reverse=True)
                if self._perm_vac_accum[ranked[0]] >= 6.0 and self._perm_vac_accum[ranked[1]] >= 6.0:
                    i1, i2 = ranked[0], ranked[1]
                    self.perm_colors[i1], self.perm_colors[i2] = self.perm_colors[i2], self.perm_colors[i1]
            self._perm_in_swap = False

        if self._is_repick_goal(goal_clean):
            self._demo_frames = [t.to(device=self.device, dtype=torch.float32) for t in self._demo_frames]
            self._analyze_repick_demo(f_exec0)
        elif self._is_place_ref_goal(goal_clean):
            self._demo_frames = [t.to(device=self.device, dtype=torch.float32) for t in self._demo_frames]
            self._analyze_place_ref_demo(f_exec0, goal_clean)

        self._demo_frames.clear()
        self._demo_states.clear()

    @torch.no_grad()
    def _detect_cubes_on_frame(
        self, frame: torch.Tensor, *, min_dist: float = 10.0, max_peaks: int = 20
    ) -> List[Tuple[float, float, str]]:
        """Detect colored `red`/`green`/`blue` cubes `(y, x, color)` via per-color connected components + NMS fallback."""
        del min_dist
        H, W, _ = frame.shape
        not_wood = ~is_wood_tabletop_mask(frame)
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        masks = {
            "red": ((r - torch.maximum(g, b)) > 60.0) & (g < 68.0) & not_wood,
            "green": ((g - torch.maximum(r, b)) > 55.0) & not_wood,
            "blue": ((b - torch.maximum(r, g)) > 55.0) & not_wood,
        }
        # Restrict to tabletop workspace (`y in [48, 218], x in [20, 236]`), excluding top button housing (`y < 56` red cap if small)
        out: List[Tuple[float, float, float, str]] = []
        for cname, m_t in masks.items():
            m_np = m_t.detach().cpu().numpy()
            m_np[:48, :] = False
            m_np[218:, :] = False
            m_np[:, :20] = False
            m_np[:, 236:] = False
            if not m_np.any():
                continue
            # Fast 8-connected components in pure NumPy
            visited = np.zeros_like(m_np, dtype=bool)
            ys_all, xs_all = np.where(m_np)
            for y0, x0 in zip(ys_all.tolist(), xs_all.tolist()):
                if visited[y0, x0]:
                    continue
                stack = [(y0, x0)]
                visited[y0, x0] = True
                cy_sum = 0.0
                cx_sum = 0.0
                cnt = 0
                pts: List[Tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    cy_sum += cy
                    cx_sum += cx
                    cnt += 1
                    pts.append((cy, cx))
                    for ny in (cy - 1, cy, cy + 1):
                        if 48 <= ny < 218:
                            for nx in (cx - 1, cx, cx + 1):
                                if 20 <= nx < 236 and m_np[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                if cnt < 10:
                    continue
                mean_y = cy_sum / cnt
                mean_x = cx_sum / cnt
                # If two same-color cubes in Hard mode touch (`cnt > 260`), split via smoothed local peaks
                if cnt > 260:
                    sub_t = torch.zeros((H, W), device=self.device, dtype=torch.float32)
                    py_idx, px_idx = zip(*pts)
                    sub_t[list(py_idx), list(px_idx)] = 100.0
                    sub_s = F.avg_pool2d(sub_t.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4)[0, 0]
                    pks = _detect_local_peaks_2d(sub_s, thresh=35.0, min_dist=11.0, max_peaks=6)
                    for py, px, pv in pks:
                        out.append((py, px, pv, cname))
                else:
                    out.append((mean_y, mean_x, float(cnt), cname))

        out.sort(key=lambda item: item[2], reverse=True)
        return [(y, x, c) for (y, x, _, c) in out[:max_peaks]]

    @torch.no_grad()
    def _analyze_repick_demo(self, f_exec0: torch.Tensor) -> None:
        """Identify demo-grasped cube via wood-tabletop exposure & transport across post-grasp swaps (`VideoRepick`)."""
        T = len(self._demo_frames)
        if T < 20:
            return
        f0 = self._demo_frames[0]
        H, W, _ = f0.shape
        init_cubes = self._detect_cubes_on_frame(f0, min_dist=9.0, max_peaks=20)
        if not init_cubes:
            return

        grips = np.array([s[7] for s in self._demo_states], dtype=np.float32)
        cl = np.where(grips < 0.030)[0]
        if len(cl) == 0:
            return
        t_close, t_open = int(cl[0]), int(cl[-1])

        # Across `[t_close + 12 .. t_open - 10]`, ONLY the grasped cube lifts off the table exposing wood tabletop (`is_wood` 0 -> 1)!
        # (Any ungrasped cube occluded by the white robot arm has `is_wood == 0`, so zero false positives in Hard 15-cube scenes!)
        w0 = is_wood_tabletop_mask(f0)
        probe_ts = list(range(max(0, t_close + 10), max(t_close + 11, t_open - 8), 3))
        best_idx = 0
        best_score = -1e9
        for m, (cy, cx, _) in enumerate(init_cubes):
            iy, ix = int(round(cy)), int(round(cx))
            bw0 = float(w0[max(0, iy - 3) : min(H, iy + 4), max(0, ix - 3) : min(W, ix + 4)].float().mean().item())
            max_wood_gain = 0.0
            max_rgb_diff = 0.0
            p0 = f0[max(0, iy - 3) : min(H, iy + 4), max(0, ix - 3) : min(W, ix + 4), :]
            for tp in probe_ts:
                ft_p = self._demo_frames[tp]
                wt = float(
                    is_wood_tabletop_mask(ft_p)[max(0, iy - 3) : min(H, iy + 4), max(0, ix - 3) : min(W, ix + 4)]
                    .float()
                    .mean()
                    .item()
                )
                max_wood_gain = max(max_wood_gain, wt - bw0)
                pt = ft_p[max(0, iy - 3) : min(H, iy + 4), max(0, ix - 3) : min(W, ix + 4), :]
                max_rgb_diff = max(max_rgb_diff, float((pt - p0).abs().mean().item()))
            score_m = 100.0 * max_wood_gain + 0.15 * max_rgb_diff
            if score_m > best_score:
                best_score = score_m
                best_idx = m

        # Re-anchor seats after drop + arm reset (`t_settle`), BEFORE post-grasp swaps begin
        t_settle = min(T - 1, t_open + 30)
        f_settle = self._demo_frames[t_settle]
        settle_cubes = self._detect_cubes_on_frame(f_settle, min_dist=9.0, max_peaks=20)
        gy0, gx0, gcol = init_cubes[best_idx]
        if not settle_cubes:
            settle_cubes = init_cubes

        same_col = [i for i, (_, _, c) in enumerate(settle_cubes) if c == gcol]
        cand_pool = same_col if same_col else list(range(len(settle_cubes)))
        target_seat = min(
            cand_pool,
            key=lambda i: (settle_cubes[i][0] - gy0) ** 2 + (settle_cubes[i][1] - gx0) ** 2,
        )

        seats = [(c[0], c[1]) for c in settle_cubes]
        labels = ["TARGET" if i == target_seat else f"other_{i}" for i in range(len(seats))]
        if T - 1 > t_settle + 25 and 2 <= len(seats) <= 6:
            w_set = is_wood_tabletop_mask(f_settle)
            base_wood = [
                float(
                    w_set[
                        max(0, int(round(sy)) - 3) : min(H, int(round(sy)) + 4),
                        max(0, int(round(sx)) - 3) : min(W, int(round(sx)) + 4),
                    ]
                    .float()
                    .mean()
                    .item()
                )
                for sy, sx in seats
            ]
            in_sw, sw_cnt, prev_nv = False, 0, 0
            vac_acc = [0.0] * len(seats)
            for tr in range(t_settle + 1, T):
                in_sw, sw_cnt, prev_nv = self._step_wood_vacation_swap(
                    self._demo_frames[tr],
                    seats,
                    base_wood,
                    labels,
                    in_swap=in_sw,
                    swap_count=sw_cnt,
                    prev_n_vac=prev_nv,
                    vac_accum=vac_acc,
                    wood_jump_thresh=0.40,
                    patch_half=3,
                )
            if in_sw and sw_cnt >= 10:
                rk = sorted(range(len(seats)), key=lambda k: vac_acc[k], reverse=True)
                if vac_acc[rk[0]] >= 5.0 and vac_acc[rk[1]] >= 5.0:
                    labels[rk[0]], labels[rk[1]] = labels[rk[1]], labels[rk[0]]

        fin_idx = labels.index("TARGET") if "TARGET" in labels else target_seat
        fy, fx = seats[fin_idx]

        exec_cubes = self._detect_cubes_on_frame(f_exec0, min_dist=9.0, max_peaks=20)
        if exec_cubes:
            col_exec = [c for c in exec_cubes if c[2] == gcol] or exec_cubes
            near = min(col_exec, key=lambda c: (c[0] - fy) ** 2 + (c[1] - fx) ** 2)
            if (near[0] - fy) ** 2 + (near[1] - fx) ** 2 <= 18.0**2:
                fy, fx = near[0], near[1]

        self.repick_yx = (fy, fx)
        self.repick_color = gcol
        self.repick_ready = True

    @torch.no_grad()
    def _detect_purple_pads(self, frame: torch.Tensor) -> List[Tuple[float, float]]:
        """Detect purple target pads `(py, px)` on tabletop via `min(R, B) - G` chroma."""
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        p_map = F.relu(torch.minimum(r, b) - g - 22.0)
        p_s = F.avg_pool2d(p_map.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4)[0, 0]
        peaks = _detect_local_peaks_2d(
            p_s, thresh=8.0, min_dist=18.0, max_peaks=6, y_bounds=(46, 218), x_bounds=(20, 236)
        )
        return [(py, px) for py, px, _ in peaks]

    @torch.no_grad()
    def _analyze_place_ref_demo(self, f_exec0: torch.Tensor, goal_clean: str) -> None:
        """Analyze `VideoPlaceButton` & `VideoPlaceOrder` demo: placements + button press + end-of-demo pad swap."""
        T = len(self._demo_frames)
        if T < 40:
            return
        f0 = self._demo_frames[0]
        H, W, _ = f0.shape

        pads = self._detect_purple_pads(f0)
        if not pads:
            return

        t_cols = extract_goal_target_colors(goal_clean)
        want_col = t_cols[0] if t_cols else None

        grips = np.array([s[7] for s in self._demo_states], dtype=np.float32)
        is_cl = grips < 0.030
        falls = (np.where((~is_cl[:-1]) & is_cl[1:])[0] + 1).tolist()
        rises = (np.where(is_cl[:-1] & (~is_cl[1:]))[0] + 1).tolist()
        if is_cl[0]:
            falls = [0] + falls
        if is_cl[-1]:
            rises = rises + [T - 1]
        n_iv = min(len(falls), len(rises))

        t_btn_press: Optional[int] = None
        pad_visits: List[Tuple[int, int, int]] = []
        last_cube_rise = 0

        for k in range(n_iv):
            fk, rk = int(falls[k]), int(rises[k])
            if rk - fk < 5:
                continue
            min_g = float(grips[fk:rk].min())
            if min_g < 0.008:
                t_btn_press = int((fk + rk) // 2)
            elif min_g >= 0.010:
                last_cube_rise = rk
                next_fk = int(falls[k + 1]) if (k + 1) < n_iv else T - 1
                # Sample at the arm-retracted midpoint `(rk + next_fk) // 2` so hovering gripper never occludes upper pads!
                t_check = min(T - 1, max(rk + 16, int((rk + next_fk) // 2)))
                fc = self._demo_frames[t_check]
                best_j = -1
                best_cov = 0.0
                for j, (py, px) in enumerate(pads):
                    iy, ix = int(round(py)), int(round(px))
                    pc = fc[max(0, iy - 5) : min(H, iy + 6), max(0, ix - 5) : min(W, ix + 6), :]
                    # Measure target-cube primary chroma sitting directly on pad `j`!
                    rc, gc, bc = pc[:, :, 0], pc[:, :, 1], pc[:, :, 2]
                    if want_col == "red":
                        c_ex = F.relu(rc - torch.maximum(gc, bc)) * (gc < 68.0).float()
                    elif want_col == "green":
                        c_ex = F.relu(gc - torch.maximum(rc, bc))
                    elif want_col == "blue":
                        c_ex = F.relu(bc - torch.maximum(rc, gc))
                    else:
                        c_ex = torch.maximum(
                            F.relu(rc - torch.maximum(gc, bc)) * (gc < 68.0).float(),
                            torch.maximum(F.relu(gc - torch.maximum(rc, bc)), F.relu(bc - torch.maximum(rc, gc))),
                        )
                    cov_score = float(c_ex.mean().item())
                    if cov_score > best_cov:
                        best_cov = cov_score
                        best_j = j
                if best_j >= 0 and best_cov > 18.0:
                    pad_visits.append((fk, rk, best_j))

        if not pad_visits:
            return

        # Track any 50-step end-of-demo purple target-pad swap (`swap == True` in Hard mode) via wood exposure!
        pad_labels = [f"pad_{j}" for j in range(len(pads))]
        t_pad_settle = min(T - 1, last_cube_rise + 18)
        if T - 1 > t_pad_settle + 20 and len(pads) >= 2:
            w_ps = is_wood_tabletop_mask(self._demo_frames[t_pad_settle])
            base_wood = [
                float(
                    w_ps[
                        max(0, int(round(py)) - 4) : min(H, int(round(py)) + 5),
                        max(0, int(round(px)) - 4) : min(W, int(round(px)) + 5),
                    ]
                    .float()
                    .mean()
                    .item()
                )
                for py, px in pads
            ]
            in_sw, sw_cnt, prev_nv = False, 0, 0
            vac_acc = [0.0] * len(pads)
            for tr in range(t_pad_settle + 1, T):
                in_sw, sw_cnt, prev_nv = self._step_wood_vacation_swap(
                    self._demo_frames[tr],
                    pads,
                    base_wood,
                    pad_labels,
                    in_swap=in_sw,
                    swap_count=sw_cnt,
                    prev_n_vac=prev_nv,
                    vac_accum=vac_acc,
                    wood_jump_thresh=0.40,
                    patch_half=4,
                )
            if in_sw and sw_cnt >= 10:
                rk = sorted(range(len(pads)), key=lambda k: vac_acc[k], reverse=True)
                if vac_acc[rk[0]] >= 5.0 and vac_acc[rk[1]] >= 5.0:
                    pad_labels[rk[0]], pad_labels[rk[1]] = pad_labels[rk[1]], pad_labels[rk[0]]

        chosen_orig_j: Optional[int] = None
        if "button was pressed" in goal_clean:
            if t_btn_press is not None:
                if "before" in goal_clean:
                    bef = [j for (fk, rk, j) in pad_visits if rk <= t_btn_press + 10]
                    if bef:
                        chosen_orig_j = bef[-1]
                elif "after" in goal_clean:
                    aft = [j for (fk, rk, j) in pad_visits if fk >= t_btn_press - 10]
                    if aft:
                        chosen_orig_j = aft[0]
            if chosen_orig_j is None:
                chosen_orig_j = pad_visits[0][2] if "before" in goal_clean else pad_visits[-1][2]
        else:
            ord_idx = 0
            for w, idx in _ORDINAL_MAP.items():
                if f"{w} target" in goal_clean:
                    ord_idx = idx
                    break
            chosen_orig_j = pad_visits[min(ord_idx, len(pad_visits) - 1)][2]

        tag = f"pad_{chosen_orig_j}"
        fin_seat = pad_labels.index(tag) if tag in pad_labels else chosen_orig_j
        ty, tx = pads[fin_seat]

        exec_pads = self._detect_purple_pads(f_exec0)
        if exec_pads:
            near_p = min(exec_pads, key=lambda c: (c[0] - ty) ** 2 + (c[1] - tx) ** 2)
            if (near_p[0] - ty) ** 2 + (near_p[1] - tx) ** 2 <= 18.0**2:
                ty, tx = near_p[0], near_p[1]
        self.place_target_yx = (ty, tx)

        exec_cubes = self._detect_cubes_on_frame(f_exec0, min_dist=9.0, max_peaks=12)
        if exec_cubes:
            col_cands = [c for c in exec_cubes if want_col is None or c[2] == want_col]
            pick_c = col_cands[0] if col_cands else exec_cubes[0]
            self.place_cube_yx = (pick_c[0], pick_c[1])

        self.place_ready = True

    @staticmethod
    def _detect_single_button_yx(ft: torch.Tensor) -> Optional[Tuple[float, float]]:
        """Detect single tabletop button cap `<row_y, col_x>` at `t = 0` (`-2.3 px` perspective front-face offset)."""
        roi = ft[48:75, 50:206]
        wood = is_wood_tabletop_mask(roi)
        mean_v = roi.mean(dim=-1)
        chroma = roi.max(dim=-1).values - roi.min(dim=-1).values
        achrom = (~wood) & (chroma < 16.0) & (mean_v >= 120.0) & (mean_v <= 215.0)
        ys, xs = torch.where(achrom)
        if int(ys.numel()) < 8:
            return None
        return (float(ys.float().mean().item() + 45.7), float(xs.float().mean().item() + 50.0))

    @torch.no_grad()
    def _step_exec_frame(self, ft: torch.Tensor, st: np.ndarray, goal_clean: str) -> None:
        idx = self._exec_frames_count
        self._exec_frames_count += 1
        grip = float(st[7])

        if idx == 0 and "both buttons" not in goal_clean:
            self.single_button_yx = self._detect_single_button_yx(ft)

        if grip <= 0.005:
            self._exec_grip_zero_streak += 1
        elif grip > 0.010:
            self._exec_grip_zero_streak = 0

        if self._exec_frames_since_release >= 0:
            self._exec_frames_since_release += 1
            if (
                self._exec_frames_since_release == 16
                and self._is_repick_goal(goal_clean)
                and self.repick_ready
                and self.repick_yx is not None
            ):
                ry, rx = self.repick_yx
                cands = self._detect_cubes_on_frame(ft, min_dist=9.0, max_peaks=20)
                col_cands = [c for c in cands if self.repick_color is None or c[2] == self.repick_color] or cands
                if col_cands:
                    near = min(col_cands, key=lambda c: (c[0] - ry) ** 2 + (c[1] - rx) ** 2)
                    if (near[0] - ry) ** 2 + (near[1] - rx) ** 2 <= 28.0**2:
                        self.repick_yx = (near[0], near[1])

        if not self._exec_grip_closed:
            if grip < 0.031:
                self._exec_grip_closed = True
                self._exec_grip_min = grip
                self._exec_grip_hold_frames = 1
        else:
            self._exec_grip_min = min(self._exec_grip_min, grip)
            self._exec_grip_hold_frames += 1
            if grip > 0.0355:
                self._exec_grip_closed = False
                if self._exec_grip_min >= 0.009 and self._exec_grip_hold_frames >= 20:
                    self._exec_completed_object_picks += 1
                    self._exec_frames_since_release = 0
                    if self.hl_ready and self._exec_active_grasp_yx is not None:
                        ay, ax = self._exec_active_grasp_yx
                        max_s = max(self.hl_scores) if self.hl_scores else 0.0
                        thresh_hl = max(10.0, 0.35 * max_s)
                        unpicked = [
                            i for i, (sc, p) in enumerate(zip(self.hl_scores, self.hl_picked))
                            if sc >= thresh_hl and not p
                        ]
                        if unpicked:
                            best_i = min(
                                unpicked,
                                key=lambda i: (self.hl_cubes[i][0] - ay) ** 2 + (self.hl_cubes[i][1] - ax) ** 2,
                            )
                            self.hl_picked[best_i] = True
                elif self._exec_grip_min < 0.008:
                    self._exec_completed_button_presses += 1
                self._exec_grip_min = 0.04
                self._exec_grip_hold_frames = 0

        # 1) `ButtonUnmask` & `ButtonUnmaskSwap`: `demo == 0`, so reveal (`t=0..31`) & swaps (`t=64..214`) run during exec!
        if self._is_permanence_goal(goal_clean) and not self.perm_ready:
            if idx < 45:
                self._exec_early_frames.append(ft)
                if self._init_permanence_from_window(self._exec_early_frames):
                    for tr in range(self.perm_t_settle + 1, len(self._exec_early_frames)):
                        self._perm_in_swap, self._perm_swap_count, self._perm_prev_n_vac = self._step_wood_vacation_swap(
                            self._exec_early_frames[tr],
                            self.perm_seats,
                            self._perm_base_wood,
                            self.perm_colors,
                            in_swap=self._perm_in_swap,
                            swap_count=self._perm_swap_count,
                            prev_n_vac=self._perm_prev_n_vac,
                            vac_accum=self._perm_vac_accum,
                        )
                    self._exec_early_frames.clear()
        elif (
            self._is_permanence_goal(goal_clean)
            and self.perm_ready
            and not self._demo_finalized
            and not self._perm_exec_pick_started
        ):
            if self._exec_completed_object_picks > 0:
                self._perm_exec_pick_started = True
            elif idx <= 235:
                self._perm_in_swap, self._perm_swap_count, self._perm_prev_n_vac = self._step_wood_vacation_swap(
                    ft,
                    self.perm_seats,
                    self._perm_base_wood,
                    self.perm_colors,
                    in_swap=self._perm_in_swap,
                    swap_count=self._perm_swap_count,
                    prev_n_vac=self._perm_prev_n_vac,
                    vac_accum=self._perm_vac_accum,
                )

        # 2) `PickHighlight`: Voronoi + 4-quadrant ring detector on SAPIEN `231`-white disk over `t=0` wood pixels!
        if self._is_highlight_goal(goal_clean):
            if idx == 0:
                self._exec_early_frames = [ft]
                self.hl_cubes = self._detect_cubes_on_frame(ft, min_dist=10.0, max_peaks=18)
                self.hl_scores = [0.0] * len(self.hl_cubes)
                self.hl_picked = [False] * len(self.hl_cubes)
                self._hl_wood0 = is_wood_tabletop_mask(ft)
                if self.hl_cubes:
                    H, W, _ = ft.shape
                    yy, xx = torch.meshgrid(
                        torch.arange(H, device=ft.device, dtype=torch.float32),
                        torch.arange(W, device=ft.device, dtype=torch.float32),
                        indexing="ij",
                    )
                    cy_t = torch.tensor([c[0] for c in self.hl_cubes], device=ft.device, dtype=torch.float32).view(-1, 1, 1)
                    cx_t = torch.tensor([c[1] for c in self.hl_cubes], device=ft.device, dtype=torch.float32).view(-1, 1, 1)
                    self._hl_d2 = (yy.unsqueeze(0) - cy_t).pow(2) + (xx.unsqueeze(0) - cx_t).pow(2)
                    self._hl_nearest_m = self._hl_d2.argmin(dim=0)
            elif 1 <= idx <= 7:
                self._exec_early_frames.append(ft)
            elif (
                11 <= idx <= 55
                and self.hl_cubes
                and self._exec_early_frames
                and self._hl_wood0 is not None
                and self._hl_nearest_m is not None
                and self._hl_d2 is not None
            ):
                f_base = self._exec_early_frames[0]
                H, W, _ = ft.shape
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=ft.device, dtype=torch.float32),
                    torch.arange(W, device=ft.device, dtype=torch.float32),
                    indexing="ij",
                )
                is_white_disk = (
                    (ft.min(dim=-1).values >= 215.0)
                    & ((ft.max(dim=-1).values - ft.min(dim=-1).values) < 18.0)
                    & self._hl_wood0
                )
                bj = F.relu((ft - f_base).mean(dim=-1) - 42.0) * is_white_disk.float()
                for m, (cy, cx, _) in enumerate(self.hl_cubes):
                    iy, ix = int(round(cy)), int(round(cx))
                    c_patch = ft[max(0, iy - 2) : min(H, iy + 3), max(0, ix - 2) : min(W, ix + 3)]
                    chroma = float((c_patch.max(dim=-1).values - c_patch.min(dim=-1).values).max().item())
                    if chroma > 45.0:
                        vor_mask = (self._hl_nearest_m == m) & (self._hl_d2[m] <= 18.0**2)
                        w_m = is_white_disk & vor_mask
                        q_cnt = (
                            int((w_m & (yy < cy)).any().item())
                            + int((w_m & (yy > cy)).any().item())
                            + int((w_m & (xx < cx)).any().item())
                            + int((w_m & (xx > cx)).any().item())
                        )
                        if q_cnt >= 3:
                            self.hl_scores[m] += float((bj * vor_mask.float()).sum().item()) / 100.0
                if idx >= 18 and any(s >= 10.0 for s in self.hl_scores):
                    self.hl_ready = True

    # -------------------------------------------------------------------------
    # Continuous `6 x 64 x 64` Evidence Maps & Sub-Pixel `<y, x>` Readouts
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def get_evidence_maps_64(self, side: int = 64) -> torch.Tensor:
        """Return `(6, side, side)` float32 causal spatial evidence maps in `[0, 1]` for `VLMGroundedPointerCoda`."""
        if self._ev_maps_64 is not None and int(self._ev_maps_64.shape[-1]) == int(side):
            return self._ev_maps_64

        H_im, W_im = ROBOMME_IMAGE_HW
        c_y = (torch.arange(side, device=self.device, dtype=torch.float32) + 0.5) * (float(H_im) / float(side)) - 0.5
        c_x = (torch.arange(side, device=self.device, dtype=torch.float32) + 0.5) * (float(W_im) / float(side)) - 0.5
        yy, xx = torch.meshgrid(c_y, c_x, indexing="ij")
        sig2 = 9.0**2
        maps = torch.zeros((6, side, side), device=self.device, dtype=torch.float32)

        if self.perm_ready:
            col_ch = {"red": 0, "green": 1, "blue": 2}
            for (sy, sx), col in zip(self.perm_seats, self.perm_colors):
                if col in col_ch:
                    ch = col_ch[col]
                    g = torch.exp(-0.5 * ((yy - sy).pow(2) + (xx - sx).pow(2)) / sig2)
                    maps[ch] = torch.maximum(maps[ch], g)

        if self.hl_ready and self.hl_cubes:
            max_s = max(self.hl_scores) if self.hl_scores else 0.0
            thresh_hl = max(10.0, 0.35 * max_s)
            for (cy, cx, _), sc, pk in zip(self.hl_cubes, self.hl_scores, self.hl_picked):
                if sc >= thresh_hl and not pk:
                    g = torch.exp(-0.5 * ((yy - cy).pow(2) + (xx - cx).pow(2)) / sig2)
                    maps[3] = torch.maximum(maps[3], g)

        if self.repick_ready and self.repick_yx is not None:
            ry, rx = self.repick_yx
            maps[4] = torch.exp(-0.5 * ((yy - ry).pow(2) + (xx - rx).pow(2)) / sig2)
        elif self.place_ready and self.place_cube_yx is not None:
            cy, cx = self.place_cube_yx
            maps[4] = torch.exp(-0.5 * ((yy - cy).pow(2) + (xx - cx).pow(2)) / sig2)

        if self.place_ready and self.place_target_yx is not None:
            ty, tx = self.place_target_yx
            maps[5] = torch.exp(-0.5 * ((yy - ty).pow(2) + (xx - tx).pow(2)) / sig2)

        self._ev_maps_64 = maps
        return maps

    def _next_highlight_template(self) -> Optional[str]:
        if not (self.hl_ready and self.hl_cubes):
            return None
        max_s = max(self.hl_scores) if self.hl_scores else 0.0
        thresh_hl = max(10.0, 0.35 * max_s)
        hl_indices = [i for i, sc in enumerate(self.hl_scores) if sc >= thresh_hl]
        if not hl_indices:
            return None
        unpicked = [i for i in hl_indices if not self.hl_picked[i]]
        chosen_i = unpicked[0] if unpicked else hl_indices[-1]
        chosen_col = self.hl_cubes[chosen_i][2]
        if len(hl_indices) == 1:
            return f"pick up the highlighted cube at <bbox>, which is {chosen_col}"
        k_done = sum(1 for i in hl_indices if self.hl_picked[i])
        ord_word = ("first", "second", "third", "fourth", "fifth")[min(k_done, 4)]
        return f"pick up the {ord_word} highlighted cube at <bbox>, which is {chosen_col}"

    def query_target_yx(
        self,
        tmpl_str: str,
        task_goal: str,
    ) -> Optional[Tuple[float, float]]:
        """Return `(row_y, col_x)` in `[0, 255]^2` when `CORM` has high-certainty causal memory for `tmpl_str`."""
        t = (tmpl_str or "").strip().lower()
        if "<bbox>" not in t:
            return None

        goal_clean = " ".join((task_goal or "").strip().lower().split())

        if "button" in t:
            # Keep VLM's 1.5px button pointer on start-of-episode button presses; use `single_button_yx` ONLY
            # when CORM itself overrides `VideoRepick` to `"press the button at <bbox> to finish"`!
            if (
                self._is_repick_goal(goal_clean)
                and ("to finish" in t or "to stop" in t)
                and self.single_button_yx is not None
            ):
                return self.single_button_yx
            return None

        if "container" in t and self.perm_ready and self.perm_seats:
            want_color: Optional[str] = None
            for c in ("red", "green", "blue"):
                if f"{c} cube" in t:
                    want_color = c
                    break
            if want_color is None:
                g_cols = extract_goal_target_colors(goal_clean)
                if g_cols:
                    idx_c = min(self._exec_completed_object_picks, len(g_cols) - 1)
                    want_color = g_cols[idx_c]
            if want_color is not None:
                for (sy, sx), col in zip(self.perm_seats, self.perm_colors):
                    if col == want_color:
                        self._exec_active_grasp_yx = (sy, sx)
                        return (sy, sx)

        if "highlighted" in t and self.hl_ready and self.hl_cubes:
            max_s = max(self.hl_scores) if self.hl_scores else 0.0
            thresh_hl = max(10.0, 0.35 * max_s)
            hl_indices = [i for i, sc in enumerate(self.hl_scores) if sc >= thresh_hl]
            if hl_indices:
                unpicked = [i for i in hl_indices if not self.hl_picked[i]]
                pool = unpicked if unpicked else hl_indices
                want_col: Optional[str] = None
                for c in ("red", "green", "blue"):
                    if f"is {c}" in t or f"{c} cube" in t:
                        want_col = c
                        break
                col_pool = [i for i in pool if self.hl_cubes[i][2] == want_col] if want_col else []
                chosen_i = col_pool[0] if col_pool else pool[0]
                cy, cx, _ = self.hl_cubes[chosen_i]
                self._exec_active_grasp_yx = (cy, cx)
                return (cy, cx)

        if ("correct cube" in t or self._is_repick_goal(goal_clean)) and self.repick_ready and self.repick_yx is not None:
            if "cube" in t:
                self._exec_active_grasp_yx = self.repick_yx
                return self.repick_yx

        if self.place_ready:
            if "correct target" in t or "onto target" in t or "onto the target" in t:
                if self.place_target_yx is not None:
                    return self.place_target_yx
            elif "pick up" in t and "cube" in t and self.place_cube_yx is not None:
                self._exec_active_grasp_yx = self.place_cube_yx
                return self.place_cube_yx

        return None

    def refine_norm_xy(
        self,
        tmpl_str: str,
        task_goal: str,
        xy_pri_norm: torch.Tensor,
        xy_sec_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace `xy_pri_norm` (`(x, y)` in `[-1, 1]`) when `CORM` has a verified causal target `<row_y, col_x>`."""
        res = self.query_target_yx(tmpl_str, task_goal)
        if res is None:
            return xy_pri_norm, xy_sec_norm
        row_y, col_x = res
        xy_corm = pixel_xy_to_norm_xy(
            torch.tensor([col_x, row_y], device=xy_pri_norm.device, dtype=torch.float32),
            image_hw=ROBOMME_IMAGE_HW,
            align_corners=False,
        ).to(xy_pri_norm.dtype)
        return xy_corm, xy_sec_norm

    def guard_stage_template(
        self,
        tmpl_str: str,
        prev_canon_stage: Optional[str],
        raw_state: np.ndarray,
        task_goal: str,
    ) -> str:
        """100% non-privileged proprioceptive stage-transition guard (`raw_state[7]` gripper width in meters)."""
        if not tmpl_str:
            return tmpl_str
        t = tmpl_str.strip()
        tl = t.lower()
        prev = (prev_canon_stage or "").strip()
        prev_l = prev.lower()
        grip = float(raw_state[7])
        goal_clean = " ".join((task_goal or "").strip().lower().split())

        holding_obj = (0.009 <= grip <= 0.0325) and (not self._exec_grip_closed or self._exec_grip_min >= 0.008)

        if self._is_permanence_goal(goal_clean):
            if (
                "both buttons" not in goal_clean
                and "press the button" in goal_clean
                and self.perm_ready
                and self._exec_frames_count >= 128
                and (self._exec_completed_button_presses >= 1 or self._exec_grip_zero_streak >= 60)
                and "button" in tl
            ):
                g_cols = extract_goal_target_colors(goal_clean)
                if g_cols:
                    want_c = g_cols[min(self._exec_completed_object_picks, len(g_cols) - 1)]
                    return f"pick up the container at <bbox> that hides the {want_c} cube"
            if holding_obj and prev_l == "put down the container" and tl.startswith("pick up the container"):
                return "put down the container"
            if tl.startswith("pick up the container"):
                g_cols = extract_goal_target_colors(goal_clean)
                if g_cols:
                    want_c = g_cols[min(self._exec_completed_object_picks, len(g_cols) - 1)]
                    return f"pick up the container at <bbox> that hides the {want_c} cube"
                if tl.startswith("pick up the container that hides"):
                    return t.replace("pick up the container that hides", "pick up the container at <bbox> that hides", 1)

        if self._is_highlight_goal(goal_clean):
            # Prevent premature button abort (`step < 110` before button reaches bottom travel; keeps VLM latched <y,x>!)
            if (
                self._exec_frames_count < 110
                and self._exec_completed_button_presses == 0
                and self._exec_grip_zero_streak < 56
                and prev_l.startswith("press the button")
            ):
                return "press the button at <bbox>"
            # Prevent post-button-press deadlock (`grip <= 0.005` held for >=60 frames past step 128!)
            if (
                self.hl_ready
                and self._exec_frames_count >= 128
                and (self._exec_completed_button_presses >= 1 or self._exec_grip_zero_streak >= 60)
                and "button" in tl
            ):
                nxt_hl = self._next_highlight_template()
                if nxt_hl is not None:
                    return nxt_hl
            if holding_obj and prev_l == "place the cube onto the table" and tl.startswith("pick up the "):
                return "place the cube onto the table"
            if self.hl_ready and self.hl_cubes and tl.startswith("pick up") and "highlighted" in tl:
                nxt_hl = self._next_highlight_template()
                if nxt_hl is not None:
                    return nxt_hl

        if self._is_repick_goal(goal_clean) and self.repick_ready:
            n_rep = (
                3
                if "three times" in goal_clean
                else (2 if ("two times" in goal_clean or "twice" in goal_clean) else 1)
            )
            if self._exec_completed_object_picks >= n_rep:
                if holding_obj:
                    return "put it down"
                if self.single_button_yx is not None:
                    return "press the button at <bbox> to finish"
            if not holding_obj and tl.startswith("pick up") and self._exec_completed_object_picks < n_rep:
                ord_w = ("first", "second", "third", "fourth", "fifth")[min(self._exec_completed_object_picks, 4)]
                return f"pick up the correct cube at <bbox> for the {ord_w} time"
            if tl.startswith("press the button") and self._exec_completed_object_picks < n_rep:
                if holding_obj:
                    return "put it down"
                ord_w = ("first", "second", "third", "fourth", "fifth")[min(self._exec_completed_object_picks, 4)]
                return f"pick up the correct cube at <bbox> for the {ord_w} time"

        if self._is_place_ref_goal(goal_clean) and self.place_ready:
            if tl.startswith("pick up"):
                return "pick up the cube at <bbox>"
            if tl.startswith("place ") or tl.startswith("drop "):
                if not holding_obj and self._exec_completed_object_picks == 0:
                    return "pick up the cube at <bbox>"
                return "place the cube onto the correct target at <bbox>"

        return tmpl_str
