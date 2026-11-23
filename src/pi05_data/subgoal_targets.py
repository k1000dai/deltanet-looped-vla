"""Dense training-only auxiliary target parser for all 16 RoboMME tasks.

Extracts leak-free, causally aligned per-frame targets from `(task_name, prompts, simple_subgoals,
grounded_subgoals, state, actions, ep_offsets, exec_start_idx)`:
  - `target_xy`: `(N, 2)` `float32` in `[0, 1]^2` canonical `(x_col, y_row)` pixel center coordinates
    (properly swapping RoboMME's `<row_y, col_x>` string order via `analysis.subgoal_grounding`!)
  - `target_xy_m11`: `(N, 2)` `float32` in `[-1, 1]^2` (`= target_xy * 2 - 1` where `has_xy`)
  - `has_xy`: `(N,)` `bool`
  - `secondary_xy`: `(N, 2)` `float32` in `[0, 1]^2` + `has_secondary_xy`: `(N,)` `bool` (`MoveCube`)
  - `count_state`: `(N, 3)` `int16` = `[completed_count, remaining_count, total_target_count]`
  - `has_count`: `(N,)` `bool`
  - `phase_id`: `(N,)` `int16` (`0..14` `STAGE_VOCAB` index)
  - `color_id`: `(N,)` `int16` (`0..8` `COLOR_VOCAB` index)
  - `ordinal_id`: `(N,)` `int16` (`0..10` parsed ordinal word/number)
  - `subgoal_boundary`: `(N,)` `bool` (`True` on subgoal transition steps within an episode)
  - `gripper_event`: `(N,)` `bool` (`|Δa_gripper| > 0.5` or `|Δq_gripper| > 0.005` m or stop inflection)
  - `proprio_events`: `(N, 4)` `float32` (`[delta_act_gripper, delta_state_gripper_norm, joint_speed_l2, vel_inflection]`)
  - `subgoal_tokens` & `subgoal_masks`: `(N, 32)` `int32` / `bool` PaliGemma token IDs of resolved `grounded_subgoal`
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import torch

from analysis.subgoal_grounding import (
    COLOR_VOCAB as SUBGOAL_COLOR_VOCAB,
    ROBOMME_IMAGE_HW,
    STAGE_TO_ID,
    STAGE_VOCAB as SUBGOAL_STAGE_VOCAB,
    classify_subgoal_stage,
    parse_color_id,
    parse_grounded_subgoals,
    parse_ordinal_count,
    resolve_episode_subgoals,
)

from .tokenizer import Pi05Tokenizer

_STOPCUBE_ORDINAL_MAP: dict[str, int] = {
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
}
_STOPCUBE_RE = re.compile(
    r"\bfor the (" + "|".join(_STOPCUBE_ORDINAL_MAP.keys()) + r") time\b",
    re.IGNORECASE,
)


def _segment_runs(labels: list[str], start_idx: int, end_idx: int) -> list[tuple[int, int, str]]:
    """Group contiguous non-empty resolved subgoal runs `[seg_s, seg_e, text_lower]` in `[start_idx, end_idx)`."""
    runs: list[tuple[int, int, str]] = []
    if start_idx >= end_idx:
        return runs
    cur_s = start_idx
    cur_txt = labels[start_idx].strip().lower()
    for t in range(start_idx + 1, end_idx):
        nxt = labels[t].strip().lower()
        if nxt != cur_txt:
            runs.append((cur_s, t, cur_txt))
            cur_s = t
            cur_txt = nxt
    runs.append((cur_s, end_idx, cur_txt))
    return runs


def _compute_episode_count_state(
    task_name: str,
    prompt: str,
    resolved_simple: list[str],
    exec_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(T, 3)` `int16` `[completed, remaining, total]` and `(T,)` `bool` `has_count` for one episode."""
    ep_len = len(resolved_simple)
    out = np.zeros((ep_len, 3), dtype=np.int16)
    has_cnt = np.zeros((ep_len,), dtype=np.bool_)

    def _fill_from_completion_predicate(
        seg_start: int,
        seg_end: int,
        is_rep_completion_fn: Any,
        *,
        count_last_active_on_entry: bool = False,
    ) -> None:
        runs = _segment_runs(resolved_simple, seg_start, seg_end)
        comp_runs = [idx for idx, (_, _, txt) in enumerate(runs) if is_rep_completion_fn(txt)]
        total = len(comp_runs)
        if total <= 0:
            return
        completed = 0
        for r_i, (rs, re_, txt) in enumerate(runs):
            out[rs:re_, 0] = completed
            out[rs:re_, 1] = max(0, total - completed)
            out[rs:re_, 2] = total
            has_cnt[rs:re_] = True
            if is_rep_completion_fn(txt):
                # Increment upon finishing the repetition's completion segment (e.g. placing into bin/target)
                completed = min(total, completed + 1)
            elif count_last_active_on_entry and r_i in comp_runs:
                completed = min(total, completed + 1)

    if task_name == "BinFill":
        _fill_from_completion_predicate(
            exec_start, ep_len, lambda txt: "into the bin" in txt
        )
    elif task_name == "PickXtimes":
        _fill_from_completion_predicate(
            exec_start, ep_len, lambda txt: ("onto the target" in txt) or ("on the target" in txt)
        )
    elif task_name == "SwingXtimes":
        _fill_from_completion_predicate(
            exec_start, ep_len, lambda txt: "left-side target" in txt
        )
    elif task_name == "StopCube":
        m = _STOPCUBE_RE.search(prompt or "")
        total_passes = _STOPCUBE_ORDINAL_MAP.get(m.group(1).lower(), 0) if m else 0
        if total_passes >= 1:
            t_press = ep_len - 1
            for t in range(exec_start, ep_len):
                s_low = resolved_simple[t].strip().lower()
                if "press" in s_low and "button" in s_low:
                    t_press = t
                    break
            denom = max(float(total_passes) - 0.5, 0.5)
            est_period = float(max(t_press - exec_start, 1)) / denom
            # StopCube.py samples move_interval from {60, 80, 120}; snap if within 22%
            nearest = min((60.0, 80.0, 120.0), key=lambda c: abs(c - est_period))
            period = nearest if abs(nearest - est_period) / nearest <= 0.22 else est_period

            for t in range(exec_start, ep_len):
                if t >= t_press:
                    comp = total_passes
                else:
                    rel_t = float(t - exec_start)
                    # Pass k in {1 .. total_passes-1} crosses target center at (k - 0.5) * period
                    comp = int(np.floor(rel_t / max(period, 1.0) + 0.5))
                    comp = max(0, min(total_passes - 1, comp))
                out[t, 0] = comp
                out[t, 1] = max(0, total_passes - comp)
                out[t, 2] = total_passes
                has_cnt[t] = True
    elif task_name == "VideoRepick":
        # Demo shows initial block pick; execution repeats pick + put-down K times then presses stop button
        _fill_from_completion_predicate(
            exec_start, ep_len, lambda txt: txt == "put it down" or ("put" in txt and "down" in txt)
        )
    elif task_name == "PickHighlight":
        runs = _segment_runs(resolved_simple, exec_start, ep_len)
        pick_run_ids = [i for i, (_, _, txt) in enumerate(runs) if "pick" in txt and "cube" in txt]
        total = len(pick_run_ids)
        if total >= 1:
            completed = 0
            seen_pick = False
            for rs, re_, txt in runs:
                is_pick = "pick" in txt and "cube" in txt
                if seen_pick and not is_pick and ("table" in txt or "press" in txt):
                    # Placing cube on table completes the previous pick
                    out[rs:re_, 0] = completed
                    out[rs:re_, 1] = max(0, total - completed)
                    out[rs:re_, 2] = total
                    has_cnt[rs:re_] = True
                    completed = min(total, completed + 1)
                    seen_pick = False
                else:
                    if is_pick:
                        seen_pick = True
                    out[rs:re_, 0] = completed
                    out[rs:re_, 1] = max(0, total - completed)
                    out[rs:re_, 2] = total
                    has_cnt[rs:re_] = True
    elif task_name in ("ButtonUnmask", "ButtonUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"):
        runs = _segment_runs(resolved_simple, exec_start, ep_len)
        total = sum(1 for _, _, txt in runs if "pick" in txt and "container" in txt)
        if total >= 1:
            completed = 0
            for rs, re_, txt in runs:
                out[rs:re_, 0] = completed
                out[rs:re_, 1] = max(0, total - completed)
                out[rs:re_, 2] = total
                has_cnt[rs:re_] = True
                if "put down" in txt and "container" in txt:
                    completed = min(total, completed + 1)
    elif task_name in ("VideoPlaceOrder", "VideoPlaceButton"):
        if exec_start > 0:
            _fill_from_completion_predicate(
                0, exec_start, lambda txt: "drop" in txt and "target" in txt
            )
        _fill_from_completion_predicate(
            exec_start, ep_len, lambda txt: "place" in txt and "target" in txt
        )

    return out, has_cnt


def compute_proprio_and_gripper_events(
    raw_state: torch.Tensor | np.ndarray,
    raw_actions: torch.Tensor | np.ndarray,
    ep_offsets: torch.Tensor | np.ndarray | Sequence[int],
    exec_start_idx: torch.Tensor | np.ndarray | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal `(N,)` `bool` `gripper_event` and `(N, 4)` `float32` `proprio_events`.

    Columns of `proprio_events`:
      - `0`: `delta_action_gripper` (`|a[t, 7] - a[t-1, 7]|`, `0.0` at `t=0` and `t=exec_start`)
      - `1`: `delta_state_gripper_norm` (`(q_grip[t] - q_grip[t-1]) / 0.04` in `[-1, 1]`)
      - `2`: `joint_speed_l2` (`||q_arm[t] - q_arm[t-1]||_2`)
      - `3`: `vel_inflection` (`|joint_speed_l2[t] - joint_speed_l2[t-1]|`)
    """
    st = raw_state.detach().cpu().numpy() if isinstance(raw_state, torch.Tensor) else np.asarray(raw_state)
    ac = raw_actions.detach().cpu().numpy() if isinstance(raw_actions, torch.Tensor) else np.asarray(raw_actions)
    offs = [int(x) for x in (ep_offsets.tolist() if hasattr(ep_offsets, "tolist") else ep_offsets)]
    ex_starts = [int(x) for x in (exec_start_idx.tolist() if hasattr(exec_start_idx, "tolist") else exec_start_idx)]

    n_total = st.shape[0]
    prop = np.zeros((n_total, 4), dtype=np.float32)

    for ep_i in range(len(offs) - 1):
        s, e = offs[ep_i], offs[ep_i + 1]
        if e - s <= 1:
            continue
        st_ep = st[s:e]
        ac_ep = ac[s:e]

        da_g = np.zeros((e - s,), dtype=np.float32)
        da_g[1:] = np.abs(ac_ep[1:, 7] - ac_ep[:-1, 7])

        ds_g = np.zeros((e - s,), dtype=np.float32)
        ds_g[1:] = np.clip((st_ep[1:, 7] - st_ep[:-1, 7]) / 0.04, -1.0, 1.0)

        jv = np.zeros((e - s,), dtype=np.float32)
        jv[1:] = np.linalg.norm(st_ep[1:, :7] - st_ep[:-1, :7], axis=-1)

        ex_s = ex_starts[ep_i] if ep_i < len(ex_starts) else 0
        if 0 < ex_s < (e - s):
            # Do not bridge demo->execution boundary jump as a physical velocity spike
            da_g[ex_s] = 0.0
            ds_g[ex_s] = 0.0
            jv[ex_s] = 0.0

        v_inf = np.zeros((e - s,), dtype=np.float32)
        v_inf[1:] = np.abs(jv[1:] - jv[:-1])
        if 0 < ex_s < (e - s):
            v_inf[ex_s] = 0.0

        prop[s:e, 0] = da_g
        prop[s:e, 1] = ds_g
        prop[s:e, 2] = jv
        prop[s:e, 3] = v_inf

    grip_evt = (
        (prop[:, 0] > 0.5)
        | (np.abs(prop[:, 1]) > 0.12)
        | ((prop[:, 3] > 0.06) & (prop[:, 2] < 0.035))
    )
    return torch.from_numpy(grip_evt), torch.from_numpy(prop)


def parse_dense_subgoal_targets(
    *,
    task_name: str,
    prompts: Sequence[str],
    simple_subgoals: Sequence[Sequence[str]],
    grounded_subgoals: Sequence[Sequence[str]],
    raw_state: torch.Tensor | np.ndarray,
    raw_actions: torch.Tensor | np.ndarray,
    ep_offsets: torch.Tensor | np.ndarray | Sequence[int],
    exec_start_idx: torch.Tensor | np.ndarray | Sequence[int],
    tokenizer: Pi05Tokenizer | None = None,
    subgoal_max_tokens: int = 32,
) -> dict[str, torch.Tensor]:
    """Parse complete dense auxiliary supervision dictionary for a 100-episode RoboMME task shard."""
    g_parsed = parse_grounded_subgoals(
        grounded_subgoals,
        resolve_complete=True,
        image_hw=ROBOMME_IMAGE_HW,
        align_corners=False,
    )
    xy_m11: torch.Tensor = g_parsed["xy_norm"]  # type: ignore[assignment]
    has_xy: torch.Tensor = g_parsed["has_xy"]  # type: ignore[assignment]
    sec_m11: torch.Tensor = g_parsed["secondary_xy_norm"]  # type: ignore[assignment]
    has_sec: torch.Tensor = g_parsed["has_secondary_xy"]  # type: ignore[assignment]

    # Convert canonical `[-1, 1]^2` (`((px + 0.5) / 256) * 2 - 1`) cleanly to `[0, 1]^2` (`(px + 0.5) / 256`)
    target_xy_01 = torch.where(has_xy.unsqueeze(-1), (xy_m11 + 1.0) * 0.5, torch.zeros_like(xy_m11))
    sec_xy_01 = torch.where(has_sec.unsqueeze(-1), (sec_m11 + 1.0) * 0.5, torch.zeros_like(sec_m11))

    offs = [int(x) for x in (ep_offsets.tolist() if hasattr(ep_offsets, "tolist") else ep_offsets)]
    ex_starts = [int(x) for x in (exec_start_idx.tolist() if hasattr(exec_start_idx, "tolist") else exec_start_idx)]
    n_total = offs[-1]

    count_state = np.zeros((n_total, 3), dtype=np.int16)
    has_count = np.zeros((n_total,), dtype=np.bool_)
    phase_id = np.zeros((n_total,), dtype=np.int16)
    color_id = np.zeros((n_total,), dtype=np.int16)
    ordinal_id = np.zeros((n_total,), dtype=np.int16)
    subgoal_boundary = np.zeros((n_total,), dtype=np.bool_)
    resolved_grounded_flat: list[str] = []

    for ep_i, (s_ep, g_ep) in enumerate(zip(simple_subgoals, grounded_subgoals)):
        s_start, s_end = offs[ep_i], offs[ep_i + 1]
        ex_s = ex_starts[ep_i] if ep_i < len(ex_starts) else 0
        prompt_ep = prompts[ep_i] if ep_i < len(prompts) else ""

        res_simple = resolve_episode_subgoals(s_ep)
        res_grounded = resolve_episode_subgoals(g_ep)
        resolved_grounded_flat.extend(res_grounded)

        c_arr, hc_arr = _compute_episode_count_state(task_name, prompt_ep, res_simple, ex_s)
        count_state[s_start:s_end] = c_arr
        has_count[s_start:s_end] = hc_arr

        prev_txt: str | None = None
        for local_t, (stxt, gtxt) in enumerate(zip(res_simple, res_grounded)):
            row = s_start + local_t
            st_name = classify_subgoal_stage(stxt)
            phase_id[row] = STAGE_TO_ID[st_name]
            color_id[row] = parse_color_id(stxt) or parse_color_id(gtxt)
            ordinal_id[row] = parse_ordinal_count(stxt) or parse_ordinal_count(gtxt)

            norm_s = stxt.strip().lower()
            if prev_txt is not None and norm_s and norm_s != prev_txt:
                subgoal_boundary[row] = True
            if norm_s:
                prev_txt = norm_s
            if local_t == ex_s and ex_s > 0:
                subgoal_boundary[row] = True

    gripper_event, proprio_events = compute_proprio_and_gripper_events(
        raw_state, raw_actions, offs, ex_starts
    )

    out: dict[str, torch.Tensor] = {
        "target_xy": target_xy_01.contiguous(),          # (N, 2) float32 in [0, 1]^2 (canonical x_col, y_row)
        "target_xy_m11": xy_m11.contiguous(),            # (N, 2) float32 in [-1, 1]^2
        "has_xy": has_xy.contiguous(),                   # (N,) bool
        "secondary_xy": sec_xy_01.contiguous(),          # (N, 2) float32 in [0, 1]^2
        "has_secondary_xy": has_sec.contiguous(),        # (N,) bool
        "count_state": torch.from_numpy(count_state),    # (N, 3) int16 [completed, remaining, total]
        "has_count": torch.from_numpy(has_count),        # (N,) bool
        "phase_id": torch.from_numpy(phase_id),          # (N,) int16 (STAGE_VOCAB 0..14)
        "color_id": torch.from_numpy(color_id),          # (N,) int16 (COLOR_VOCAB 0..8)
        "ordinal_id": torch.from_numpy(ordinal_id),      # (N,) int16 (0..10)
        "subgoal_boundary": torch.from_numpy(subgoal_boundary),  # (N,) bool
        "gripper_event": gripper_event.contiguous(),     # (N,) bool
        "proprio_events": proprio_events.contiguous(),   # (N, 4) float32
    }

    if tokenizer is not None:
        sg_toks, sg_masks = tokenizer.tokenize_subgoals(resolved_grounded_flat, max_len=subgoal_max_tokens)
        out["subgoal_tokens"] = sg_toks.contiguous()     # (N, 32) int32
        out["subgoal_masks"] = sg_masks.contiguous()     # (N, 32) bool

    return out
