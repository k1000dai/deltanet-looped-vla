"""Closed-loop `StatefulPolicy` wrapper, SentencePiece tokenizer, and image resize tools for π0.5.

Implements `robomme_eval.interface.StatefulPolicy` with:
- Exact MME-VLA `PaligemmaTokenizer` (`discrete_state_input=False` default, `max_token_len=64`).
- Exact `resize_with_pad(224, 224)` in `[-1, 1]` matching `openpi/shared/image_tools.py`.
- Chunked `vision_batch_size=32` demo/history SigLIP encoding so 1,145-frame `VideoPlaceOrder`
  episodes stay `< 10 GB` VRAM on our local 16 GB RTX 5080.
- Seamless plug-in protocol (`Pi05MemoryAdapter`) for `@pi05-trainer`'s `DeltaMem-π0.5` & `T²Mem`
  models, plus native support for `Yinpei/pi05_baseline` and `Yinpei/perceptual-framesamp-modul`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor
import torch.nn.functional as F

from robomme_eval.interface import (
    MME_VLA_NORM_STATS,
    ActionStateNormalizer,
    PolicyChunkOutput,
    StatefulPolicy,
    to_uint8_rgb,
)

from .gemma_expert import PrefixKVCache
from .model import Pi05Model, load_pi05_checkpoint
from .perceptual_mem import FrameSamplingBuffer

DEFAULT_TOKENIZER_CANDIDATES: tuple[str, ...] = (
    "/home/kohei/.cache/openpi/big_vision/paligemma_tokenizer.model",
    "/work/gj26/b20098/checkpoints/pi05_init/paligemma_tokenizer.model",
)


def resolve_tokenizer_path(explicit_path: str | Path | None = None, ckpt_dir: str | Path | None = None) -> Path:
    if explicit_path is not None and Path(explicit_path).exists():
        return Path(explicit_path)
    if ckpt_dir is not None:
        cand = Path(ckpt_dir) / "paligemma_tokenizer.model"
        if cand.exists():
            return cand
        cand_parent = Path(ckpt_dir).parent / "paligemma_tokenizer.model"
        if cand_parent.exists():
            return cand_parent
    for p in DEFAULT_TOKENIZER_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(
        f"Could not locate `paligemma_tokenizer.model` in {explicit_path}, {ckpt_dir}, or {DEFAULT_TOKENIZER_CANDIDATES}"
    )


class Pi05Tokenizer:
    """Exact port of `mme_vla_suite/training/config.py:PaligemmaTokenizer`."""

    def __init__(
        self,
        max_len: int = 64,
        *,
        discrete_state_input: bool = False,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        self.max_len = max_len
        self.discrete_state_input = discrete_state_input
        resolved = resolve_tokenizer_path(tokenizer_path)
        self._sp = spm.SentencePieceProcessor(model_proto=resolved.read_bytes())

    def tokenize(
        self,
        prompt: str,
        norm_state: np.ndarray | None = None,
        subgoal: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        state_in = norm_state if self.discrete_state_input else None

        if state_in is not None:
            if subgoal is None:
                clipped = np.clip(np.asarray(state_in, dtype=np.float32), -1.0, 1.0)
                bins = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
                disc = np.digitize(clipped, bins=bins) - 1
                state_str = " ".join(map(str, disc))
                full = f"Task: {cleaned}; State: {state_str};\nAction: "
                tokens = self._sp.encode(full, add_bos=True)
            else:
                sg = subgoal.strip().replace("_", " ").replace("\n", " ")
                full = f"Task: {cleaned}\nCurrent Subgoal: {sg}.\nAction: "
                tokens = self._sp.encode(full, add_bos=True)
        elif subgoal is not None:
            sg = subgoal.strip().replace("_", " ").replace("\n", " ")
            full = f"Task: {cleaned};\nCurrent Subgoal: {sg};\nAction: "
            tokens = self._sp.encode(full, add_bos=True)
        else:
            tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        t_len = len(tokens)
        if t_len < self.max_len:
            pad_len = self.max_len - t_len
            mask = [True] * t_len + [False] * pad_len
            tokens = tokens + [0] * pad_len
        else:
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len

        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def resize_with_pad_minus1_1(
    images_uint8: np.ndarray | Tensor,
    height: int = 224,
    width: int = 224,
    *,
    device: torch.device | str = "cpu",
    round_uint8_first: bool = True,
) -> Tensor:
    """Resize `(N, H, W, 3)` `uint8` RGB frames to `(N, height, width, 3)` `float32` in `[-1, 1]`.

    - `round_uint8_first=True`: matches `ResizeImages(224, 224)` -> `Observation.from_dict` used on
      the current step's `base_0_rgb` & `left_wrist_0_rgb`.
    - `round_uint8_first=False`: matches `MemoryBuffer.add_buffer` (`float32 / 255 * 2 - 1` before resize).
    """
    if isinstance(images_uint8, np.ndarray):
        arr = torch.from_numpy(np.ascontiguousarray(images_uint8))
    else:
        arr = images_uint8
    if arr.ndim == 3:
        arr = arr.unsqueeze(0)
    arr = arr.to(device=device)

    n, cur_h, cur_w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected NHWC with 3 channels, got {tuple(arr.shape)}")

    ratio = max(cur_w / float(width), cur_h / float(height))
    res_h = int(cur_h / ratio)
    res_w = int(cur_w / ratio)

    if round_uint8_first:
        x_nchw = arr.float().permute(0, 3, 1, 2)
        if (cur_h, cur_w) != (res_h, res_w):
            x_nchw = F.interpolate(x_nchw, size=(res_h, res_w), mode="bilinear", align_corners=False)
            x_nchw = torch.round(x_nchw).clamp(0.0, 255.0)
        pad_h0, rem_h = divmod(height - res_h, 2)
        pad_w0, rem_w = divmod(width - res_w, 2)
        if pad_h0 or rem_h or pad_w0 or rem_w:
            x_nchw = F.pad(x_nchw, (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h), value=0.0)
        out = x_nchw.permute(0, 2, 3, 1) / 255.0 * 2.0 - 1.0
        return out.contiguous()

    x_nchw = (arr.float() / 255.0 * 2.0 - 1.0).permute(0, 3, 1, 2)
    if (cur_h, cur_w) != (res_h, res_w):
        x_nchw = F.interpolate(x_nchw, size=(res_h, res_w), mode="bilinear", align_corners=False).clamp(-1.0, 1.0)
    pad_h0, rem_h = divmod(height - res_h, 2)
    pad_w0, rem_w = divmod(width - res_w, 2)
    if pad_h0 or rem_h or pad_w0 or rem_w:
        x_nchw = F.pad(x_nchw, (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h), value=-1.0)
    return x_nchw.permute(0, 2, 3, 1).contiguous()


class Pi05MemoryAdapter(Protocol):
    """Plug-in online memory interface for `@pi05-trainer` (`DeltaMem-π0.5`, `T²Mem`, etc.)."""

    def reset(self) -> None: ...

    def ingest_frames(
        self,
        front_tokens_256: Tensor,  # [T, 256, 2048]
        wrist_tokens_256: Tensor | None,  # [T, 256, 2048] or None
        raw_states: np.ndarray,  # [T, 8] float32
        norm_states: np.ndarray,  # [T, 8] float32 in [-1, 1]
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None: ...

    def read_memory(
        self,
        *,
        prefix_hidden: Tensor | None,  # [1, L_prefix, 2048]
        front_tokens_256: Tensor,  # [1, 256, 2048]
        wrist_tokens_256: Tensor,  # [1, 256, 2048]
        raw_state: np.ndarray,  # [8]
        norm_state: np.ndarray,  # [8]
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> tuple[Tensor | None, Tensor | None, dict[str, Any]]: ...


class Pi05RoboMMEPolicy(StatefulPolicy):
    """Closed-loop RoboMME evaluator policy wrapping `Pi05Model` (`< 10 GB` VRAM on RTX 5080)."""

    def __init__(
        self,
        model_or_dir: Pi05Model | str | Path,
        *,
        norm_stats: dict[str, Any] | None = None,
        memory_adapter: Pi05MemoryAdapter | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        precision: str = "bfloat16",
        mem_mod_mode: str | None = None,
        num_inference_steps: int | None = None,
        mem_cfg_scale: float = 1.0,
        noise_temp: float = 1.0,
        antithetic: bool = False,
        best_of_k: int = 1,
        rtc_weight: float = 0.25,
        ode_solver: str = "euler",
        vision_batch_size: int = 32,
        seed: int = 7,
        suppress_reset: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if isinstance(model_or_dir, (str, Path)):
            ckpt_dir: Path | None = Path(model_or_dir)
            self.model, loaded_stats = load_pi05_checkpoint(
                ckpt_dir,
                device=self.device,
                precision="bfloat16" if precision == "bfloat16" else "float32",
            )
            if mem_mod_mode is not None and mem_mod_mode != self.model.config.mem_mod_mode:
                self.model.enable_memory_modulation(mem_mod_mode)  # type: ignore[arg-type]
            stats_to_use = norm_stats if norm_stats is not None else (loaded_stats or MME_VLA_NORM_STATS)
        else:
            ckpt_dir = None
            self.model = model_or_dir.to(self.device)
            self.model.eval()
            stats_to_use = norm_stats if norm_stats is not None else MME_VLA_NORM_STATS

        self._base_max_token_len = int(self.model.config.max_token_len)
        self.normalizer = ActionStateNormalizer(mode="quantile_delta", stats=stats_to_use)
        self.tokenizer = Pi05Tokenizer(
            max_len=self.model.config.max_token_len,
            discrete_state_input=self.model.config.discrete_state_input,
            tokenizer_path=resolve_tokenizer_path(tokenizer_path, ckpt_dir),
        )
        self.memory_adapter = memory_adapter
        self.num_inference_steps = num_inference_steps or self.model.config.num_inference_steps
        self.mem_cfg_scale = mem_cfg_scale
        self.noise_temp = noise_temp
        self.antithetic = antithetic
        self.best_of_k = max(1, int(best_of_k))
        self.rtc_weight = float(rtc_weight)
        self.ode_solver = ode_solver
        self.vision_batch_size = vision_batch_size
        self.suppress_reset = suppress_reset  # Used for live k=3 un-reset interference tests!

        self.framesamp_buffer: FrameSamplingBuffer | None = None
        if self.model.mem_encoder is not None and self.memory_adapter is None:
            self.framesamp_buffer = FrameSamplingBuffer(self.model.config.perceptual_mem)

        self._seed = seed
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self.demo_frames_seen = 0
        self.exec_frames_seen = 0
        self._prev_abs_tail: np.ndarray | None = None

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    def force_reset(self) -> None:
        """Unconditionally clear all recurrent/buffer state (e.g. between k-episode sessions)."""
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self._seed)
        self.demo_frames_seen = 0
        self.exec_frames_seen = 0
        self._prev_abs_tail = None
        if self.framesamp_buffer is not None:
            self.framesamp_buffer.clear()
        if self.memory_adapter is not None:
            self.memory_adapter.reset()

    def reset(self) -> None:
        if self.suppress_reset:
            # Keep memory state alive across un-reset episode boundaries (`k=3` interference protocol)
            return
        self.force_reset()

    @property
    def needs_history_ingestion(self) -> bool:
        return self.framesamp_buffer is not None or self.memory_adapter is not None

    @staticmethod
    def _peek_adapter_selected_indices(
        adapter: Any,
        t_in: int,
        task_goal: str,
        *,
        is_demo: bool,
    ) -> list[int] | None:
        """Non-mutating pre-filter matching `DeltaMemEvalAdapter.ingest_frames` stride selection (`0.0` diff)."""
        req_attrs = (
            "cfg",
            "stream_total_frames",
            "session_start_frame",
            "session_exec_start_frame",
            "_w_is_demo",
            "_last_prompt_pool",
            "_pending_prompt_jump",
            "_encode_prompt_pool",
        )
        if not all(hasattr(adapter, a) for a in req_attrs):
            return None
        if getattr(adapter.cfg, "arch", "") == "p2_b0_pi05_base":
            return []

        sim_sess_start = int(adapter.session_start_frame)
        sim_sess_exec_start = int(adapter.session_exec_start_frame)
        sim_pending_jump = float(adapter._pending_prompt_jump)
        cur_p_pool = adapter._encode_prompt_pool(task_goal)
        if adapter._last_prompt_pool is not None:
            cos = F.cosine_similarity(cur_p_pool.float(), adapter._last_prompt_pool.float(), dim=-1).item()
            sim_pending_jump = max(sim_pending_jump, max(0.0, 1.0 - float(cos)))

        has_prev_exec = len(adapter._w_is_demo) > 0 and (not adapter._w_is_demo[-1])
        sel: list[int] = []
        for i in range(t_in):
            g_step = int(adapter.stream_total_frames) + i
            if is_demo and has_prev_exec:
                sim_sess_start = g_step
                sim_sess_exec_start = g_step
            local_idx = g_step - sim_sess_start
            is_end_demo = is_demo and (i == t_in - 1)
            if is_demo:
                stride = (
                    int(adapter.cfg.long_demo_stride)
                    if local_idx >= int(adapter.cfg.adaptive_demo_threshold)
                    else max(1, int(adapter.cfg.write_stride_demo))
                )
                write_flag = (local_idx % stride == 0) or is_end_demo
            else:
                exec_rel = max(0, local_idx - (sim_sess_exec_start - sim_sess_start))
                write_flag = (exec_rel % max(1, int(adapter.cfg.write_stride_exec))) == 0
            if write_flag or (sim_pending_jump > 0.02):
                sel.append(i)
                sim_pending_jump = 0.0
        return sel

    @torch.no_grad()
    def _encode_history_stream(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
        *,
        is_demo: bool,
    ) -> None:
        n = len(front_rgb_list)
        if n == 0 or not self.needs_history_ingestion:
            return

        if (
            self.memory_adapter is not None
            and getattr(self.memory_adapter, "wants_raw_rgb", False)
            and hasattr(self.memory_adapter, "ingest_raw_rgb_stream")
        ):
            f_raw_all = [to_uint8_rgb(x) for x in front_rgb_list]
            s_raw_all = np.stack([np.asarray(s, dtype=np.float32) for s in state_list], axis=0)
            self.memory_adapter.ingest_raw_rgb_stream(
                f_raw_all, s_raw_all, task_goal=task_goal, is_demo=is_demo
            )

        need_wrist = self.memory_adapter is not None
        bs = self.vision_batch_size
        num_patches = (self.model.config.image_size // self.model.config.siglip_patch_size) ** 2
        tok_dim = self.model.config.paligemma_width

        import inspect

        sel_accepts_demo_len = (
            self.memory_adapter is not None
            and hasattr(self.memory_adapter, "select_write_indices")
            and "total_demo_len" in inspect.signature(self.memory_adapter.select_write_indices).parameters
        )

        for start in range(0, n, bs):
            end = min(start + bs, n)
            t_in = end - start
            raw_st = np.stack([np.asarray(s, dtype=np.float32) for s in state_list[start:end]], axis=0)
            norm_st = self.normalizer.normalize_state(raw_st)

            if (
                self.framesamp_buffer is None
                and self.memory_adapter is not None
                and hasattr(self.memory_adapter, "select_write_indices")
                and hasattr(self.memory_adapter, "ingest_selected_frames")
            ):
                sel_kw: dict[str, Any] = {"task_goal": task_goal, "is_demo": is_demo}
                if sel_accepts_demo_len:
                    sel_kw["total_demo_len"] = n if is_demo else None
                sel_direct = self.memory_adapter.select_write_indices(t_in, **sel_kw)
                if sel_direct:
                    k_sel = len(sel_direct)
                    f_sel_np = np.stack([to_uint8_rgb(front_rgb_list[start + i]) for i in sel_direct], axis=0)
                    w_sel_np = np.stack([to_uint8_rgb(wrist_rgb_list[start + i]) for i in sel_direct], axis=0)
                    both_sel_img = resize_with_pad_minus1_1(
                        np.concatenate([f_sel_np, w_sel_np], axis=0),
                        self.model.config.image_size,
                        self.model.config.image_size,
                        device=self.device,
                        round_uint8_first=False,
                    )
                    vis_chunks = [
                        self.model.encode_images(both_sel_img[c0 : c0 + 16])
                        for c0 in range(0, both_sel_img.shape[0], 16)
                    ]
                    both_sel_tok = torch.cat(vis_chunks, dim=0)
                    self.memory_adapter.ingest_selected_frames(
                        both_sel_tok[:k_sel],
                        both_sel_tok[k_sel:],
                        raw_st[sel_direct],
                        norm_st[sel_direct],
                        task_goal=task_goal,
                        is_demo=is_demo,
                    )
                continue

            sel_idx: list[int] | None = None
            if self.framesamp_buffer is None and self.memory_adapter is not None:
                sel_idx = self._peek_adapter_selected_indices(
                    self.memory_adapter, t_in, task_goal, is_demo=is_demo
                )

            if sel_idx is not None:
                f_tok = torch.zeros(
                    (t_in, num_patches, tok_dim), device=self.device, dtype=self.model.compute_dtype
                )
                w_tok = torch.zeros_like(f_tok)
                if sel_idx:
                    f_sel_np = np.stack([to_uint8_rgb(front_rgb_list[start + i]) for i in sel_idx], axis=0)
                    w_sel_np = np.stack([to_uint8_rgb(wrist_rgb_list[start + i]) for i in sel_idx], axis=0)
                    both_sel_img = resize_with_pad_minus1_1(
                        np.concatenate([f_sel_np, w_sel_np], axis=0),
                        self.model.config.image_size,
                        self.model.config.image_size,
                        device=self.device,
                        round_uint8_first=False,
                    )
                    both_sel_tok = self.model.encode_images(both_sel_img)
                    k_sel = len(sel_idx)
                    idx_t = torch.tensor(sel_idx, device=self.device, dtype=torch.long)
                    f_tok[idx_t] = both_sel_tok[:k_sel]
                    w_tok[idx_t] = both_sel_tok[k_sel:]
                self.memory_adapter.ingest_frames(
                    f_tok, w_tok, raw_st, norm_st, task_goal=task_goal, is_demo=is_demo
                )
                continue

            f_np = np.stack([to_uint8_rgb(x) for x in front_rgb_list[start:end]], axis=0)
            f_img = resize_with_pad_minus1_1(
                f_np,
                self.model.config.image_size,
                self.model.config.image_size,
                device=self.device,
                round_uint8_first=False,
            )
            f_tok = self.model.encode_images(f_img)

            w_tok_opt: Tensor | None = None
            if need_wrist:
                w_np = np.stack([to_uint8_rgb(x) for x in wrist_rgb_list[start:end]], axis=0)
                w_img = resize_with_pad_minus1_1(
                    w_np,
                    self.model.config.image_size,
                    self.model.config.image_size,
                    device=self.device,
                    round_uint8_first=False,
                )
                w_tok_opt = self.model.encode_images(w_img)

            if self.framesamp_buffer is not None:
                self.framesamp_buffer.add_encoded_frames(f_tok, raw_st)
            if self.memory_adapter is not None:
                self.memory_adapter.ingest_frames(
                    f_tok, w_tok_opt, raw_st, norm_st, task_goal=task_goal, is_demo=is_demo
                )

    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        self.demo_frames_seen += len(front_rgb_list)
        self._encode_history_stream(front_rgb_list, wrist_rgb_list, state_list, task_goal, is_demo=True)
        if len(front_rgb_list) >= 128 and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def ingest_step_history(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        self.exec_frames_seen += len(front_rgb_list)
        self._encode_history_stream(front_rgb_list, wrist_rgb_list, state_list, task_goal, is_demo=False)

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
        self.exec_frames_seen += 1
        raw_state = np.asarray(state, dtype=np.float32)
        norm_state = self.normalizer.normalize_state(raw_state)

        # If framesamp_buffer is active, push current frame with exact `MemoryBuffer.add_buffer` float resize
        if self.framesamp_buffer is not None:
            f_hist = resize_with_pad_minus1_1(
                to_uint8_rgb(front_rgb),
                self.model.config.image_size,
                self.model.config.image_size,
                device=self.device,
                round_uint8_first=False,
            )
            f_hist_tok = self.model.encode_images(f_hist)
            self.framesamp_buffer.add_encoded_frames(f_hist_tok, raw_state[None, :])

        # Current-step VLM observation (`base_0_rgb`, `left_wrist_0_rgb`)
        both_np = np.stack([to_uint8_rgb(front_rgb), to_uint8_rgb(wrist_rgb)], axis=0)
        both_img = resize_with_pad_minus1_1(
            both_np,
            self.model.config.image_size,
            self.model.config.image_size,
            device=self.device,
            round_uint8_first=True,
        )
        both_tok = self.model.encode_images(both_img)
        front_tok_256 = both_tok[0:1]
        wrist_tok_256 = both_tok[1:2]

        if self.memory_adapter is not None:
            if getattr(self.memory_adapter, "wants_raw_rgb", False) and hasattr(
                self.memory_adapter, "ingest_raw_rgb_stream"
            ):
                self.memory_adapter.ingest_raw_rgb_stream(
                    [to_uint8_rgb(front_rgb)], raw_state[None, :], task_goal=task_goal, is_demo=False
                )
            self.memory_adapter.ingest_frames(
                front_tok_256.squeeze(0).unsqueeze(0),
                wrist_tok_256.squeeze(0).unsqueeze(0),
                raw_state[None, :],
                norm_state[None, :],
                task_goal=task_goal,
                is_demo=False,
            )

        subgoal_override: str | None = None
        if self.memory_adapter is not None and hasattr(self.memory_adapter, "predict_subgoal"):
            subgoal_override = self.memory_adapter.predict_subgoal(
                front_tokens_256=front_tok_256,
                wrist_tokens_256=wrist_tok_256,
                raw_state=raw_state,
                norm_state=norm_state,
                task_goal=task_goal,
                step_idx=step_idx,
                exec_start_idx=exec_start_idx,
            )
        is_dyn_base = bool(
            getattr(self.memory_adapter, "dynamic_policy_lora_on_subgoal", False)
            or getattr(getattr(self.memory_adapter, "dg_adapter", None), "dynamic_policy_lora_on_subgoal", False)
        )
        if subgoal_override is None and is_dyn_base:
            target_tok_len = self._base_max_token_len
        else:
            target_tok_len = max(
                int(self.tokenizer.max_len),
                int(self.model.config.max_token_len),
                128 if subgoal_override is not None else 0,
            )
        if self.tokenizer.max_len != target_tok_len:
            self.tokenizer.max_len = target_tok_len
        tok_np, mask_np = self.tokenizer.tokenize(task_goal, norm_state=norm_state, subgoal=subgoal_override)
        lang_tokens = torch.from_numpy(tok_np).unsqueeze(0).to(self.device)
        lang_masks = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        extra_prefix_embs: Tensor | None = None
        extra_prefix_mask: Tensor | None = None
        if self.memory_adapter is not None and hasattr(self.memory_adapter, "predict_vlm_prefix_tokens"):
            extra_prefix_embs, extra_prefix_mask = self.memory_adapter.predict_vlm_prefix_tokens(
                front_tokens_256=front_tok_256,
                wrist_tokens_256=wrist_tok_256,
                raw_state=raw_state,
                norm_state=norm_state,
                task_goal=task_goal,
                step_idx=step_idx,
                exec_start_idx=exec_start_idx,
            )

        need_prefix_hidden = self.memory_adapter is not None and self.model.config.mem_mod_mode != "none"
        prefill_out = self.model.prefill_prefix_kv(
            lang_tokens,
            lang_masks,
            image_embs=[front_tok_256, wrist_tok_256],
            extra_prefix_embs=extra_prefix_embs,
            extra_prefix_mask=extra_prefix_mask,
            return_prefix_hidden=need_prefix_hidden,
        )
        if need_prefix_hidden:
            assert isinstance(prefill_out, tuple)
            kv_cache, prefix_hidden = prefill_out
        else:
            assert isinstance(prefill_out, PrefixKVCache)
            kv_cache = prefill_out
            prefix_hidden = None

        mem_tokens: Tensor | None = None
        mem_mask: Tensor | None = None
        aux_info: dict[str, Any] = {
            "demo_frames_seen": self.demo_frames_seen,
            "exec_frames_seen": self.exec_frames_seen,
        }
        if subgoal_override is not None:
            aux_info["override_subgoal"] = subgoal_override
        loop_depth: float | int = 1

        if self.framesamp_buffer is not None and self.model.mem_encoder is not None:
            img_e, pos_e, st_e, m_e = self.framesamp_buffer.prepare_inputs(
                norm_state_fn=self.normalizer.normalize_state
            )
            img_t = torch.from_numpy(img_e).unsqueeze(0).to(self.device)
            pos_t = torch.from_numpy(pos_e).unsqueeze(0).to(self.device)
            st_t = torch.from_numpy(st_e).unsqueeze(0).to(self.device)
            mem_tokens = self.model.mem_encoder(img_t, pos_t, st_t)
            mem_mask = torch.from_numpy(m_e).unsqueeze(0).to(device=self.device, dtype=torch.bool)
            aux_info["mem_active_tokens"] = int(m_e.sum())
        elif self.memory_adapter is not None:
            mem_tokens, mem_mask, adapter_info = self.memory_adapter.read_memory(
                prefix_hidden=prefix_hidden,
                front_tokens_256=front_tok_256,
                wrist_tokens_256=wrist_tok_256,
                raw_state=raw_state,
                norm_state=norm_state,
                task_goal=task_goal,
                step_idx=step_idx,
                exec_start_idx=exec_start_idx,
            )
            if adapter_info:
                loop_depth = adapter_info.pop("loop_depth", 1)
                aux_info.update(adapter_info)

        if self.best_of_k > 1:
            sampled_k = self.model.sample_actions_from_kv(
                kv_cache,
                num_steps=self.num_inference_steps,
                mem_tokens=mem_tokens,
                mem_mask=mem_mask,
                mem_cfg_scale=self.mem_cfg_scale,
                noise_temp=self.noise_temp,
                best_of_k=self.best_of_k,
                return_all_candidates=True,
                ode_solver=self.ode_solver,
                generator=self._generator,
            )
            cands_norm = sampled_k[0, :, :, : self.model.config.real_action_dim].detach().float().cpu().numpy()
            cands_abs = np.stack(
                [self.normalizer.decode_actions(cands_norm[ki], raw_state) for ki in range(self.best_of_k)],
                axis=0,
            )  # (K, 20, 8)
            cands_7 = cands_abs[:, :, :7]
            pw_l1 = np.abs(cands_7[:, None, :, :] - cands_7[None, :, :, :]).mean(axis=(-2, -1))
            scores = pw_l1.sum(axis=-1) / float(max(1, self.best_of_k - 1))

            if (
                self.rtc_weight > 0.0
                and self._prev_abs_tail is not None
                and self._prev_abs_tail.shape[0] > 0
                and float(np.max(np.abs(raw_state[:7] - self._prev_abs_tail[0, :7]))) <= 0.35
            ):
                ov = min(int(self._prev_abs_tail.shape[0]), int(cands_7.shape[1]))
                rtc_rms = np.sqrt(np.mean((cands_7[:, :ov, :] - self._prev_abs_tail[None, :ov, :7]) ** 2, axis=(-2, -1)))
                scores = scores + self.rtc_weight * rtc_rms

            k_star = int(np.argmin(scores))
            abs_actions = cands_abs[k_star]
            aux_info["best_of_k_idx"] = k_star
        else:
            sampled_32 = self.model.sample_actions_from_kv(
                kv_cache,
                num_steps=self.num_inference_steps,
                mem_tokens=mem_tokens,
                mem_mask=mem_mask,
                mem_cfg_scale=self.mem_cfg_scale,
                noise_temp=self.noise_temp,
                antithetic=self.antithetic,
                ode_solver=self.ode_solver,
                generator=self._generator,
            )
            norm_act_8 = sampled_32[0, :, : self.model.config.real_action_dim].detach().float().cpu().numpy()
            abs_actions = self.normalizer.decode_actions(norm_act_8, raw_state)

        rec_h = (
            None
            if os.environ.get("PI05_DISABLE_ADAPTIVE_HORIZON", "0") == "1"
            else aux_info.get("recommended_exec_horizon")
        )
        if isinstance(rec_h, (int, np.integer)) and 1 <= int(rec_h) < int(abs_actions.shape[0]):
            eff_h = int(rec_h)
            self._prev_abs_tail = abs_actions[eff_h:].copy()
            abs_actions = abs_actions[:eff_h]
        else:
            exec_h = int(self.model.config.exec_horizon)
            self._prev_abs_tail = abs_actions[exec_h:].copy() if abs_actions.shape[0] > exec_h else None
        return PolicyChunkOutput(actions=abs_actions, loop_depth=loop_depth, aux_info=aux_info)
