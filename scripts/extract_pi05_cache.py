#!/usr/bin/env python3
"""Fast multi-worker Miyabi GH200 `π0.5` (`PaliGemma` `So400m/14`) episode cache & subgoal target extractor.

Reads `/work/gj26/b20098/datasets/robomme_data_lerobot` (`1,600` Parquet episodes = `768,897` frames)
and writes `/work/gj26/b20098/datasets/robomme_pi05_cache/`:
  - `{TaskName}.pt` (`16` memory-mappable task shards, `< 0.5s` total `mmap=True` load time):
      * `front_emb`: `(sum_T, 64, 1152)` `bfloat16` (`8x8` pooled over 100% of demo + exec frames)
      * `wrist_emb`: `(sum_T, 16, 1152)` `bfloat16` (`4x4` pooled over 100% of demo + exec frames)
      * `exec_front_256` & `exec_wrist_256`: `(sum_M, 256, 1152)` `bfloat16` (exact unpooled `16x16`
        tokens at stride-4 execution steps -> `F.linear(x, siglip_head_weight, siglip_head_bias)` gives
        bit-exact `2048-D` `PaliGemma.img` tokens with `0.0` diff!)
      * `siglip_head_weight (2048, 1152)` & `siglip_head_bias (2048,)` `bfloat16`
      * `state (sum_T, 8)`, `state_norm (sum_T, 8)` (`quantile_delta`), `actions (sum_T, 8)`,
        `exec_target_action_chunks (sum_M, 20, 8)` (`quantile_delta`), `is_demo (sum_T,)`,
        `ep_offsets (101,)`, `exec_ep_offsets (101,)`, `exec_rel_step_idx (sum_M,)`, `exec_start_idx (100,)`
      * `lang_tokens (100, 64)` & `lang_masks (100, 64)` (exact `MME-VLA` `PaligemmaTokenizer`)
      * Dense training-only auxiliary targets: `target_xy` (`[0, 1]^2`), `target_xy_m11` (`[-1, 1]^2`),
        `has_xy`, `secondary_xy`, `has_secondary_xy`, `count_state` (`[completed, remaining, total]`),
        `has_count`, `phase_id` / `stage_id`, `color_id`, `ordinal_id`, `subgoal_boundary`,
        `gripper_event`, `proprio_events`, `subgoal_tokens (sum_T, 32)`, `subgoal_masks (sum_T, 32)`
  - `norm_stats.json` (exact `MME-VLA`, `Yinpei/pi05_baseline`, & `Yinpei/perceptual-framesamp-modul` stats)
  - `manifest.json` (full extraction timing, byte sizes, & verified `mmap=True` benchmark)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - simulator-only local venvs omit pyarrow
    pq = None  # type: ignore[assignment]
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pi05_data.dataset import (
    DEFAULT_MIYABI_PI05_CACHE_DIR,
    Pi05EpisodeDataset,
    Pi05StepDataset,
    compute_episode_exec_grid_steps,
)
from pi05_data.subgoal_targets import parse_dense_subgoal_targets
from pi05_data.tokenizer import DEFAULT_MIYABI_TOKENIZER_PATH, Pi05Tokenizer
from pi05_data.vision_tower import (
    DEFAULT_MIYABI_PI05_BASE_SAFETENSORS,
    DEFAULT_MIYABI_SIGLIP_PKL,
    Pi05SiglipVisionTower,
    pool_siglip_tokens,
    preprocess_pi05_rgb,
    project_siglip_1152_to_2048,
)
from robomme_data.cache_dataset import build_padded_action_chunk
from robomme_data.constants import LEROBOT_BLOCK_TO_TASK, TASK_TO_LEROBOT_BLOCK, TASK_TO_SUITE
from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decode_png_bytes(item: Any) -> np.ndarray:
    raw = item["bytes"] if isinstance(item, dict) else item
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _decode_episode_worker(args: tuple[str, int, str]) -> dict[str, Any]:
    parquet_path, ep_idx_expected, prompt_str = args
    cols = [
        "image",
        "wrist_image",
        "state",
        "actions",
        "is_demo",
        "exec_start_idx",
        "epis_idx",
        "simple_subgoal",
        "grounded_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal_online",
    ]
    schema_names = set(pq.read_schema(parquet_path).names)
    use_cols = [c for c in cols if c in schema_names]
    tbl = pq.read_table(parquet_path, columns=use_cols)

    front = np.stack([_decode_png_bytes(x) for x in tbl["image"].to_pylist()], axis=0)
    wrist = np.stack([_decode_png_bytes(x) for x in tbl["wrist_image"].to_pylist()], axis=0)
    state = np.asarray(tbl["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(tbl["actions"].to_pylist(), dtype=np.float32)
    is_demo = np.asarray(tbl["is_demo"].to_pylist(), dtype=bool)
    exec_start_col = tbl["exec_start_idx"].to_pylist()
    exec_start_idx = int(exec_start_col[0]) if exec_start_col else int(is_demo.sum())
    epis_col = tbl["epis_idx"].to_pylist()
    epis_idx = int(epis_col[0]) if epis_col else int(ep_idx_expected)

    simple_sgs = [str(x) if x is not None else "" for x in tbl["simple_subgoal"].to_pylist()]
    grounded_sgs = [str(x) if x is not None else "" for x in tbl["grounded_subgoal"].to_pylist()]
    simple_sgs_on = (
        [str(x) if x is not None else "" for x in tbl["simple_subgoal_online"].to_pylist()]
        if "simple_subgoal_online" in schema_names
        else list(simple_sgs)
    )
    grounded_sgs_on = (
        [str(x) if x is not None else "" for x in tbl["grounded_subgoal_online"].to_pylist()]
        if "grounded_subgoal_online" in schema_names
        else list(grounded_sgs)
    )

    return {
        "epis_idx": epis_idx,
        "exec_start_idx": exec_start_idx,
        "prompt": prompt_str,
        "front": front,
        "wrist": wrist,
        "state": state,
        "actions": actions,
        "is_demo": is_demo,
        "simple_subgoals": simple_sgs,
        "grounded_subgoals": grounded_sgs,
        "simple_subgoals_online": simple_sgs_on,
        "grounded_subgoals_online": grounded_sgs_on,
    }


def write_pi05_norm_stats(
    output_dir: Path,
    *,
    pi05_init_dir: Path = Path("/work/gj26/b20098/checkpoints/pi05_init"),
) -> dict[str, Any]:
    """Bundle canonical `MME-VLA` `norm_stats.json` alongside checkpoint-specific `norm_stats.json`."""
    repo_stats_path = REPO_ROOT / "external" / "robomme_policy_learning" / "assets" / "norm_stats.json"
    if repo_stats_path.exists():
        mme_stats = json.loads(repo_stats_path.read_text())["norm_stats"]
    else:
        mme_stats = MME_VLA_NORM_STATS

    payload: dict[str, Any] = {
        "mode": "quantile_delta",
        "action_horizon": 20,
        "exec_horizon": 16,
        "delta_joint_mask": [True, True, True, True, True, True, True, False],
        "norm_stats": mme_stats,
    }

    base_ns = pi05_init_dir / "pi05_baseline" / "79999" / "assets" / "robomme" / "norm_stats.json"
    if base_ns.exists():
        payload["pi05_baseline_norm_stats"] = json.loads(base_ns.read_text())["norm_stats"]

    fs_ns = pi05_init_dir / "perceptual_framesamp_modul" / "79999" / "assets" / "robomme" / "norm_stats.json"
    if fs_ns.exists():
        payload["framesamp_modul_norm_stats"] = json.loads(fs_ns.read_text())["norm_stats"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "norm_stats.json").write_text(json.dumps(payload, indent=2))
    return payload


def verify_vision_tower_parity(
    dataset_root: Path,
    siglip_pkl: Path,
    pi05_base_safetensors: Path,
    *,
    device: str = "cuda",
) -> dict[str, float]:
    """Verify exact numerical parity between `siglip_params.pkl` and `lerobot/pi05_base` on real RGB frames."""
    sample_pq = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    tbl = pq.read_table(str(sample_pq), columns=["image", "wrist_image"])
    rgb_pair = np.stack(
        [_decode_png_bytes(tbl["image"][0].as_py()), _decode_png_bytes(tbl["wrist_image"][0].as_py())],
        axis=0,
    )
    px = preprocess_pi05_rgb(rgb_pair, device=device)

    tower_pkl = Pi05SiglipVisionTower(compute_dtype=torch.float32).to(device).eval()
    tower_pkl.load_from_siglip_pickle(siglip_pkl)

    with torch.inference_mode():
        enc_f32, proj_f32 = tower_pkl.forward_features(px)
        proj_via_helper = project_siglip_1152_to_2048(enc_f32, tower_pkl.head.weight, tower_pkl.head.bias)
        helper_diff = float((proj_f32 - proj_via_helper).abs().max().item())

    stats: dict[str, float] = {"helper_2048_max_diff": helper_diff}

    if pi05_base_safetensors.exists():
        tower_hf = Pi05SiglipVisionTower(compute_dtype=torch.float32).to(device).eval()
        tower_hf.load_from_hf_safetensors(pi05_base_safetensors)
        with torch.inference_mode():
            enc_hf, proj_hf = tower_hf.forward_features(px)
            diff_enc = float((enc_f32 - enc_hf).abs().max().item())
            cos_hf = float(
                torch.nn.functional.cosine_similarity(
                    proj_f32.reshape(-1, 2048), proj_hf.reshape(-1, 2048), dim=-1
                )
                .min()
                .item()
            )
        stats["vs_lerobot_pi05_base_enc1152_max_diff"] = diff_enc
        stats["vs_lerobot_pi05_base_proj2048_min_cos"] = cos_hf
        assert cos_hf > 0.999, f"Unexpected vision encoder cosine mismatch vs lerobot/pi05_base: {cos_hf}"
        del tower_hf

    # Also verify bf16 `project_siglip_1152_to_2048` bit-exactness (`0.0` diff!)
    tower_bf16 = Pi05SiglipVisionTower(compute_dtype=torch.bfloat16).to(device).eval()
    tower_bf16.load_from_siglip_pickle(siglip_pkl)
    with torch.inference_mode():
        enc_bf16, proj_bf16 = tower_bf16.forward_features(px)
        proj_bf16_helper = project_siglip_1152_to_2048(
            enc_bf16, tower_bf16.head.weight, tower_bf16.head.bias
        )
        bf16_helper_diff = float((proj_bf16.float() - proj_bf16_helper.float()).abs().max().item())
        bf16_vs_fp32_cos = float(
            torch.nn.functional.cosine_similarity(
                proj_bf16.float().reshape(-1, 2048), proj_f32.reshape(-1, 2048), dim=-1
            )
            .min()
            .item()
        )
    assert bf16_helper_diff == 0.0, f"Expected exact 0.0 bf16 linear head diff, got {bf16_helper_diff}"
    stats["bf16_helper_exact_diff"] = bf16_helper_diff
    stats["bf16_vs_fp32_min_cosine"] = bf16_vs_fp32_cos

    del tower_pkl, tower_bf16
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"[extract_pi05_cache] Vision tower parity verified: {stats}", flush=True)
    return stats


class _Pi05DualCamExtractor:
    """High-throughput GH200 `bfloat16` `Pi05SiglipVisionTower` feature extractor."""

    def __init__(
        self,
        siglip_pkl: Path,
        *,
        device: str = "cuda",
        front_grid: int = 8,
        wrist_grid: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.front_grid = int(front_grid)
        self.wrist_grid = int(wrist_grid)

        self.tower = Pi05SiglipVisionTower(compute_dtype=torch.bfloat16).to(self.device).eval()
        self.tower.load_from_siglip_pickle(siglip_pkl)
        self.tower.requires_grad_(False)

        self.head_weight_bf16 = self.tower.head.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        self.head_bias_bf16 = self.tower.head.bias.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

    @torch.inference_mode()
    def encode_episode_group(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        exec_mask_global: np.ndarray,
        *,
        vision_batch_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a group of episodes, returning `(front_8x8, wrist_4x4, exec_front_256, exec_wrist_256)` on CPU."""
        total_n = int(front_rgb.shape[0])
        pair_bs = max(1, int(vision_batch_size) // 2)

        f_n = self.front_grid * self.front_grid
        w_n = self.wrist_grid * self.wrist_grid
        f_hist = torch.empty((total_n, f_n, 1152), dtype=torch.bfloat16)
        w_hist = torch.empty((total_n, w_n, 1152), dtype=torch.bfloat16)

        exec_indices = np.flatnonzero(exec_mask_global)
        n_exec = int(exec_indices.size)
        f_exec_256 = torch.empty((n_exec, 256, 1152), dtype=torch.bfloat16)
        w_exec_256 = torch.empty((n_exec, 256, 1152), dtype=torch.bfloat16)
        exec_write = 0

        for s in range(0, total_n, pair_bs):
            e = min(total_n, s + pair_bs)
            m = e - s
            f_px = preprocess_pi05_rgb(front_rgb[s:e], device=self.device)
            w_px = preprocess_pi05_rgb(wrist_rgb[s:e], device=self.device)
            both_px = torch.cat([f_px, w_px], dim=0)  # (2*m, 3, 224, 224)

            enc_1152 = self.tower(both_px, return_pre_head=True)  # (2*m, 256, 1152) bf16
            f_256 = enc_1152[:m]
            w_256 = enc_1152[m:]

            f_pool = pool_siglip_tokens(f_256, target_grid=self.front_grid)
            w_pool = pool_siglip_tokens(w_256, target_grid=self.wrist_grid)
            f_hist[s:e].copy_(f_pool, non_blocking=False)
            w_hist[s:e].copy_(w_pool, non_blocking=False)

            sub_mask = exec_mask_global[s:e]
            if sub_mask.any():
                rel_sel = torch.from_numpy(np.flatnonzero(sub_mask)).to(device=self.device)
                k_sel = int(rel_sel.numel())
                f_exec_256[exec_write : exec_write + k_sel].copy_(f_256.index_select(0, rel_sel), non_blocking=False)
                w_exec_256[exec_write : exec_write + k_sel].copy_(w_256.index_select(0, rel_sel), non_blocking=False)
                exec_write += k_sel

        assert exec_write == n_exec, f"exec_write mismatch: {exec_write} != {n_exec}"
        return f_hist, w_hist, f_exec_256, w_exec_256


def extract_all_pi05_shards(
    dataset_root: Path,
    output_dir: Path,
    *,
    siglip_pkl: Path = Path(DEFAULT_MIYABI_SIGLIP_PKL),
    pi05_base_safetensors: Path = Path(DEFAULT_MIYABI_PI05_BASE_SAFETENSORS),
    tokenizer_path: Path = Path(DEFAULT_MIYABI_TOKENIZER_PATH),
    tasks: list[str] | None = None,
    exec_256_stride: int = 4,
    action_horizon: int = 20,
    front_grid: int = 8,
    wrist_grid: int = 4,
    num_workers: int = 20,
    vision_batch_size: int = 512,
    episode_group_size: int = 20,
    resume: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_payload = write_pi05_norm_stats(output_dir)
    normalizer = ActionStateNormalizer(mode="quantile_delta", stats=norm_payload["norm_stats"])

    parity_stats = verify_vision_tower_parity(
        dataset_root, siglip_pkl, pi05_base_safetensors, device=device
    )

    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])
    episodes_meta = {int(e["episode_index"]): e for e in _read_jsonl(meta_dir / "episodes.jsonl")}

    target_tasks = list(LEROBOT_BLOCK_TO_TASK) if tasks is None else list(tasks)
    extractor = _Pi05DualCamExtractor(
        siglip_pkl, device=device, front_grid=front_grid, wrist_grid=wrist_grid
    )
    tokenizer = Pi05Tokenizer(max_len=64, model_path=tokenizer_path)

    t_all0 = time.perf_counter()
    shard_summaries: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for task_name in target_tasks:
            blk = TASK_TO_LEROBOT_BLOCK[task_name]
            shard_path = output_dir / f"{task_name}.pt"
            if resume and shard_path.exists() and shard_path.stat().st_size > 4 * 1024 * 1024:
                sh_meta = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
                n_frames = int(sh_meta["ep_offsets"][-1].item())
                n_exec_256 = int(sh_meta["exec_ep_offsets"][-1].item())
                print(
                    f"[skip] {task_name}.pt already exists ({n_frames} frames, {n_exec_256} exec_256 steps, "
                    f"{shard_path.stat().st_size / (1024**3):.2f} GiB)",
                    flush=True,
                )
                shard_summaries[task_name] = {
                    "block": blk,
                    "num_episodes": int(sh_meta["epis_idx"].shape[0]),
                    "num_frames": n_frames,
                    "num_exec_256_steps": n_exec_256,
                    "size_bytes": shard_path.stat().st_size,
                }
                continue

            t_task0 = time.perf_counter()
            ep_indices = list(range(blk * 100, (blk + 1) * 100))
            expected_total_t = sum(int(episodes_meta[ep_i]["length"]) for ep_i in ep_indices)

            front_all = torch.empty((expected_total_t, front_grid * front_grid, 1152), dtype=torch.bfloat16)
            wrist_all = torch.empty((expected_total_t, wrist_grid * wrist_grid, 1152), dtype=torch.bfloat16)
            state_all = torch.empty((expected_total_t, 8), dtype=torch.float32)
            state_norm_all = torch.empty((expected_total_t, 8), dtype=torch.float32)
            actions_all = torch.empty((expected_total_t, 8), dtype=torch.float32)
            is_demo_all = torch.empty((expected_total_t,), dtype=torch.bool)

            exec_front_256_chunks: list[torch.Tensor] = []
            exec_wrist_256_chunks: list[torch.Tensor] = []
            exec_target_chunks_list: list[np.ndarray] = []
            exec_rel_steps_list: list[np.ndarray] = []
            exec_abs_frames_list: list[np.ndarray] = []
            exec_ep_counts: list[int] = []

            ep_lengths: list[int] = []
            epis_idx_out: list[int] = []
            exec_start_out: list[int] = []
            prompts_out: list[str] = []
            simple_sgs_out: list[list[str]] = []
            grounded_sgs_out: list[list[str]] = []
            simple_sgs_on_out: list[list[str]] = []
            grounded_sgs_on_out: list[list[str]] = []

            write_pos = 0

            for g_start in range(0, len(ep_indices), episode_group_size):
                sub_eps = ep_indices[g_start : g_start + episode_group_size]
                w_args: list[tuple[str, int, str]] = []
                for ep_i in sub_eps:
                    p = dataset_root / data_tpl.format(
                        episode_chunk=ep_i // chunks_size, episode_index=ep_i
                    )
                    ep_tasks = episodes_meta[ep_i].get("tasks", [""])
                    prompt_str = str(ep_tasks[0]) if ep_tasks else ""
                    w_args.append((str(p), ep_i, prompt_str))

                decoded_group = list(pool.map(_decode_episode_worker, w_args, chunksize=1))

                group_lengths = [int(d["front"].shape[0]) for d in decoded_group]
                group_total_t = sum(group_lengths)
                group_exec_mask = np.zeros((group_total_t,), dtype=bool)

                local_cursor = 0
                per_ep_rel_steps: list[np.ndarray] = []
                for d, length in zip(decoded_group, group_lengths):
                    rel_steps = compute_episode_exec_grid_steps(
                        int(d["exec_start_idx"]), length, exec_stride=exec_256_stride
                    )
                    per_ep_rel_steps.append(rel_steps)
                    group_exec_mask[local_cursor + rel_steps] = True
                    local_cursor += length

                cat_front = np.concatenate([d["front"] for d in decoded_group], axis=0)
                cat_wrist = np.concatenate([d["wrist"] for d in decoded_group], axis=0)

                f_hist, w_hist, f_ex256, w_ex256 = extractor.encode_episode_group(
                    cat_front,
                    cat_wrist,
                    group_exec_mask,
                    vision_batch_size=vision_batch_size,
                )
                del cat_front, cat_wrist

                exec_front_256_chunks.append(f_ex256)
                exec_wrist_256_chunks.append(w_ex256)

                cursor = 0
                for d, length, rel_steps in zip(decoded_group, group_lengths, per_ep_rel_steps):
                    front_all[write_pos : write_pos + length].copy_(f_hist[cursor : cursor + length])
                    wrist_all[write_pos : write_pos + length].copy_(w_hist[cursor : cursor + length])

                    raw_st = d["state"]
                    raw_ac = d["actions"]
                    norm_st = normalizer.normalize_state(raw_st)
                    state_all[write_pos : write_pos + length].copy_(torch.from_numpy(raw_st))
                    state_norm_all[write_pos : write_pos + length].copy_(torch.from_numpy(norm_st))
                    actions_all[write_pos : write_pos + length].copy_(torch.from_numpy(raw_ac))
                    is_demo_all[write_pos : write_pos + length].copy_(torch.from_numpy(d["is_demo"]))

                    ep_chunks = np.stack(
                        [
                            build_padded_action_chunk(
                                raw_ac,
                                raw_st[int(t_step)],
                                int(t_step),
                                action_horizon,
                                normalizer,
                            )
                            for t_step in rel_steps
                        ],
                        axis=0,
                    )
                    exec_target_chunks_list.append(ep_chunks)
                    exec_rel_steps_list.append(rel_steps)
                    exec_abs_frames_list.append(write_pos + rel_steps)
                    exec_ep_counts.append(int(rel_steps.size))

                    cursor += length
                    write_pos += length

                    ep_lengths.append(length)
                    epis_idx_out.append(int(d["epis_idx"]))
                    exec_start_out.append(int(d["exec_start_idx"]))
                    prompts_out.append(d["prompt"])
                    simple_sgs_out.append(d["simple_subgoals"])
                    grounded_sgs_out.append(d["grounded_subgoals"])
                    simple_sgs_on_out.append(d["simple_subgoals_online"])
                    grounded_sgs_on_out.append(d["grounded_subgoals_online"])

                del decoded_group, f_hist, w_hist, f_ex256, w_ex256

            assert write_pos == expected_total_t, f"Frame count mismatch: {write_pos} != {expected_total_t}"

            ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths, dtype=np.int64)])
            exec_ep_offsets = np.concatenate([[0], np.cumsum(exec_ep_counts, dtype=np.int64)])
            exec_front_256_all = torch.cat(exec_front_256_chunks, dim=0)
            exec_wrist_256_all = torch.cat(exec_wrist_256_chunks, dim=0)
            del exec_front_256_chunks, exec_wrist_256_chunks

            exec_target_chunks_all = torch.from_numpy(
                np.concatenate(exec_target_chunks_list, axis=0).astype(np.float32)
            )
            exec_rel_step_all = torch.from_numpy(np.concatenate(exec_rel_steps_list, axis=0).astype(np.int64))
            exec_abs_frame_all = torch.from_numpy(np.concatenate(exec_abs_frames_list, axis=0).astype(np.int64))
            del exec_target_chunks_list, exec_rel_steps_list, exec_abs_frames_list

            lang_tokens, lang_masks = tokenizer.tokenize_batch(prompts_out, max_len=64, lower_prompt=True)

            aux_targets = parse_dense_subgoal_targets(
                task_name=task_name,
                prompts=prompts_out,
                simple_subgoals=simple_sgs_out,
                grounded_subgoals=grounded_sgs_out,
                raw_state=state_all,
                raw_actions=actions_all,
                ep_offsets=ep_offsets,
                exec_start_idx=exec_start_out,
                tokenizer=tokenizer,
                subgoal_max_tokens=32,
            )

            shard_payload: dict[str, Any] = {
                "task_name": task_name,
                "suite": TASK_TO_SUITE[task_name],
                "model_id": "Yinpei/pi05_baseline",
                "vision_model_id": "Yinpei/pi05_vision_encoder",
                "feature_dim": 1152,
                "projected_dim": 2048,
                "front_grid": (front_grid, front_grid),
                "wrist_grid": (wrist_grid, wrist_grid),
                "exec_256_grid": (16, 16),
                "exec_256_stride": int(exec_256_stride),
                "siglip_head_weight": extractor.head_weight_bf16.clone(),          # (2048, 1152) bf16
                "siglip_head_bias": extractor.head_bias_bf16.clone(),              # (2048,) bf16
                "ep_offsets": torch.from_numpy(ep_offsets),                        # (101,) int64
                "epis_idx": torch.tensor(epis_idx_out, dtype=torch.int64),         # (100,) int64
                "exec_start_idx": torch.tensor(exec_start_out, dtype=torch.int64), # (100,) int64
                "front_emb": front_all,                                            # (sum_T, 64, 1152) bf16
                "wrist_emb": wrist_all,                                            # (sum_T, 16, 1152) bf16
                "exec_ep_offsets": torch.from_numpy(exec_ep_offsets),              # (101,) int64
                "exec_rel_step_idx": exec_rel_step_all,                            # (sum_M,) int64
                "exec_abs_frame_idx": exec_abs_frame_all,                          # (sum_M,) int64
                "exec_front_256": exec_front_256_all,                              # (sum_M, 256, 1152) bf16
                "exec_wrist_256": exec_wrist_256_all,                              # (sum_M, 256, 1152) bf16
                "exec_target_action_chunks": exec_target_chunks_all,               # (sum_M, 20, 8) fp32
                "state": state_all,                                                # (sum_T, 8) fp32
                "state_norm": state_norm_all,                                      # (sum_T, 8) fp32
                "actions": actions_all,                                            # (sum_T, 8) fp32
                "is_demo": is_demo_all,                                            # (sum_T,) bool
                "lang_tokens": lang_tokens.contiguous(),                           # (100, 64) int32
                "lang_masks": lang_masks.contiguous(),                             # (100, 64) bool
                "prompts": prompts_out,
                "simple_subgoals": simple_sgs_out,
                "grounded_subgoals": grounded_sgs_out,
                "simple_subgoals_online": simple_sgs_on_out,
                "grounded_subgoals_online": grounded_sgs_on_out,
                **aux_targets,
                # Zero-cost storage alias so `sh["stage_id"]` and `sh["phase_id"]` both work identically
                "stage_id": aux_targets["phase_id"],
            }

            tmp_path = output_dir / f".{task_name}.pt.tmp"
            torch.save(shard_payload, tmp_path, _use_new_zipfile_serialization=True)
            os.replace(tmp_path, shard_path)

            n_exec_m = int(exec_ep_offsets[-1])
            has_xy_pct = 100.0 * float(aux_targets["has_xy"].float().mean().item())
            has_cnt_pct = 100.0 * float(aux_targets["has_count"].float().mean().item())
            grip_evt_cnt = int(aux_targets["gripper_event"].sum().item())

            del (
                shard_payload,
                front_all,
                wrist_all,
                exec_front_256_all,
                exec_wrist_256_all,
                exec_target_chunks_all,
                state_all,
                state_norm_all,
                actions_all,
                is_demo_all,
                aux_targets,
            )
            gc.collect()

            dt_task = time.perf_counter() - t_task0
            total_t = int(ep_offsets[-1])
            sz_gib = shard_path.stat().st_size / (1024**3)
            print(
                f"[{blk + 1:2d}/16 | {task_name:<16s}] {total_t:6d} frames ({n_exec_m:5d} exec_256) -> "
                f"{sz_gib:.2f} GiB in {dt_task:.1f}s ({total_t / max(dt_task, 1e-3):.0f} fr/s) | "
                f"has_xy={has_xy_pct:.1f}%, has_cnt={has_cnt_pct:.1f}%, grip_evts={grip_evt_cnt}",
                flush=True,
            )
            shard_summaries[task_name] = {
                "block": blk,
                "num_episodes": len(ep_lengths),
                "num_frames": total_t,
                "num_exec_256_steps": n_exec_m,
                "has_xy_pct": round(has_xy_pct, 2),
                "has_count_pct": round(has_cnt_pct, 2),
                "gripper_events": grip_evt_cnt,
                "size_bytes": shard_path.stat().st_size,
                "extract_seconds": round(dt_task, 2),
            }

    total_dt = time.perf_counter() - t_all0

    # Benchmark `mmap=True` load speed & verify counts across all extracted tasks
    t_mmap0 = time.perf_counter()
    ep_ds = Pi05EpisodeDataset(output_dir, tasks=target_tasks, mmap=True)
    mmap_dt = time.perf_counter() - t_mmap0
    sample0 = ep_ds[0]
    step_ds = Pi05StepDataset(output_dir, tasks=target_tasks, mmap=True)
    step0 = step_ds[0]

    total_bytes = sum(int(v["size_bytes"]) for v in shard_summaries.values())
    total_frames_all = sum(int(v["num_frames"]) for v in shard_summaries.values())
    total_exec_256_all = sum(int(v["num_exec_256_steps"]) for v in shard_summaries.values())

    manifest = {
        "model_id": "Yinpei/pi05_baseline",
        "vision_model_id": "Yinpei/pi05_vision_encoder",
        "feature_dim_stored": 1152,
        "projected_dim_vlm": 2048,
        "front_grid": [front_grid, front_grid],
        "wrist_grid": [wrist_grid, wrist_grid],
        "exec_256_grid": [16, 16],
        "exec_256_stride": int(exec_256_stride),
        "dtype": "bfloat16",
        "total_episodes": len(ep_ds),
        "total_frames": total_frames_all,
        "total_exec_256_steps": total_exec_256_all,
        "total_size_gib": round(total_bytes / (1024**3), 3),
        "total_extract_seconds": round(total_dt, 2),
        "mmap_load_all_shards_seconds": round(mmap_dt, 4),
        "vision_parity": parity_stats,
        "sample_shapes": {
            "ep_front_emb": list(sample0["front_emb"].shape),
            "ep_wrist_emb": list(sample0["wrist_emb"].shape),
            "ep_exec_front_256": list(sample0["exec_front_256"].shape),
            "ep_target_action_chunks": list(sample0["target_action_chunks"].shape),
            "step_front_256": list(step0["front_256"].shape),
        },
        "shards": shard_summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[extract_pi05_cache] COMPLETE! {total_frames_all:,} frames + {total_exec_256_all:,} exec_256 steps "
        f"({total_bytes / (1024**3):.2f} GiB) in {total_dt:.1f}s | mmap=True load={mmap_dt:.4f}s!",
        flush=True,
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Extract 16-task pi0.5 SigLIP episode shards & grounded subgoals on GH200")
    p.add_argument("--dataset-root", type=Path, default=Path("/work/gj26/b20098/datasets/robomme_data_lerobot"))
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_MIYABI_PI05_CACHE_DIR))
    p.add_argument("--siglip-pkl", type=Path, default=Path(DEFAULT_MIYABI_SIGLIP_PKL))
    p.add_argument("--pi05-base-safetensors", type=Path, default=Path(DEFAULT_MIYABI_PI05_BASE_SAFETENSORS))
    p.add_argument("--tokenizer-path", type=Path, default=Path(DEFAULT_MIYABI_TOKENIZER_PATH))
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--exec-256-stride", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=20)
    p.add_argument("--front-grid", type=int, default=8)
    p.add_argument("--wrist-grid", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--vision-batch-size", type=int, default=512)
    p.add_argument("--episode-group-size", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    extract_all_pi05_shards(
        args.dataset_root,
        args.output_dir,
        siglip_pkl=args.siglip_pkl,
        pi05_base_safetensors=args.pi05_base_safetensors,
        tokenizer_path=args.tokenizer_path,
        tasks=args.tasks,
        exec_256_stride=args.exec_256_stride,
        action_horizon=args.action_horizon,
        front_grid=args.front_grid,
        wrist_grid=args.wrist_grid,
        num_workers=args.num_workers,
        vision_batch_size=args.vision_batch_size,
        episode_group_size=args.episode_group_size,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
