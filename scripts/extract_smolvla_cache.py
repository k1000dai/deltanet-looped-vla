#!/usr/bin/env python3
"""Fast multi-worker Miyabi GH200 `SmolVLA` VLM connector-token & `MEAN_STD` stats extractor (`U6`).

Reads `/work/gj26/b20098/datasets/robomme_data_lerobot` (1,600 Parquet episodes across all 16 tasks)
and writes:
  - `/work/gj26/b20098/datasets/robomme_smolvla_v1/{TaskName}.pt` (16 memory-mappable task shards)
      * `front_vlm`: `(sum_T, 64, 960)` `bfloat16` (unscaled `connector(vision_model(px_512))`)
      * `wrist_vlm`: `(sum_T, 64, 960)` `bfloat16` (unscaled `connector(vision_model(px_512))`)
      * `lang_tokens`: `(100, 48)` `int64` & `lang_masks`: `(100, 48)` `bool` (`trailing "\n"`)
      * `ep_offsets`, `epis_idx`, `exec_start_idx`, `state`, `actions`, `is_demo`, `prompts`,
        `simple_subgoals`, `grounded_subgoals`
  - `/work/gj26/b20098/datasets/robomme_smolvla_v1/norm_stats.json` (16-task `MEAN_STD` state & `(20, 8)` actions)
  - `/work/gj26/b20098/datasets/robomme_smolvla_v1/manifest.json`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.smolvla_deltaloop import (
    DEFAULT_MIYABI_SMOLVLA_SAFETENSORS,
    SMOLVLM2_MODEL_ID,
    SmolVLADeltaLoopConfig,
    SmolVLADeltaLoopPolicy,
    load_smolvla_tokenizer,
    preprocess_smolvla_rgb,
    tokenize_smolvla_prompts,
)
from robomme_data.constants import (
    DELTA_JOINT_MASK,
    LEROBOT_BLOCK_TO_TASK,
    TASK_TO_LEROBOT_BLOCK,
    TASK_TO_SUITE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_exec_trajectory_worker(args: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return execution `states (T_exec, 8)` and padded `(T_exec, H_act, 8)` delta-arm/abs-gripper chunks."""
    parquet_path, exec_start, horizon = args
    tbl = pq.read_table(parquet_path, columns=["state", "actions"])
    states = np.asarray(tbl["state"].to_pylist(), dtype=np.float32)[exec_start:]
    actions = np.asarray(tbl["actions"].to_pylist(), dtype=np.float32)[exec_start:]
    t_exec = int(states.shape[0])
    if t_exec == 0:
        return np.zeros((0, 8), dtype=np.float32), np.zeros((0, horizon, 8), dtype=np.float32)

    idx = np.minimum(np.arange(t_exec)[:, None] + np.arange(horizon)[None, :], t_exec - 1)
    chunks = actions[idx].copy()  # (T_exec, H_act, 8)
    mask = np.asarray(DELTA_JOINT_MASK, dtype=bool)
    chunks[:, :, mask] -= states[:, None, mask]
    return states, chunks


def _decode_png_bytes(item: Any) -> np.ndarray:
    raw = item["bytes"] if isinstance(item, dict) else item
    with Image.open(io.BytesIO(raw)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return arr


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
    ]
    tbl = pq.read_table(parquet_path, columns=cols)
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
    }


def compute_and_save_smolvla_stats(
    dataset_root: Path,
    output_dir: Path,
    *,
    action_horizon: int = 20,
    num_workers: int = 16,
) -> dict[str, Any]:
    """Compute exact 1,600-episode `MEAN_STD` state `(8,)` & per-horizon `(20, 8)` delta-action stats."""
    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])

    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")
    ep_exec_start = {
        int(s["episode_index"]): int(round(s["stats"]["exec_start_idx"]["min"][0]))
        for s in _read_jsonl(meta_dir / "episodes_stats.jsonl")
    }

    worker_args: list[tuple[str, int, int]] = []
    for ep in episodes_meta:
        ep_i = int(ep["episode_index"])
        p = dataset_root / data_tpl.format(episode_chunk=ep_i // chunks_size, episode_index=ep_i)
        worker_args.append((str(p), ep_exec_start.get(ep_i, 0), int(action_horizon)))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pairs = list(pool.map(_read_exec_trajectory_worker, worker_args, chunksize=8))

    all_states = np.concatenate([p[0] for p in pairs if p[0].shape[0] > 0], axis=0).astype(np.float64)
    all_chunks = np.concatenate([p[1] for p in pairs if p[1].shape[0] > 0], axis=0).astype(np.float64)

    st_mean = all_states.mean(axis=0)
    st_std = np.maximum(all_states.std(axis=0), 1e-4)

    act_mean_hd = all_chunks.mean(axis=0)
    act_std_hd = np.maximum(all_chunks.std(axis=0), 1e-4)
    # Keep binary {-1, +1} gripper target symmetric at exact mean=0.0, std=1.0
    act_mean_hd[:, 7] = 0.0
    act_std_hd[:, 7] = 1.0

    payload = {
        "mode": "mean_std_delta",
        "num_exec_frames": int(all_states.shape[0]),
        "action_horizon": int(action_horizon),
        "state": {
            "mean": st_mean.tolist(),
            "std": st_std.tolist(),
            "q01": np.quantile(all_states, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_states, 0.99, axis=0).tolist(),
        },
        "actions": {
            "mean": act_mean_hd.tolist(),
            "std": act_std_hd.tolist(),
        },
        "actions_step0": {
            "mean": act_mean_hd[0].tolist(),
            "std": act_std_hd[0].tolist(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "norm_stats.json").write_text(json.dumps(payload, indent=2))
    dt = time.perf_counter() - t0
    print(
        f"[extract_smolvla_cache] Computed 16-task MEAN_STD stats over {all_states.shape[0]:,} exec frames "
        f"in {dt:.2f}s -> {output_dir / 'norm_stats.json'}",
        flush=True,
    )
    return payload


class _SmolVLAVisionConnectorExtractor:
    """Minimal frozen `SmolVLM2` `vision_model + connector` wrapper (`< 0.4 GB` GPU VRAM!)."""

    def __init__(
        self,
        safetensors_path: str | Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        image_size: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = int(image_size)

        tmp_cfg = SmolVLADeltaLoopConfig(
            arch="smolvla_m0_base",
            use_gripper_head=False,
            use_grounding_head=False,
        )
        full_policy = SmolVLADeltaLoopPolicy(tmp_cfg)
        load_info = full_policy.load_pretrained_smolvla(safetensors_path)
        print(f"[extract_smolvla_cache] Loaded pretrained smolvla_base: {load_info}", flush=True)

        self.vision_model = full_policy.vlm_with_expert.vlm.model.vision_model.to(
            device=self.device, dtype=self.dtype
        ).eval()
        self.connector = full_policy.vlm_with_expert.vlm.model.connector.to(
            device=self.device, dtype=self.dtype
        ).eval()
        self.vision_model.requires_grad_(False)
        self.connector.requires_grad_(False)
        del full_policy
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode_views(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        *,
        batch_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_n = int(front_rgb.shape[0])
        pair_bs = max(1, batch_size // 2)
        f_out = torch.empty((total_n, 64, 960), dtype=torch.bfloat16)
        w_out = torch.empty((total_n, 64, 960), dtype=torch.bfloat16)

        for s in range(0, total_n, pair_bs):
            e = min(total_n, s + pair_bs)
            m = e - s
            f_px = preprocess_smolvla_rgb(
                front_rgb[s:e],
                image_size=(self.image_size, self.image_size),
                device=self.device,
                out_dtype=self.dtype,
            )
            w_px = preprocess_smolvla_rgb(
                wrist_rgb[s:e],
                image_size=(self.image_size, self.image_size),
                device=self.device,
                out_dtype=self.dtype,
            )
            both_px = torch.cat([f_px, w_px], dim=0)
            hs = self.vision_model(pixel_values=both_px, patch_attention_mask=None).last_hidden_state
            conn = self.connector(hs).to(dtype=torch.bfloat16)
            f_out[s:e].copy_(conn[:m], non_blocking=False)
            w_out[s:e].copy_(conn[m:], non_blocking=False)

        return f_out, w_out


def extract_all_smolvla_shards(
    dataset_root: Path,
    output_dir: Path,
    *,
    safetensors_path: str | Path = DEFAULT_MIYABI_SMOLVLA_SAFETENSORS,
    tasks: list[str] | None = None,
    num_workers: int = 16,
    vision_batch_size: int = 256,
    episode_group_size: int = 25,
    action_horizon: int = 20,
    resume: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract packed `{TaskName}.pt` `SmolVLA` VLM-connector + language token shards on GH200."""
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = output_dir / "norm_stats.json"
    if not norm_stats_path.exists():
        compute_and_save_smolvla_stats(
            dataset_root, output_dir, action_horizon=action_horizon, num_workers=num_workers
        )

    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])
    episodes_meta = {int(e["episode_index"]): e for e in _read_jsonl(meta_dir / "episodes.jsonl")}

    target_tasks = list(LEROBOT_BLOCK_TO_TASK) if tasks is None else list(tasks)
    extractor = _SmolVLAVisionConnectorExtractor(safetensors_path, device=device, dtype=torch.bfloat16)
    tokenizer = load_smolvla_tokenizer(SMOLVLM2_MODEL_ID)

    t_all0 = time.perf_counter()
    shard_summaries: dict[str, Any] = {}

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for task_name in target_tasks:
            blk = TASK_TO_LEROBOT_BLOCK[task_name]
            shard_path = output_dir / f"{task_name}.pt"
            if resume and shard_path.exists() and shard_path.stat().st_size > 1024 * 1024:
                sh_meta = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
                n_frames = int(sh_meta["ep_offsets"][-1].item())
                print(
                    f"[skip] {task_name}.pt already exists ({n_frames} frames, "
                    f"{shard_path.stat().st_size / (1024**3):.2f} GiB)",
                    flush=True,
                )
                shard_summaries[task_name] = {
                    "block": blk,
                    "num_episodes": int(sh_meta["epis_idx"].shape[0]),
                    "num_frames": n_frames,
                    "size_bytes": shard_path.stat().st_size,
                }
                continue

            t_task0 = time.perf_counter()
            ep_indices = list(range(blk * 100, (blk + 1) * 100))
            expected_total_t = sum(int(episodes_meta[ep_i]["length"]) for ep_i in ep_indices)

            front_vlm_all = torch.empty((expected_total_t, 64, 960), dtype=torch.bfloat16)
            wrist_vlm_all = torch.empty((expected_total_t, 64, 960), dtype=torch.bfloat16)
            state_all = torch.empty((expected_total_t, 8), dtype=torch.float32)
            actions_all = torch.empty((expected_total_t, 8), dtype=torch.float32)
            is_demo_all = torch.empty((expected_total_t,), dtype=torch.bool)

            ep_lengths: list[int] = []
            epis_idx_out: list[int] = []
            exec_start_out: list[int] = []
            prompts_out: list[str] = []
            simple_sgs_out: list[list[str]] = []
            grounded_sgs_out: list[list[str]] = []
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

                group_lengths = [d["front"].shape[0] for d in decoded_group]
                cat_front = np.concatenate([d["front"] for d in decoded_group], axis=0)
                cat_wrist = np.concatenate([d["wrist"] for d in decoded_group], axis=0)

                f_tok, w_tok = extractor.encode_views(
                    cat_front,
                    cat_wrist,
                    batch_size=vision_batch_size,
                )
                del cat_front, cat_wrist

                cursor = 0
                for d, length in zip(decoded_group, group_lengths):
                    front_vlm_all[write_pos : write_pos + length].copy_(f_tok[cursor : cursor + length])
                    wrist_vlm_all[write_pos : write_pos + length].copy_(w_tok[cursor : cursor + length])
                    state_all[write_pos : write_pos + length].copy_(torch.from_numpy(d["state"]))
                    actions_all[write_pos : write_pos + length].copy_(torch.from_numpy(d["actions"]))
                    is_demo_all[write_pos : write_pos + length].copy_(torch.from_numpy(d["is_demo"]))
                    cursor += length
                    write_pos += length

                    ep_lengths.append(length)
                    epis_idx_out.append(int(d["epis_idx"]))
                    exec_start_out.append(int(d["exec_start_idx"]))
                    prompts_out.append(d["prompt"])
                    simple_sgs_out.append(d["simple_subgoals"])
                    grounded_sgs_out.append(d["grounded_subgoals"])
                del decoded_group, f_tok, w_tok

            assert write_pos == expected_total_t, f"Frame count mismatch: {write_pos} != {expected_total_t}"
            lang_tokens, lang_masks = tokenize_smolvla_prompts(
                tokenizer, prompts_out, max_length=48, device="cpu"
            )
            offsets = np.concatenate([[0], np.cumsum(ep_lengths, dtype=np.int64)])

            shard_payload = {
                "task_name": task_name,
                "suite": TASK_TO_SUITE[task_name],
                "model_id": "lerobot/smolvla_base",
                "vlm_model_id": SMOLVLM2_MODEL_ID,
                "front_grid": (8, 8),
                "wrist_grid": (8, 8),
                "ep_offsets": torch.from_numpy(offsets),                           # (101,) int64
                "epis_idx": torch.tensor(epis_idx_out, dtype=torch.int64),         # (100,) int64
                "exec_start_idx": torch.tensor(exec_start_out, dtype=torch.int64), # (100,) int64
                "front_vlm": front_vlm_all,                                        # (sum_T, 64, 960) bf16
                "wrist_vlm": wrist_vlm_all,                                        # (sum_T, 64, 960) bf16
                "state": state_all,                                                # (sum_T, 8) fp32
                "actions": actions_all,                                            # (sum_T, 8) fp32
                "is_demo": is_demo_all,                                            # (sum_T,) bool
                "lang_tokens": lang_tokens.contiguous(),                           # (100, 48) int64
                "lang_masks": lang_masks.contiguous(),                             # (100, 48) bool
                "prompts": prompts_out,
                "simple_subgoals": simple_sgs_out,
                "grounded_subgoals": grounded_sgs_out,
            }

            tmp_path = output_dir / f".{task_name}.pt.tmp"
            torch.save(shard_payload, tmp_path, _use_new_zipfile_serialization=True)
            os.replace(tmp_path, shard_path)
            del shard_payload, front_vlm_all, wrist_vlm_all, state_all, actions_all, is_demo_all
            gc.collect()

            dt_task = time.perf_counter() - t_task0
            total_t = int(offsets[-1])
            sz_gib = shard_path.stat().st_size / (1024**3)
            print(
                f"[{blk + 1:2d}/16 | {task_name:<16s}] {total_t:6d} frames -> "
                f"{sz_gib:.2f} GiB in {dt_task:.1f}s ({total_t / max(dt_task, 1e-3):.0f} frames/s)",
                flush=True,
            )
            shard_summaries[task_name] = {
                "block": blk,
                "num_episodes": len(ep_lengths),
                "num_frames": total_t,
                "size_bytes": shard_path.stat().st_size,
                "extract_seconds": round(dt_task, 2),
            }

    total_dt = time.perf_counter() - t_all0
    manifest = {
        "model_id": "lerobot/smolvla_base",
        "vlm_model_id": SMOLVLM2_MODEL_ID,
        "front_grid": [8, 8],
        "wrist_grid": [8, 8],
        "token_dim": 960,
        "dtype": "bfloat16",
        "total_extract_seconds": round(total_dt, 2),
        "shards": shard_summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[extract_smolvla_cache] ALL DONE in {total_dt:.1f}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Extract 16-task SmolVLA VLM-connector feature shards on GH200")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/work/gj26/b20098/datasets/robomme_data_lerobot"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/gj26/b20098/datasets/robomme_smolvla_v1"),
    )
    p.add_argument(
        "--pretrained-safetensors",
        type=Path,
        default=Path(DEFAULT_MIYABI_SMOLVLA_SAFETENSORS),
    )
    p.add_argument("--tasks", nargs="+", default=None, help="Optional subset of task names")
    p.add_argument("--stats-only", action="store_true", help="Only compute norm_stats.json")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--vision-batch-size", type=int, default=256)
    p.add_argument("--episode-group-size", type=int, default=25)
    p.add_argument("--action-horizon", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.stats_only:
        compute_and_save_smolvla_stats(
            args.dataset_root,
            args.output_dir,
            action_horizon=args.action_horizon,
            num_workers=args.num_workers,
        )
        return

    extract_all_smolvla_shards(
        args.dataset_root,
        args.output_dir,
        safetensors_path=args.pretrained_safetensors,
        tasks=args.tasks,
        num_workers=args.num_workers,
        vision_batch_size=args.vision_batch_size,
        episode_group_size=args.episode_group_size,
        action_horizon=args.action_horizon,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
