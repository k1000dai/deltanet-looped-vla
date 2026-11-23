"""Fast multi-worker Miyabi GH200 SigLIP spatial feature & global stats extractor for RoboMME.

Reads `/work/gj26/b20098/datasets/robomme_data_lerobot` (1,600 Parquet episodes) and writes:
- `/work/gj26/b20098/datasets/robomme_siglip_v1/{TaskName}.pt` (16 memory-mappable task shards)
- `/work/gj26/b20098/datasets/robomme_siglip_v1/norm_stats.json` (both `quantile_delta` & `minmax_abs`)
- `/work/gj26/b20098/datasets/robomme_siglip_v1/dp_stats.json` (exact `RoboMME/DP` `stats.json` format)
- `/work/gj26/b20098/datasets/robomme_siglip_v1/manifest.json`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from .constants import DELTA_JOINT_MASK, LEROBOT_BLOCK_TO_TASK, TASK_TO_LEROBOT_BLOCK, TASK_TO_SUITE
from .vision_encoder import SIGLIP_MODEL_ID, SiglipFeatureEncoder


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_state_action_worker(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    parquet_path, exec_start = args
    tbl = pq.read_table(parquet_path, columns=["state", "actions"])
    states = np.asarray(tbl["state"].to_pylist(), dtype=np.float32)[exec_start:]
    actions = np.asarray(tbl["actions"].to_pylist(), dtype=np.float32)[exec_start:]
    return states, actions


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


def _compute_vec_stats(arr: np.ndarray) -> dict[str, list[float]]:
    arr64 = arr.astype(np.float64)
    return {
        "mean": arr64.mean(axis=0).tolist(),
        "std": arr64.std(axis=0).tolist(),
        "min": arr64.min(axis=0).tolist(),
        "max": arr64.max(axis=0).tolist(),
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
        "median": np.quantile(arr64, 0.50, axis=0).tolist(),
    }


def compute_and_save_global_stats(
    dataset_root: Path,
    output_dir: Path,
    *,
    num_workers: int = 16,
) -> dict[str, Any]:
    """Compute exact 1,600-episode execution-frame stats for both `quantile_delta` & `minmax_abs`."""
    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])

    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")
    ep_exec_start = {
        int(s["episode_index"]): int(round(s["stats"]["exec_start_idx"]["min"][0]))
        for s in _read_jsonl(meta_dir / "episodes_stats.jsonl")
    }

    worker_args: list[tuple[str, int]] = []
    for ep in episodes_meta:
        ep_i = int(ep["episode_index"])
        p = dataset_root / data_tpl.format(episode_chunk=ep_i // chunks_size, episode_index=ep_i)
        worker_args.append((str(p), ep_exec_start.get(ep_i, 0)))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pairs = list(pool.map(_read_state_action_worker, worker_args, chunksize=8))

    all_states = np.concatenate([p[0] for p in pairs], axis=0)
    all_actions_abs = np.concatenate([p[1] for p in pairs], axis=0)

    mask = np.asarray(DELTA_JOINT_MASK, dtype=bool)
    all_actions_delta = all_actions_abs.copy()
    all_actions_delta[:, mask] -= all_states[:, mask]

    state_stats = _compute_vec_stats(all_states)
    abs_act_stats = _compute_vec_stats(all_actions_abs)
    delta_act_stats = _compute_vec_stats(all_actions_delta)

    combined = {
        "num_exec_frames": int(all_states.shape[0]),
        "quantile_delta": {
            "state": state_stats,
            "actions": delta_act_stats,
        },
        "minmax_abs": {
            "state": state_stats,
            "actions": abs_act_stats,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "norm_stats.json").write_text(json.dumps(combined, indent=2))

    # Also write exact `RoboMME/DP` `stats.json` schema (`{"norm_stats": {"state": ..., "action": ...}}`)
    dp_payload = {
        "norm_stats": {
            "state": state_stats,
            "action": abs_act_stats,
        }
    }
    (output_dir / "dp_stats.json").write_text(json.dumps(dp_payload, indent=2))
    dt = time.perf_counter() - t0
    print(
        f"[extract_siglip_cache] Computed exact global stats over {all_states.shape[0]} exec frames "
        f"in {dt:.2f}s -> {output_dir / 'norm_stats.json'}",
        flush=True,
    )
    return combined


def extract_all_shards(
    dataset_root: Path,
    output_dir: Path,
    *,
    tasks: list[str] | None = None,
    num_workers: int = 16,
    vision_batch_size: int = 1024,
    episode_group_size: int = 25,
    resume: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract packed `{TaskName}.pt` SigLIP multi-camera + text shards on GH200."""
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = output_dir / "norm_stats.json"
    if not norm_stats_path.exists() or not (output_dir / "dp_stats.json").exists():
        compute_and_save_global_stats(dataset_root, output_dir, num_workers=num_workers)

    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    data_tpl = str(info["data_path"])
    episodes_meta = {int(e["episode_index"]): e for e in _read_jsonl(meta_dir / "episodes.jsonl")}

    target_tasks = list(LEROBOT_BLOCK_TO_TASK) if tasks is None else list(tasks)
    encoder = SiglipFeatureEncoder(model_id=SIGLIP_MODEL_ID, device=device, dtype=torch.bfloat16)

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

            # Pre-allocate contiguous shard tensors so no 2x `torch.cat` memory spike occurs!
            front_emb_all = torch.empty((expected_total_t, 64, 768), dtype=torch.bfloat16)
            wrist_emb_all = torch.empty((expected_total_t, 16, 768), dtype=torch.bfloat16)
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

                f_tok, w_tok = encoder.encode_views(
                    cat_front,
                    cat_wrist,
                    batch_size=vision_batch_size,
                    return_cpu=True,
                )
                del cat_front, cat_wrist

                cursor = 0
                for d, length in zip(decoded_group, group_lengths):
                    front_emb_all[write_pos : write_pos + length].copy_(f_tok[cursor : cursor + length])
                    wrist_emb_all[write_pos : write_pos + length].copy_(w_tok[cursor : cursor + length])
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
            txt_out = encoder.encode_texts(prompts_out, return_cpu=True)
            offsets = np.concatenate([[0], np.cumsum(ep_lengths, dtype=np.int64)])

            shard_payload = {
                "task_name": task_name,
                "suite": TASK_TO_SUITE[task_name],
                "model_id": SIGLIP_MODEL_ID,
                "front_grid": (8, 8),
                "wrist_grid": (4, 4),
                "ep_offsets": torch.from_numpy(offsets),                           # (101,) int64
                "epis_idx": torch.tensor(epis_idx_out, dtype=torch.int64),         # (100,) int64
                "exec_start_idx": torch.tensor(exec_start_out, dtype=torch.int64), # (100,) int64
                "front_emb": front_emb_all,                                        # (sum_T, 64, 768) bf16
                "wrist_emb": wrist_emb_all,                                        # (sum_T, 16, 768) bf16
                "state": state_all,                                                # (sum_T, 8) fp32
                "actions": actions_all,                                            # (sum_T, 8) fp32
                "is_demo": is_demo_all,                                            # (sum_T,) bool
                "text_token_emb": txt_out["text_token_emb"].contiguous(),          # (100, 64, 768) bf16
                "text_pool_emb": txt_out["text_pool_emb"].contiguous(),            # (100, 768) bf16
                "text_attn_mask": txt_out["text_attn_mask"].contiguous(),          # (100, 64) bool
                "prompts": prompts_out,
                "simple_subgoals": simple_sgs_out,
                "grounded_subgoals": grounded_sgs_out,
            }

            tmp_path = output_dir / f".{task_name}.pt.tmp"
            torch.save(shard_payload, tmp_path, _use_new_zipfile_serialization=True)
            os.replace(tmp_path, shard_path)
            del shard_payload, front_emb_all, wrist_emb_all, state_all, actions_all, is_demo_all
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
        "model_id": SIGLIP_MODEL_ID,
        "front_grid": [8, 8],
        "wrist_grid": [4, 4],
        "dtype": "bfloat16",
        "total_extract_seconds": round(total_dt, 2),
        "shards": shard_summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[extract_siglip_cache] ALL DONE in {total_dt:.1f}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Extract multi-camera SigLIP feature shards for RoboMME")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/work/gj26/b20098/datasets/robomme_data_lerobot"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/gj26/b20098/datasets/robomme_siglip_v1"),
    )
    p.add_argument("--tasks", nargs="+", default=None, help="Optional subset of task names")
    p.add_argument("--stats-only", action="store_true", help="Only compute norm_stats.json & dp_stats.json")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--vision-batch-size", type=int, default=1024)
    p.add_argument("--episode-group-size", type=int, default=25)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.stats_only:
        compute_and_save_global_stats(args.dataset_root, args.output_dir, num_workers=args.num_workers)
        return

    extract_all_shards(
        args.dataset_root,
        args.output_dir,
        tasks=args.tasks,
        num_workers=args.num_workers,
        vision_batch_size=args.vision_batch_size,
        episode_group_size=args.episode_group_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
