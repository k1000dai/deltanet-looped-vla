#!/usr/bin/env python3
"""Low-RAM Orbax/OCDBT -> PyTorch `.safetensors` converter & Flax-equation parity verifier.

Streams Zarr arrays one-by-one from OCDBT via `tensorstore` (`< 8 GiB` peak CPU RAM, safe on login
nodes), validates 100% of parameter keys and shapes against `Pi05Model` on `torch.device("meta")`
(`0 B` extra RAM, `strict=True` equivalent), exports an independent un-transposed Flax-`einsum`
reference fixture (`flax_parity_fixture.pt`, `< 1.5 GiB` peak RAM), and verifies `Pi05Model`
numerical forward parity in both `float32` (`< 1e-5`) and `bfloat16` (`< 1e-3`) plus exact `0.0`
step-0 `modul_residual` hook parity.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pi05 import (  # noqa: E402
    PerceptualMemConfig,
    Pi05Config,
    Pi05Model,
    apply_rope,
    load_pi05_checkpoint,
    make_attn_mask,
    posemb_sincos,
)
from pi05.gemma_expert import OPENPI_MASK_NEG  # noqa: E402


class OCDBTReader:
    """On-demand reader for Orbax OCDBT Zarr arrays so full 14 GiB fp32 trees never sit in RAM at once."""

    def __init__(self, params_dir: Path) -> None:
        import tensorstore as ts

        self._ts = ts
        self.kv = {"driver": "ocdbt", "base": {"driver": "file", "path": str(params_dir.resolve())}}
        kvs = ts.KvStore.open(self.kv).result()
        raw_keys = kvs.list().result()
        self.key_to_zarr: dict[str, str] = {}
        for rk in raw_keys:
            if rk.endswith(b"/.zarray"):
                zp = rk.decode("utf-8")[:-8]
                parts = zp.split(".")
                if parts[0] == "params":
                    parts = parts[1:]
                if parts[-1] == "value":
                    parts = parts[:-1]
                self.key_to_zarr["/".join(parts)] = zp
        if not self.key_to_zarr:
            raise FileNotFoundError(f"No `/.zarray` entries found in {params_dir}")
        self.consumed: set[str] = set()

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_zarr

    def __len__(self) -> int:
        return len(self.key_to_zarr)

    def read_tensor(self, key: str, *, dtype: torch.dtype = torch.float32, mark: bool = True) -> torch.Tensor:
        if key not in self.key_to_zarr:
            raise KeyError(f"Missing Orbax key: {key!r}")
        if mark:
            self.consumed.add(key)
        zp = self.key_to_zarr[key]
        arr = self._ts.open({"driver": "zarr", "kvstore": self.kv, "path": zp}).result().read().result()
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return t.to(dtype) if dtype != torch.float32 else t

    def unconsumed_keys(self) -> list[str]:
        return sorted(set(self.key_to_zarr) - self.consumed)


def stream_convert_orbax_to_safetensors(
    reader: OCDBTReader,
    config: Pi05Config,
) -> dict[str, torch.Tensor]:
    """Stream-convert every Orbax array directly into target precision (`bfloat16` + `fp32` norms/IO)."""
    heavy_dt = torch.bfloat16 if config.precision == "bfloat16" else torch.float32
    sd: dict[str, torch.Tensor] = {}

    # 1. SigLIP-So400m/14 (`patch_embedding` and `pos_embedding` kept in fp32)
    sd["vision_tower.patch_embedding.weight"] = (
        reader.read_tensor("PaliGemma/img/embedding/kernel", dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    )
    sd["vision_tower.patch_embedding.bias"] = reader.read_tensor(
        "PaliGemma/img/embedding/bias", dtype=torch.float32
    ).contiguous()
    sd["vision_tower.pos_embedding"] = reader.read_tensor(
        "PaliGemma/img/pos_embedding", dtype=torch.float32
    ).contiguous()

    ln0_s = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale", dtype=heavy_dt)
    ln0_b = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias", dtype=heavy_dt)
    ln1_s = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale", dtype=heavy_dt)
    ln1_b = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias", dtype=heavy_dt)
    fc1_k = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel", dtype=heavy_dt)
    fc1_b = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias", dtype=heavy_dt)
    fc2_k = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel", dtype=heavy_dt)
    fc2_b = reader.read_tensor("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias", dtype=heavy_dt)
    q_k = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel", dtype=heavy_dt
    )
    q_b = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias", dtype=heavy_dt
    )
    k_k = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel", dtype=heavy_dt
    )
    k_b = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias", dtype=heavy_dt
    )
    v_k = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel", dtype=heavy_dt
    )
    v_b = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias", dtype=heavy_dt
    )
    o_k = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel", dtype=heavy_dt
    )
    o_b = reader.read_tensor(
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias", dtype=heavy_dt
    )

    vw = config.siglip_width
    for i in range(config.siglip_depth):
        pfx = f"vision_tower.layers.{i}"
        sd[f"{pfx}.layer_norm1.weight"] = ln0_s[i].clone()
        sd[f"{pfx}.layer_norm1.bias"] = ln0_b[i].clone()
        sd[f"{pfx}.layer_norm2.weight"] = ln1_s[i].clone()
        sd[f"{pfx}.layer_norm2.bias"] = ln1_b[i].clone()
        sd[f"{pfx}.mlp.fc1.weight"] = fc1_k[i].t().contiguous()
        sd[f"{pfx}.mlp.fc1.bias"] = fc1_b[i].clone()
        sd[f"{pfx}.mlp.fc2.weight"] = fc2_k[i].t().contiguous()
        sd[f"{pfx}.mlp.fc2.bias"] = fc2_b[i].clone()
        sd[f"{pfx}.self_attn.q_proj.weight"] = q_k[i].reshape(vw, vw).t().contiguous()
        sd[f"{pfx}.self_attn.q_proj.bias"] = q_b[i].reshape(vw).clone()
        sd[f"{pfx}.self_attn.k_proj.weight"] = k_k[i].reshape(vw, vw).t().contiguous()
        sd[f"{pfx}.self_attn.k_proj.bias"] = k_b[i].reshape(vw).clone()
        sd[f"{pfx}.self_attn.v_proj.weight"] = v_k[i].reshape(vw, vw).t().contiguous()
        sd[f"{pfx}.self_attn.v_proj.bias"] = v_b[i].reshape(vw).clone()
        sd[f"{pfx}.self_attn.out_proj.weight"] = o_k[i].reshape(vw, vw).t().contiguous()
        sd[f"{pfx}.self_attn.out_proj.bias"] = o_b[i].clone()
    del ln0_s, ln0_b, ln1_s, ln1_b, fc1_k, fc1_b, fc2_k, fc2_b, q_k, q_b, k_k, k_b, v_k, v_b, o_k, o_b
    gc.collect()

    sd["vision_tower.post_layernorm.weight"] = reader.read_tensor(
        "PaliGemma/img/Transformer/encoder_norm/scale", dtype=heavy_dt
    ).contiguous()
    sd["vision_tower.post_layernorm.bias"] = reader.read_tensor(
        "PaliGemma/img/Transformer/encoder_norm/bias", dtype=heavy_dt
    ).contiguous()
    sd["vision_tower.head.weight"] = (
        reader.read_tensor("PaliGemma/img/head/kernel", dtype=heavy_dt).t().contiguous()
    )
    sd["vision_tower.head.bias"] = reader.read_tensor("PaliGemma/img/head/bias", dtype=heavy_dt).contiguous()

    # 2. PaliGemma-2B + ActionExpert-300M (streamed group by group to bound RAM)
    sd["paligemma_with_expert.embed_tokens.weight"] = reader.read_tensor(
        "PaliGemma/llm/embedder/input_embedding", dtype=heavy_dt
    ).contiguous()
    sd["paligemma_with_expert.vlm_norm.weight"] = reader.read_tensor(
        "PaliGemma/llm/final_norm/scale", dtype=torch.float32
    ).contiguous()
    sd["paligemma_with_expert.expert_norm.dense.weight"] = (
        reader.read_tensor("PaliGemma/llm/final_norm_1/Dense_0/kernel", dtype=heavy_dt).t().contiguous()
    )
    sd["paligemma_with_expert.expert_norm.dense.bias"] = reader.read_tensor(
        "PaliGemma/llm/final_norm_1/Dense_0/bias", dtype=heavy_dt
    ).contiguous()

    nh, nkv, hd = config.num_heads, config.num_kv_heads, config.head_dim
    pw, ew = config.paligemma_width, config.expert_width

    vlm_in_ln = reader.read_tensor("PaliGemma/llm/layers/pre_attention_norm/scale", dtype=torch.float32)
    vlm_post_ln = reader.read_tensor("PaliGemma/llm/layers/pre_ffw_norm/scale", dtype=torch.float32)
    vlm_q = reader.read_tensor("PaliGemma/llm/layers/attn/q_einsum/w", dtype=heavy_dt)
    vlm_kv = reader.read_tensor("PaliGemma/llm/layers/attn/kv_einsum/w", dtype=heavy_dt)
    vlm_o = reader.read_tensor("PaliGemma/llm/layers/attn/attn_vec_einsum/w", dtype=heavy_dt)
    for i in range(config.depth):
        lp = f"paligemma_with_expert.layers.{i}"
        sd[f"{lp}.vlm.input_layernorm.weight"] = vlm_in_ln[i].clone()
        sd[f"{lp}.vlm.post_attention_layernorm.weight"] = vlm_post_ln[i].clone()
        sd[f"{lp}.vlm.q_proj.weight"] = vlm_q[i].permute(0, 2, 1).reshape(nh * hd, pw).contiguous()
        sd[f"{lp}.vlm.k_proj.weight"] = vlm_kv[i, 0].permute(0, 2, 1).reshape(nkv * hd, pw).contiguous()
        sd[f"{lp}.vlm.v_proj.weight"] = vlm_kv[i, 1].permute(0, 2, 1).reshape(nkv * hd, pw).contiguous()
        sd[f"{lp}.vlm.o_proj.weight"] = vlm_o[i].permute(2, 0, 1).reshape(pw, nh * hd).contiguous()
    del vlm_in_ln, vlm_post_ln, vlm_q, vlm_kv, vlm_o
    gc.collect()

    vlm_gate_up = reader.read_tensor("PaliGemma/llm/layers/mlp/gating_einsum", dtype=heavy_dt)
    for i in range(config.depth):
        lp = f"paligemma_with_expert.layers.{i}"
        sd[f"{lp}.vlm.mlp.gate_proj.weight"] = vlm_gate_up[i, 0].t().contiguous()
        sd[f"{lp}.vlm.mlp.up_proj.weight"] = vlm_gate_up[i, 1].t().contiguous()
    del vlm_gate_up
    gc.collect()

    vlm_down = reader.read_tensor("PaliGemma/llm/layers/mlp/linear", dtype=heavy_dt)
    for i in range(config.depth):
        sd[f"paligemma_with_expert.layers.{i}.vlm.mlp.down_proj.weight"] = vlm_down[i].t().contiguous()
    del vlm_down
    gc.collect()

    exp_in_k = reader.read_tensor("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel", dtype=heavy_dt)
    exp_in_b = reader.read_tensor("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias", dtype=heavy_dt)
    exp_post_k = reader.read_tensor("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel", dtype=heavy_dt)
    exp_post_b = reader.read_tensor("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias", dtype=heavy_dt)
    exp_q = reader.read_tensor("PaliGemma/llm/layers/attn/q_einsum_1/w", dtype=heavy_dt)
    exp_kv = reader.read_tensor("PaliGemma/llm/layers/attn/kv_einsum_1/w", dtype=heavy_dt)
    exp_o = reader.read_tensor("PaliGemma/llm/layers/attn/attn_vec_einsum_1/w", dtype=heavy_dt)
    exp_gate_up = reader.read_tensor("PaliGemma/llm/layers/mlp_1/gating_einsum", dtype=heavy_dt)
    exp_down = reader.read_tensor("PaliGemma/llm/layers/mlp_1/linear", dtype=heavy_dt)
    for i in range(config.depth):
        lp = f"paligemma_with_expert.layers.{i}"
        sd[f"{lp}.expert.input_layernorm.dense.weight"] = exp_in_k[i].t().contiguous()
        sd[f"{lp}.expert.input_layernorm.dense.bias"] = exp_in_b[i].clone()
        sd[f"{lp}.expert.post_attention_layernorm.dense.weight"] = exp_post_k[i].t().contiguous()
        sd[f"{lp}.expert.post_attention_layernorm.dense.bias"] = exp_post_b[i].clone()
        sd[f"{lp}.expert.q_proj.weight"] = exp_q[i].permute(0, 2, 1).reshape(nh * hd, ew).contiguous()
        sd[f"{lp}.expert.k_proj.weight"] = exp_kv[i, 0].permute(0, 2, 1).reshape(nkv * hd, ew).contiguous()
        sd[f"{lp}.expert.v_proj.weight"] = exp_kv[i, 1].permute(0, 2, 1).reshape(nkv * hd, ew).contiguous()
        sd[f"{lp}.expert.o_proj.weight"] = exp_o[i].permute(2, 0, 1).reshape(ew, nh * hd).contiguous()
        sd[f"{lp}.expert.mlp.gate_proj.weight"] = exp_gate_up[i, 0].t().contiguous()
        sd[f"{lp}.expert.mlp.up_proj.weight"] = exp_gate_up[i, 1].t().contiguous()
        sd[f"{lp}.expert.mlp.down_proj.weight"] = exp_down[i].t().contiguous()
    del exp_in_k, exp_in_b, exp_post_k, exp_post_b, exp_q, exp_kv, exp_o, exp_gate_up, exp_down
    gc.collect()

    if "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w" in reader:
        mem_ln = reader.read_tensor("PaliGemma/llm/layers/mem_attn/mem_rms_norm/scale", dtype=torch.float32)
        mem_q = reader.read_tensor("PaliGemma/llm/layers/mem_attn/q_einsum_mem/w", dtype=heavy_dt)
        mem_kv = reader.read_tensor("PaliGemma/llm/layers/mem_attn/kv_einsum_mem/w", dtype=heavy_dt)
        mem_o = reader.read_tensor("PaliGemma/llm/layers/mem_attn/out_einsum_mem/w", dtype=heavy_dt)
        mem_ffn_k = reader.read_tensor("PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/kernel", dtype=heavy_dt)
        mem_ffn_b = reader.read_tensor("PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/bias", dtype=heavy_dt)
        mnh, mnkv, mhd = config.mem_num_heads, config.mem_num_kv_heads, config.mem_head_dim
        for i in range(config.depth):
            lp = f"paligemma_with_expert.layers.{i}"
            sd[f"{lp}.mem_attn.mem_rms_norm.weight"] = mem_ln[i].clone()
            sd[f"{lp}.mem_attn.q_proj.weight"] = mem_q[i].permute(0, 2, 1).reshape(mnh * mhd, ew).contiguous()
            sd[f"{lp}.mem_attn.k_proj.weight"] = mem_kv[i, 0].permute(0, 2, 1).reshape(mnkv * mhd, ew).contiguous()
            sd[f"{lp}.mem_attn.v_proj.weight"] = mem_kv[i, 1].permute(0, 2, 1).reshape(mnkv * mhd, ew).contiguous()
            sd[f"{lp}.mem_attn.out_proj.weight"] = mem_o[i].permute(2, 0, 1).reshape(ew, mnh * mhd).contiguous()
            sd[f"{lp}.mem_rms_norm_ffn.dense.weight"] = mem_ffn_k[i].t().contiguous()
            sd[f"{lp}.mem_rms_norm_ffn.dense.bias"] = mem_ffn_b[i].clone()
        del mem_ln, mem_q, mem_kv, mem_o, mem_ffn_k, mem_ffn_b
        gc.collect()

    # 3. Action / Time IO projections (kept in float32)
    for proj_name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        sd[f"{proj_name}.weight"] = reader.read_tensor(f"{proj_name}/kernel", dtype=torch.float32).t().contiguous()
        sd[f"{proj_name}.bias"] = reader.read_tensor(f"{proj_name}/bias", dtype=torch.float32).contiguous()

    # 4. Optional `PerceptualMemory` encoder
    if "mem_encoder/feature_encoder/encoder_static/kernel" in reader:
        sd["mem_encoder.encoder_static.weight"] = (
            reader.read_tensor("mem_encoder/feature_encoder/encoder_static/kernel", dtype=heavy_dt).t().contiguous()
        )
        sd["mem_encoder.encoder_static.bias"] = reader.read_tensor(
            "mem_encoder/feature_encoder/encoder_static/bias", dtype=heavy_dt
        ).contiguous()
        if "mem_encoder/feature_encoder/pos_proj/kernel" in reader:
            sd["mem_encoder.pos_proj.weight"] = (
                reader.read_tensor("mem_encoder/feature_encoder/pos_proj/kernel", dtype=heavy_dt).t().contiguous()
            )
            sd["mem_encoder.pos_proj.bias"] = reader.read_tensor(
                "mem_encoder/feature_encoder/pos_proj/bias", dtype=heavy_dt
            ).contiguous()

    leftover = reader.unconsumed_keys()
    if leftover:
        raise RuntimeError(f"Unconsumed Orbax keys ({len(leftover)}): {leftover}")

    # Validate 100% key and shape match on `meta` device (0 bytes RAM!)
    with torch.device("meta"):
        meta_model = Pi05Model(config)
        meta_sd = meta_model.state_dict()
    missing = sorted(set(meta_sd) - set(sd))
    unexpected = sorted(set(sd) - set(meta_sd))
    if missing or unexpected:
        raise RuntimeError(f"State dict key mismatch! missing={missing[:5]}, unexpected={unexpected[:5]}")
    for k, meta_v in meta_sd.items():
        if sd[k].shape != meta_v.shape:
            raise RuntimeError(f"Shape mismatch on {k}: got {sd[k].shape}, expected {meta_v.shape}")

    return sd


@torch.no_grad()
def streaming_flax_einsum_forward(
    reader: OCDBTReader,
    imgs: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    dtype_mm: torch.dtype,
    quantize_heavy_to_bf16: bool = False,
    static_img_emb: torch.Tensor | None = None,
    static_pos_emb: torch.Tensor | None = None,
    mem_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-RAM streaming reference executing un-transposed Flax `einsum` equations on GPU/CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = [im.to(dev) for im in imgs]
    lang_tokens = lang_tokens.to(dev)
    lang_masks = lang_masks.to(dev)
    x_t = x_t.to(dev)
    timestep = timestep.to(dev)
    if static_img_emb is not None:
        static_img_emb = static_img_emb.to(dev)
    if static_pos_emb is not None:
        static_pos_emb = static_pos_emb.to(dev)
    if mem_mask is not None:
        mem_mask = mem_mask.to(dev)

    def rt(key: str, dt: torch.dtype = dtype_mm) -> torch.Tensor:
        is_fp32_param = (
            key.endswith("/scale") and "Transformer" not in key
        ) or key in (
            "PaliGemma/img/embedding/kernel",
            "PaliGemma/img/embedding/bias",
            "PaliGemma/img/pos_embedding",
            "action_in_proj/kernel",
            "action_in_proj/bias",
            "action_out_proj/kernel",
            "action_out_proj/bias",
            "time_mlp_in/kernel",
            "time_mlp_in/bias",
            "time_mlp_out/kernel",
            "time_mlp_out/bias",
        )
        if quantize_heavy_to_bf16 and dt == torch.float32 and not is_fp32_param:
            return reader.read_tensor(key, dtype=torch.bfloat16, mark=False).float().to(dev)
        return reader.read_tensor(key, dtype=dt, mark=False).to(dev)

    conv_k = rt("PaliGemma/img/embedding/kernel", torch.float32)
    conv_b = rt("PaliGemma/img/embedding/bias", torch.float32)
    pos_emb = rt("PaliGemma/img/pos_embedding", torch.float32)

    ln0_s = rt("PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale")
    ln0_b = rt("PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias")
    ln1_s = rt("PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale")
    ln1_b = rt("PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias")
    fc1_k = rt("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    fc1_b = rt("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias")
    fc2_k = rt("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    fc2_b = rt("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias")
    q_k = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel")
    q_b = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias")
    k_k = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel")
    k_b = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias")
    v_k = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel")
    v_b = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias")
    o_k = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel")
    o_b = rt("PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias")

    img_tokens_list: list[torch.Tensor] = []
    for img in imgs:
        patches = img.float().unfold(1, 14, 14).unfold(2, 14, 14)
        stem = torch.einsum("bhwcij,ijcd->bhwd", patches, conv_k) + conv_b
        x = (stem.reshape(img.shape[0], 256, 1152) + pos_emb).to(dtype_mm)
        for i in range(27):
            y = F.layer_norm(x.float(), (1152,), None, None, 1e-6).to(dtype_mm) * ln0_s[i] + ln0_b[i]
            q = torch.einsum("btd,dnh->btnh", y, q_k[i]) + q_b[i]
            k = torch.einsum("btd,dnh->btnh", y, k_k[i]) + k_b[i]
            v = torch.einsum("btd,dnh->btnh", y, v_k[i]) + v_b[i]
            logits = torch.einsum("btnh,bsnh->bnts", q.float() * (72**-0.5), k.float())
            probs = F.softmax(logits, dim=-1).to(dtype_mm)
            enc = torch.einsum("bnts,bsnh->btnh", probs, v)
            x = x + torch.einsum("btnh,nhd->btd", enc, o_k[i]) + o_b[i]

            y2 = F.layer_norm(x.float(), (1152,), None, None, 1e-6).to(dtype_mm) * ln1_s[i] + ln1_b[i]
            h = F.gelu(torch.matmul(y2, fc1_k[i]) + fc1_b[i], approximate="tanh")
            x = x + torch.matmul(h, fc2_k[i]) + fc2_b[i]

        x = F.layer_norm(x.float(), (1152,), None, None, 1e-6).to(dtype_mm)
        x = x * rt("PaliGemma/img/Transformer/encoder_norm/scale") + rt("PaliGemma/img/Transformer/encoder_norm/bias")
        img_tokens_list.append(torch.matmul(x, rt("PaliGemma/img/head/kernel")) + rt("PaliGemma/img/head/bias"))

    del ln0_s, ln0_b, ln1_s, ln1_b, fc1_k, fc1_b, fc2_k, fc2_b, q_k, q_b, k_k, k_b, v_k, v_b, o_k, o_b
    gc.collect()

    emb_row = rt("PaliGemma/llm/embedder/input_embedding", dtype_mm)[lang_tokens.long()]
    lang_emb = emb_row * torch.tensor(math.sqrt(2048.0), dtype=dtype_mm, device=dev)
    del emb_row
    gc.collect()

    prefix_tokens = torch.cat([*img_tokens_list, lang_emb], dim=1)
    bsz = lang_tokens.shape[0]
    img_pad = torch.ones((bsz, sum(t.shape[1] for t in img_tokens_list)), dtype=torch.bool, device=dev)
    prefix_mask = torch.cat([img_pad, lang_masks.bool()], dim=1)
    prefix_ar = torch.zeros((prefix_tokens.shape[1],), dtype=torch.bool, device=dev)
    prefix_ar[0] = True

    act_tokens = (
        torch.matmul(x_t.float(), rt("action_in_proj/kernel", torch.float32))
        + rt("action_in_proj/bias", torch.float32)
    ).to(dtype_mm)
    t_emb = posemb_sincos(timestep, 1024, 4e-3, 4.0)
    t_emb = F.silu(torch.matmul(t_emb, rt("time_mlp_in/kernel", torch.float32)) + rt("time_mlp_in/bias", torch.float32))
    t_emb = F.silu(
        torch.matmul(t_emb, rt("time_mlp_out/kernel", torch.float32)) + rt("time_mlp_out/bias", torch.float32)
    )
    adarms_cond = t_emb.to(dtype_mm)

    suffix_mask = torch.ones((bsz, x_t.shape[1]), dtype=torch.bool, device=dev)
    suffix_ar = torch.zeros((x_t.shape[1],), dtype=torch.bool, device=dev)
    suffix_ar[0] = True

    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = torch.cumsum(input_mask.long(), dim=1) - 1

    mem_seq: torch.Tensor | None = None
    if static_img_emb is not None and "mem_encoder/feature_encoder/encoder_static/kernel" in reader:
        assert static_pos_emb is not None
        p_feat = F.silu(
            torch.matmul(static_pos_emb.to(dtype_mm), rt("mem_encoder/feature_encoder/pos_proj/kernel"))
            + rt("mem_encoder/feature_encoder/pos_proj/bias")
        )
        cat_feat = torch.cat([static_img_emb.to(dtype_mm), p_feat], dim=-1)
        mem_seq = torch.matmul(cat_feat, rt("mem_encoder/feature_encoder/encoder_static/kernel")) + rt(
            "mem_encoder/feature_encoder/encoder_static/bias"
        )

    def rms_norm_std(z: torch.Tensor, scale_1d_fp32: torch.Tensor) -> torch.Tensor:
        zf = z.float()
        normed = zf * torch.rsqrt(torch.mean(zf * zf, dim=-1, keepdim=True) + 1e-6)
        return (normed * (1.0 + scale_1d_fp32)).to(dtype_mm)

    def rms_norm_ada(z: torch.Tensor, cond: torch.Tensor, k_mat: torch.Tensor, b_vec: torch.Tensor):
        zf = z.float()
        normed = zf * torch.rsqrt(torch.mean(zf * zf, dim=-1, keepdim=True) + 1e-6)
        mod = torch.matmul(cond, k_mat) + b_vec
        sc, sh, gt = mod[:, None, :].chunk(3, dim=-1)
        return (normed * (1.0 + sc.float()) + sh.float()).to(dtype_mm), gt

    vlm_in_ln = rt("PaliGemma/llm/layers/pre_attention_norm/scale", torch.float32)
    vlm_post_ln = rt("PaliGemma/llm/layers/pre_ffw_norm/scale", torch.float32)
    vlm_q = rt("PaliGemma/llm/layers/attn/q_einsum/w")
    vlm_kv = rt("PaliGemma/llm/layers/attn/kv_einsum/w")
    vlm_o = rt("PaliGemma/llm/layers/attn/attn_vec_einsum/w")
    exp_in_k = rt("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel")
    exp_in_b = rt("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias")
    exp_post_k = rt("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel")
    exp_post_b = rt("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias")
    exp_q = rt("PaliGemma/llm/layers/attn/q_einsum_1/w")
    exp_kv = rt("PaliGemma/llm/layers/attn/kv_einsum_1/w")
    exp_o = rt("PaliGemma/llm/layers/attn/attn_vec_einsum_1/w")
    exp_gu = rt("PaliGemma/llm/layers/mlp_1/gating_einsum")
    exp_dn = rt("PaliGemma/llm/layers/mlp_1/linear")

    has_mem = "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w" in reader
    if has_mem:
        mem_ln = rt("PaliGemma/llm/layers/mem_attn/mem_rms_norm/scale", torch.float32)
        mem_q = rt("PaliGemma/llm/layers/mem_attn/q_einsum_mem/w")
        mem_kv = rt("PaliGemma/llm/layers/mem_attn/kv_einsum_mem/w")
        mem_o = rt("PaliGemma/llm/layers/mem_attn/out_einsum_mem/w")
        mem_fk = rt("PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/kernel")
        mem_fb = rt("PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/bias")

    # Read VLM MLP weights ONCE in bfloat16 (2.25 GiB + 1.12 GiB = 3.37 GiB total) to avoid 18x OCDBT Zarr chunk re-decompression!
    vlm_gu_all = reader.read_tensor("PaliGemma/llm/layers/mlp/gating_einsum", dtype=torch.bfloat16, mark=False).to(dev)
    vlm_dn_all = reader.read_tensor("PaliGemma/llm/layers/mlp/linear", dtype=torch.bfloat16, mark=False).to(dev)

    x0 = prefix_tokens
    x1 = act_tokens
    for i in range(18):
        p0 = rms_norm_std(x0, vlm_in_ln[i])
        p1, g1 = rms_norm_ada(x1, adarms_cond, exp_in_k[i], exp_in_b[i])

        q0 = torch.einsum("btd,ndh->btnh", p0, vlm_q[i])
        kv0 = torch.einsum("bsd,ckdh->cbskh", p0, vlm_kv[i])
        q1 = torch.einsum("btd,ndh->btnh", p1, exp_q[i])
        kv1 = torch.einsum("bsd,ckdh->cbskh", p1, exp_kv[i])

        q = apply_rope(torch.cat([q0, q1], dim=1), positions) * (256**-0.5)
        k = apply_rope(torch.cat([kv0[0], kv1[0]], dim=1), positions)
        v = torch.cat([kv0[1], kv1[1]], dim=1)

        qg = q.view(bsz, q.shape[1], 1, 8, 256)
        logits = torch.einsum("btkgh,bskh->bkgts", qg.float(), k.float())
        masked = torch.where(attn_mask[:, None, None, :, :], logits, OPENPI_MASK_NEG)
        probs = F.softmax(masked, dim=-1).to(dtype_mm)
        enc = torch.einsum("bkgts,bskh->btkgh", probs, v).reshape(bsz, q.shape[1], 8, 256)

        t0 = x0.shape[1]
        x0 = x0 + torch.einsum("btnh,nhd->btd", enc[:, :t0], vlm_o[i])
        x1 = x1 + torch.einsum("btnh,nhd->btd", enc[:, t0:], exp_o[i]) * g1

        x1_ffn_in = x1
        if has_mem and mem_seq is not None:
            assert mem_mask is not None
            x1_mn = rms_norm_std(x1, mem_ln[i])
            ms_mn = rms_norm_std(mem_seq, mem_ln[i])
            mq = torch.einsum("btd,ndh->btnh", x1_mn, mem_q[i])
            mkv = torch.einsum("bsd,ckdh->cbskh", ms_mn, mem_kv[i])
            s_mem = mem_seq.shape[1]
            q_pos = torch.arange(s_mem, s_mem + x1.shape[1], device=dev).unsqueeze(0).expand(bsz, -1)
            k_pos = torch.arange(0, s_mem, device=dev).unsqueeze(0).expand(bsz, -1)
            mq = apply_rope(mq, q_pos) * (256**-0.5)
            mk = apply_rope(mkv[0], k_pos)
            mlogits = torch.einsum("btkgh,bskh->bkgts", mq.view(bsz, x1.shape[1], 1, 4, 256).float(), mk.float())
            mmasked = torch.where(mem_mask[:, None, None, None, :], mlogits, OPENPI_MASK_NEG)
            mprobs = F.softmax(mmasked, dim=-1).to(dtype_mm)
            menc = torch.einsum("bkgts,bskh->btkgh", mprobs, mkv[1]).reshape(bsz, x1.shape[1], 4, 256)
            mvec = torch.einsum("btnh,nhd->btd", menc, mem_o[i])

            x1f = x1.float()
            x1_normed = x1f * torch.rsqrt(torch.mean(x1f * x1f, dim=-1, keepdim=True) + 1e-6)
            fmod = torch.matmul(mvec, mem_fk[i]) + mem_fb[i]
            fsc, fsh = fmod.chunk(2, dim=-1)
            x1_ffn_in = (x1_normed * (1.0 + fsc.float()) + fsh.float()).to(dtype_mm)

        f0 = rms_norm_std(x0, vlm_post_ln[i])
        gu0_i = vlm_gu_all[i].float() if dtype_mm == torch.float32 else vlm_gu_all[i]
        dn0_i = vlm_dn_all[i].float() if dtype_mm == torch.float32 else vlm_dn_all[i]
        m0 = F.gelu(torch.matmul(f0, gu0_i[0]), approximate="tanh") * torch.matmul(f0, gu0_i[1])
        x0 = x0 + torch.matmul(m0, dn0_i)

        f1, fg1 = rms_norm_ada(x1_ffn_in, adarms_cond, exp_post_k[i], exp_post_b[i])
        m1 = F.gelu(torch.matmul(f1, exp_gu[i, 0]), approximate="tanh") * torch.matmul(f1, exp_gu[i, 1])
        x1 = x1 + torch.matmul(m1, exp_dn[i]) * fg1

    x1_final, _ = rms_norm_ada(
        x1,
        adarms_cond,
        rt("PaliGemma/llm/final_norm_1/Dense_0/kernel"),
        rt("PaliGemma/llm/final_norm_1/Dense_0/bias"),
    )
    v_t = torch.matmul(x1_final.float(), rt("action_out_proj/kernel", torch.float32)) + rt(
        "action_out_proj/bias", torch.float32
    )
    out_img = img_tokens_list[0].float().cpu()
    out_vt = v_t.float().cpu()
    del vlm_gu_all, vlm_dn_all
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out_img, out_vt


def export_flax_parity_fixture(reader: OCDBTReader, has_percep: bool, out_path: Path) -> None:
    """Export deterministic inputs + exact Flax `einsum` outputs in both `float32` and `bfloat16`."""
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_dnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(1234)
        imgs = [torch.empty(1, 224, 224, 3).uniform_(-1.0, 1.0), torch.empty(1, 224, 224, 3).uniform_(-1.0, 1.0)]
        lang_tokens = torch.zeros((1, 64), dtype=torch.long)
        lang_tokens[0, :8] = torch.tensor([2, 18075, 908, 573, 12345, 28660, 108, 1])
        lang_masks = torch.zeros((1, 64), dtype=torch.bool)
        lang_masks[0, :8] = True
        x_t = torch.randn((1, 20, 32), dtype=torch.float32)
        t_step = torch.tensor([0.45], dtype=torch.float32)

        static_img = torch.randn((1, 512, 2048)) if has_percep else None
        static_pos = torch.randn((1, 512, 768)) if has_percep else None
        mem_mask = torch.zeros((1, 512), dtype=torch.bool) if has_percep else None
        if mem_mask is not None:
            mem_mask[:, :96] = True

        ref_bf16_img0, ref_bf16_vt = streaming_flax_einsum_forward(
            reader,
            imgs,
            lang_tokens,
            lang_masks,
            x_t,
            t_step,
            dtype_mm=torch.bfloat16,
            static_img_emb=static_img,
            static_pos_emb=static_pos,
            mem_mask=mem_mask,
        )
        gc.collect()
        ref_fp32_img0, ref_fp32_vt = streaming_flax_einsum_forward(
            reader,
            imgs,
            lang_tokens,
            lang_masks,
            x_t,
            t_step,
            dtype_mm=torch.float32,
            quantize_heavy_to_bf16=True,
            static_img_emb=static_img,
            static_pos_emb=static_pos,
            mem_mask=mem_mask,
        )
        fixture = {
            "imgs": imgs,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "x_t": x_t,
            "t_step": t_step,
            "static_img": static_img,
            "static_pos": static_pos,
            "mem_mask": mem_mask,
            "ref_bf16_img0": ref_bf16_img0,
            "ref_bf16_vt": ref_bf16_vt,
            "ref_fp32_img0": ref_fp32_img0,
            "ref_fp32_vt": ref_fp32_vt,
        }
        torch.save(fixture, out_path)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_dnn


@torch.no_grad()
def verify_from_fixture(ckpt_dir: Path, device: torch.device) -> dict[str, float]:
    """Verify converted PyTorch checkpoint on GPU/CPU against `flax_parity_fixture.pt`."""
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_dnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        fixture = torch.load(ckpt_dir / "flax_parity_fixture.pt", map_location=device, weights_only=False)
        imgs = [t.to(device) for t in fixture["imgs"]]
        lang_tokens = fixture["lang_tokens"].to(device)
        lang_masks = fixture["lang_masks"].to(device)
        x_t = fixture["x_t"].to(device)
        t_step = fixture["t_step"].to(device)
        static_img = fixture["static_img"].to(device) if fixture["static_img"] is not None else None
        static_pos = fixture["static_pos"].to(device) if fixture["static_pos"] is not None else None
        mem_mask = fixture["mem_mask"].to(device) if fixture["mem_mask"] is not None else None

        metrics: dict[str, float] = {}
        for prec, ref_img_key, ref_vt_key in [
            ("bfloat16", "ref_bf16_img0", "ref_bf16_vt"),
            ("float32", "ref_fp32_img0", "ref_fp32_vt"),
        ]:
            model, _ = load_pi05_checkpoint(ckpt_dir, device=device, precision=prec)
            pt_img0 = model.encode_images(imgs[0], exact_fp32_attn=True)
            kv = model.prefill_prefix_kv(lang_tokens, lang_masks, images=imgs, exact_fp32_attn=True)
            mem_tok = (
                model.mem_encoder(static_img, static_pos, None)
                if (static_img is not None and model.mem_encoder is not None)
                else None
            )
            pt_vt = model.denoise_step_from_kv(
                kv, x_t, t_step, mem_tokens=mem_tok, mem_mask=mem_mask, exact_fp32_attn=True
            )
            kv_sdpa = model.prefill_prefix_kv(lang_tokens, lang_masks, images=imgs, exact_fp32_attn=False)
            pt_vt_sdpa = model.denoise_step_from_kv(
                kv_sdpa, x_t, t_step, mem_tokens=mem_tok, mem_mask=mem_mask, exact_fp32_attn=False
            )

            ref_img = fixture[ref_img_key].to(device).float()
            ref_vt = fixture[ref_vt_key].to(device).float()
            img_diff = float((ref_img - pt_img0.float()).abs().max().item())
            vt_diff = float((ref_vt - pt_vt.float()).abs().max().item())
            vt_mean = float((ref_vt - pt_vt.float()).abs().mean().item())
            sdpa_diff = float((ref_vt - pt_vt_sdpa.float()).abs().max().item())
            metrics[f"{prec}_vs_flax_{prec}_siglip_max_diff"] = img_diff
            metrics[f"{prec}_vs_flax_{prec}_vt_max_diff"] = vt_diff
            metrics[f"{prec}_vs_flax_{prec}_vt_mean_diff"] = vt_mean
            metrics[f"{prec}_vs_flax_{prec}_sdpa_vt_max_diff"] = sdpa_diff
            print(
                f"[verify:{prec}] siglip_max_diff={img_diff:.3e} | "
                f"vt_max_diff={vt_diff:.3e} (mean={vt_mean:.3e}) | sdpa_max_diff={sdpa_diff:.3e}",
                flush=True,
            )

            if static_img is None and prec == "bfloat16":
                out_nomem = model.sample_actions_from_kv(kv_sdpa, noise=x_t, num_steps=10)
                model.enable_memory_modulation("modul_residual")
                dummy_mem = torch.randn((1, 64, 1024), dtype=torch.bfloat16, device=device)
                dummy_mask = torch.ones((1, 64), dtype=torch.bool, device=device)
                out_mod0 = model.sample_actions_from_kv(
                    kv_sdpa, noise=x_t, num_steps=10, mem_tokens=dummy_mem, mem_mask=dummy_mask
                )
                hook_diff = float((out_nomem - out_mod0).abs().max().item())
                metrics["bf16_step0_modul_residual_diff"] = hook_diff
                print(f"[verify:hook] bf16_step0_modul_residual_diff={hook_diff:.3e}", flush=True)
                assert hook_diff == 0.0

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        meta_path = ckpt_dir / "conversion_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["parity_metrics"] = metrics
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return metrics
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_dnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-RAM Orbax -> PyTorch pi0.5 converter & parity verifier.")
    parser.add_argument("--orbax-dir", type=Path, default=None, help="Path to `<ckpt>/79999` directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Target `.safetensors` directory.")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-token-len", type=int, default=64, help="Prompt token length (64 default, 128 for symbolic subgoal).")
    parser.add_argument("--skip-existing-weights", action="store_true", help="Skip rewriting model.safetensors if present.")
    parser.add_argument("--export-fixture", action="store_true", help="Export Flax-einsum reference fixture.")
    parser.add_argument("--verify-from-fixture", action="store_true", help="Verify `.safetensors` vs fixture.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.orbax_dir is not None:
        t0 = time.perf_counter()
        orbax_dir = args.orbax_dir.resolve()
        params_dir = orbax_dir / "params" if (orbax_dir / "params").exists() else orbax_dir
        print(f"Opening OCDBT store at {params_dir} ...", flush=True)
        reader = OCDBTReader(params_dir)
        has_mem_attn = "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w" in reader
        has_percep = "mem_encoder/feature_encoder/encoder_static/kernel" in reader

        config = Pi05Config(
            max_token_len=int(args.max_token_len),
            mem_mod_mode="mme_vla_exact" if has_mem_attn else "none",
            has_perceptual_mem_encoder=has_percep,
            perceptual_mem=PerceptualMemConfig() if has_percep else None,
            precision=args.precision,
        )
        if args.skip_existing_weights and (out_dir / "model.safetensors").exists():
            print(f"Skipping existing weights at {out_dir / 'model.safetensors'}", flush=True)
            total_params = 0
            total_bytes = (out_dir / "model.safetensors").stat().st_size
        else:
            sd = stream_convert_orbax_to_safetensors(reader, config)
            total_params = sum(v.numel() for v in sd.values())
            total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
            print(
                f"Converted {len(reader)} Orbax arrays -> {len(sd)} PyTorch tensors "
                f"({total_params / 1e9:.3f}B params, {total_bytes / (1024**3):.2f} GiB) in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

            config.save_json(out_dir / "config.json")
            save_file(sd, str(out_dir / "model.safetensors"))
            del sd
            gc.collect()

        norm_cand = orbax_dir / "assets" / "robomme" / "norm_stats.json"
        if norm_cand.exists():
            (out_dir / "norm_stats.json").write_text(norm_cand.read_text())

        if args.export_fixture:
            t_f = time.perf_counter()
            print("Exporting streaming Flax-einsum parity fixture ...", flush=True)
            export_flax_parity_fixture(reader, has_percep, out_dir / "flax_parity_fixture.pt")
            print(f"Exported parity fixture in {time.perf_counter() - t_f:.2f}s", flush=True)

        meta: dict[str, Any] = {
            "source_orbax_dir": str(orbax_dir),
            "num_orbax_arrays": len(reader),
            "total_params": total_params,
            "total_bytes": total_bytes,
            "precision": args.precision,
            "mem_mod_mode": config.mem_mod_mode,
            "has_perceptual_mem_encoder": has_percep,
        }
        (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Saved converted checkpoint to {out_dir} (total {time.perf_counter() - t0:.2f}s)", flush=True)

    if args.verify_from_fixture:
        verify_from_fixture(out_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
