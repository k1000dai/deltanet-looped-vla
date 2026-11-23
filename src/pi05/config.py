"""Configuration dataclasses for PyTorch π0.5 and MME-VLA / DeltaMem-π0.5 integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

MemModMode = Literal["none", "modul_residual", "mme_vla_exact"]


@dataclass
class PerceptualMemConfig:
    """Configuration for MME-VLA `perceptual-framesamp-modul` compatibility."""

    budget: int = 512
    num_views: int = 1
    token_per_image: int = 16
    streaming_obs_horizon: int = 16
    pool_type: str = "mean"
    use_pos_emb: bool = True
    use_state_emb: bool = False
    img_input_dim: int = 2048
    pos_input_dim: int = 768
    pos_hidden_dim: int = 768
    state_input_dim: int = 8
    state_hidden_dim: int = 512
    memory_token_dim: int = 1024


@dataclass
class Pi05Config:
    """Exact architectural configuration for `Yinpei/pi05_baseline` and `HistoryBlock`-augmented π0.5."""

    # PaliGemma-2B VLM stream (Stream 0)
    paligemma_width: int = 2048
    paligemma_mlp_dim: int = 16384

    # Gemma-300M Action Expert stream (Stream 1)
    expert_width: int = 1024
    expert_mlp_dim: int = 4096

    # Shared joint attention geometry across all 18 transformer blocks
    depth: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 257152
    rope_max_wavelength: float = 10_000.0
    rms_norm_eps: float = 1e-6

    # SigLIP-So400m/14 vision encoder
    siglip_width: int = 1152
    siglip_depth: int = 27
    siglip_mlp_dim: int = 4304
    siglip_num_heads: int = 16
    siglip_patch_size: int = 14
    image_size: int = 224
    layernorm_eps: float = 1e-6

    # Action / prompt dimensions (MME-VLA-Suite convention)
    action_dim: int = 32  # Padded internal action dim; RoboMME uses first `real_action_dim=8`
    real_action_dim: int = 8
    action_horizon: int = 20
    exec_horizon: int = 16
    max_token_len: int = 64
    discrete_state_input: bool = False  # Confirmed False in MME-VLA-Suite `training/config.py` L564,596
    num_views: int = 2  # `base_0_rgb` + `left_wrist_0_rgb` (2 cameras in RoboMMEInputs)

    # Flow-matching ODE schedule
    num_inference_steps: int = 10
    time_min_period: float = 4e-3
    time_max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    # 18-layer ActionExpert memory modulation hook
    # - "none": pure memoryless `pi05_baseline`
    # - "modul_residual": 18L `MemoryAttention` (4Q, 1KV, head_dim=256) + zero-init residual FiLM
    #   (`x + norm(x)*scale + shift`), giving bit-exact 0.0 diff vs `pi05_baseline` at step 0
    #   and when `mem_tokens=None`, with safe zeroing on all-False `mem_mask` rows.
    # - "mme_vla_exact": exact upstream `HistoryBlock` (`MemoryRMSNorm(x, mem_attn(...))` before
    #   `pre_ffw_norm`) for bit-exact loading of `Yinpei/perceptual-framesamp-modul` (44.51% SR).
    mem_mod_mode: MemModMode = "none"
    mem_num_heads: int = 4
    mem_num_kv_heads: int = 1
    mem_head_dim: int = 256
    has_perceptual_mem_encoder: bool = False
    perceptual_mem: PerceptualMemConfig | None = None

    # Compute precision
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pi05Config:
        d = dict(data)
        pm = d.pop("perceptual_mem", None)
        p_cfg = PerceptualMemConfig(**pm) if isinstance(pm, dict) else pm
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered, perceptual_mem=p_cfg)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load_json(cls, path: str | Path) -> Pi05Config:
        return cls.from_dict(json.loads(Path(path).read_text()))
