"""Configuration & Controlled Ablation Presets for Phase-2 `DeltaMem-π0.5` (`src/pi05_mem/`).

Covers all Phase-2 architectures plugging into `Yinpei/pi05_baseline` (`3.44B`: `PaliGemma-2B`
width=2048, depth=18 + `ActionExpert-300M` width=1024, depth=18 + `SigLIP-400M`):
  - `p2_b0_pi05_base`:   Zero-memory `Yinpei/pi05_baseline` pass-through (`0 B` state, `0.0` diff).
  - `p2_b1_t2mem`:       `T²Mem`-style 1-Layer Online Self-Supervised Fast-Weight TTT (`arXiv:2609.36720`)
                         with mini-batch residual reconstruction update & observation-grounded readout.
  - `p2_b2_framesamp`:   Non-privileged strided Perceptual Frame-Sampling buffer (`FrameSamp+Modul`).
  - `p2_m1_gdn2l_ms`:    Multi-Timescale (`Short / Mid / Long`) 2-Layer `fp32` Gated DeltaNet (`IDEA.md`
                         3-scale decay + Cross-Scale Consolidation Gate + `M5` 2L read-before-write +
                         dual-view `front`+`wrist`+proprio+gripper-edge + Boundary Forget Gate).
  - `p2_m2_deltamem`:    **Flagship `DeltaMem-π0.5`**: `p2_m1_gdn2l_ms` (`O(1)` GDN readout tokens) +
                         Non-Privileged `GDN-Surprise` & Gripper-Event Visual Landmark Bank (`K=14`
                         keyframes x `6` spatial slots = `84` tokens; total `<= 104` tokens) + Dense
                         Training-Time Grounded Supervision (`<x, y>`, counts, subgoals) + optional
                         Zero-Oracle Self-Predicted `<x, y>` Subgoal Token Injection.
  - `p2_m3_deltamem_loop`: Stage-C Zero-Init Residual Multi-Hop Query Refinement (`m^(1) + tanh(gamma)*dm^(2)`,
                         `gamma_0 = 0`) + `M_CRN = 8` Adaptive Uncertainty Gate (provably `>= L=1` at init).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Tuple

ArchPi05Mem = Literal[
    "p2_b0_pi05_base",
    "p2_b1_t2mem",
    "p2_b2_framesamp",
    "p2_m1_gdn2l_ms",
    "p2_m2_deltamem",
    "p2_m3_deltamem_loop",
    "p2_m4_deltamem_aligned",
    "p2_m5_hybrid_surprise_gdn",
    "p2_m7_grounded_hybrid",
    "p2_m8_vlmprefix_surprise_hybrid",
]

ALL_PI05_MEM_ARCHS: Tuple[ArchPi05Mem, ...] = (
    "p2_b0_pi05_base",
    "p2_b1_t2mem",
    "p2_b2_framesamp",
    "p2_m1_gdn2l_ms",
    "p2_m2_deltamem",
    "p2_m3_deltamem_loop",
    "p2_m4_deltamem_aligned",
    "p2_m5_hybrid_surprise_gdn",
    "p2_m7_grounded_hybrid",
    "p2_m8_vlmprefix_surprise_hybrid",
)

TimescaleMode = Literal[
    "three_band_consolidate",     # IDEA.md Model 5: Short/Mid/Long learnable bands + Consolidation Gate
    "three_band_no_consolidate",  # IDEA.md Model 4: Short/Mid/Long learnable bands, no Consolidation Gate
    "three_band_fixed",           # IDEA.md Model 3: Short/Mid/Long fixed geometric decay
    "uniform_same_decay",         # IDEA.md Model 2: Identical fixed decay across all heads
    "log_uniform_learnable",      # Phase-1 M5 log-uniform [8, 2048] learnable decay
]

TrainStage = Literal[
    "stage_a",  # Freeze pi05 backbone; train Memory + 18L ActionExpert Modulators + Aux Grounding Heads
    "stage_b",  # Alternating Co-Adaptation: freeze/slow-LR Memory; train ActionExpert + PaliGemma LoRA
    "stage_c",  # Residual Multi-Hop: freeze Stage-B L=1; train zero-init Hop-2 + CRN Halting Gate
]

ModulatorMode = Literal[
    "modul_zero_init",   # Exact 0.0 step-0 diff vs Yinpei/pi05_baseline (zero-init residual + FiLM)
    "modul_residual",    # Alias matching src/pi05/config.py MemModMode
    "mme_vla_exact",     # Upstream HistoryBlock MemoryRMSNorm(x, mem_mod_vec) for framesamp ckpt compat
    "grafted_residual",  # P2-M7A: zero-init scalar gate guarding grafted HistoryBlock on symbolic_grounded_subgoal_pt
]


@dataclass
class Pi05MemConfig:
    """Complete serializable configuration for `DeltaMemPi05Policy`."""

    arch: ArchPi05Mem = "p2_m2_deltamem"
    train_stage: TrainStage = "stage_a"

    # ---- Backbone (`Yinpei/pi05_baseline`: PaliGemma-2B + ActionExpert-300M) ----
    vlm_width: int = 2048
    vlm_mlp_dim: int = 16_384
    expert_width: int = 1024
    expert_mlp_dim: int = 4096
    num_layers: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 257_152

    # Memory Cross-Attention inside each of the 18 ActionExpert layers (matches HistoryBlock)
    mem_token_dim: int = 1024
    mem_attn_heads: int = 4
    mem_attn_kv_heads: int = 1
    mem_attn_head_dim: int = 256
    modulator_mode: ModulatorMode = "modul_zero_init"
    hook_dropout: float = 0.05

    # LoRA settings for Stage-B PaliGemma-2B or regularized ActionExpert-300M PEFT co-adaptation
    use_vlm_lora: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_target_ffn: bool = True
    use_expert_lora: bool = False
    expert_lora_rank: int = 16
    expert_lora_alpha: float = 16.0
    expert_lora_dropout: float = 0.05
    freeze_dense_expert_in_stage_b: bool = False

    # ---- Observation, Action & Horizon Conventions (`MME-VLA` / `openpi`) ----
    vis_dim: int = 2048              # PaliGemma SigLIP output token dim (auto-adapts if 768/1152)
    front_tokens_hist: int = 64      # 8x8 spatial tokens per history frame (matches robomme_pi05_cache)
    wrist_tokens_hist: int = 16      # 4x4 (16) wrist spatial tokens per history frame (matches robomme_pi05_cache)
    front_tokens_curr: int = 256     # 16x16 (256) current-frame front tokens for Pi05Model VLM
    wrist_tokens_curr: int = 256     # 16x16 (256) current-frame wrist tokens for Pi05Model VLM
    max_token_len: int = 64          # Upstream HistoryPi0Config max_token_len (128 with symbolic subgoal)
    discrete_state_input: bool = False  # Upstream config.py L564/L596 sets discrete_state_input=False!

    state_dim: int = 8
    raw_action_dim: int = 8
    padded_action_dim: int = 32      # Upstream PadStatesAndActions(32)
    action_horizon: int = 20         # H_pred = 20
    exec_horizon: int = 16           # H_exec = 16
    exclusion_window: int = 16       # Invariant A3 causal exclusion cutoff (preserves 100% of is_demo!)
    write_stride_exec: int = 2       # Write every 2 execution frames (5 Hz -> captures fast picks/swings!)
    write_stride_demo: int = 2       # Write every 2 demo frames (adaptive stride 4 if T_demo > 512)
    adaptive_demo_threshold: int = 512
    long_demo_stride: int = 4

    # ---- Internal Memory Encoder & Multi-Timescale 2-Layer `fp32` GDN (`IDEA.md`) ----
    d_mem: int = 512
    mem_enc_heads: int = 8
    num_write_slots: int = 8         # N_w object/event slots per written frame
    slot_conv_kernel: int = 4        # Per-slot CausalConv1D along frame axis
    use_spatial_softmax: bool = True
    spatial_kp_front: int = 32
    spatial_kp_wrist: int = 16

    # 3-Timescale GDN (`Short` + `Mid` + `Long` = 4 + 4 + 4 = 12 heads, K=64 native Hopper SMEM fast-path!)
    gdn_heads: int = 12              # Divisible by 3 in 3-band mode (4 Short, 4 Mid, 4 Long)
    gdn_d_k: int = 64
    gdn_d_v: int = 64
    gdn_layers: int = 2              # Winning Phase-1 M5 2-layer read-before-write composition
    beta_max: float = 1.5            # Expressive write/erase in (0, beta_max); motion-gated on static waits
    use_motion_beta_gate: bool = True
    motion_beta_bias: float = -2.0
    timescale_mode: TimescaleMode = "three_band_consolidate"
    short_halflife_bounds: Tuple[float, float] = (4.0, 32.0)
    mid_halflife_bounds: Tuple[float, float] = (32.0, 256.0)
    long_halflife_bounds: Tuple[float, float] = (256.0, 2048.0)
    readout_eps: float = 0.1         # Invariant C3 damped RMSNorm readout denominator
    num_gdn_readout_tokens: int = 18 # 6 Short + 6 Mid + 6 Long observation-queried 1024-D tokens

    # ---- Instruction / Visual-Surprise Cross-Session Boundary Forget Gate ----
    use_boundary_reset_gate: bool = True
    boundary_reset_strength: float = 12.0  # Adds -12.0 * r_t to g_s0 (exp(-12) ~ 6e-6 state flush!)
    boundary_flush_threshold: float = 0.5  # Flushes pre-boundary landmark bank slots when r_t > 0.5

    # ---- Non-Privileged Dual-Memory (`GDN-Surprise` + Gripper-Event Visual Landmark Bank) ----
    use_landmark_bank: bool = True
    landmark_encoder: Literal["multiscale_pyramid", "perceiver"] = "multiscale_pyramid"
    num_landmarks: int = 8           # K = 8 keyframes x 16 multiscale pyramid tokens = 128 tokens (or 6x16=96)
    tokens_per_landmark: int = 16    # 4x4 grid fused with zero-init 8x8 sub-cells + 2x2 context (Dual-Stream Eq. 4)
    demo_landmark_quota: int = 4     # Up to 4 slots reserved for conditioning video demo when T_demo > 0
    num_persistent_tokens: int = 4   # Titans input-independent persistent task memory tokens (zero-init)
    surprise_ema_beta: float = 0.80  # Titans surprise momentum across 10-35 frame physical swaps/reveals
    yaad_huber_delta: float = 2.5    # Miras Yaad Huber bound on value residual outliers
    surprise_weight: float = 1.0
    gripper_spike_weight: float = 1.5
    motion_spike_weight: float = 0.5

    # ---- Memory Classifier-Free Guidance (`Mem-CFG`) & Stage-B Pointer Jitter ----
    mem_cfg_dropout: float = 0.10    # Stage-B condition-masking dropout (`mem_mask = False` -> 0 residual)
    default_cfg_scale: float = 1.0   # Eval Mem-CFG weight w_cfg (e.g. 1.0 unguided, 1.25-1.50 guided)
    self_xy_jitter_std: float = 8.0 / 112.0  # MME-VLA dataset.py L171 ±8px jitter in Stage B

    # ---- Frame-Sampling Baseline (`p2_b2_framesamp`) & `T²Mem` Baseline (`p2_b1_t2mem`) ----
    framesamp_num_frames: int = 32
    framesamp_tokens_per_frame: int = 8  # 32 * 8 = 256 tokens
    t2mem_heads: int = 4
    t2mem_head_dim: int = 128            # 4 * 128 = 512 hidden dim
    t2mem_mini_batch_frames: int = 4     # Chunkwise fast-weight update every 4 written frames
    t2mem_base_lr: float = 0.02
    t2mem_max_lr: float = 0.15
    t2mem_max_grad_norm: float = 5.0
    t2mem_num_query_tokens: int = 32     # Observation-grounded readout tokens -> [B, 32, 1024]

    # ---- Dense Training-Only Grounded Supervision & Zero-Oracle Self-Subgoal Tokens ----
    use_grounded_aux: bool = True
    xy_loss_weight: float = 0.20         # 2D <x, y> coordinate + spatial heatmap KL loss
    count_loss_weight: float = 0.15      # Pass/pick/swing completed & remaining count loss
    subgoal_loss_weight: float = 0.15    # Subgoal stage/color/latent distillation loss
    gripper_bce_weight: float = 0.05     # Auxiliary chunk gripper transition loss
    max_count_classes: int = 12          # Counts 0..11
    num_stage_classes: int = 15          # Matches analysis/subgoal_grounding.py STAGE_VOCAB
    num_color_classes: int = 9           # Matches analysis/subgoal_grounding.py COLOR_VOCAB
    subgoal_latent_dim: int = 512

    # Optional Test-Time Self-Predicted Grounded `<x, y>` + Count/Stage Token Injection (`0` oracle!)
    use_self_subgoal_tokens: bool = True
    num_self_subgoal_tokens: int = 2     # Appends 2 zero-init 1024-D self-predicted grounded tokens

    # ---- Stage-C Zero-Init Residual Multi-Hop Query Refinement + CRN Gate ----
    use_residual_hop2: bool = False
    crn_draws: int = 8                   # M_CRN = 8 paired flow noise draws in Stage C
    crn_quantile: float = 0.58           # Invariant B4 relative-quantile calibration
    crn_min_delta: float = 1e-5

    # ---- Track-2 Hybrid Surprise-Keyframe + Multi-Timescale GDN (`p2_m5_hybrid_surprise_gdn`, `#265`+`#266`!) ----
    hybrid_num_keyframes: int = 32       # K_hyb = 32 keyframes -> 512t zero-aliasing full-resolution canvas (#265/#266!)
    hybrid_canvas_tokens: int = 512      # Full 512-token canvas (also supports 256t via build_eval_adapter_256!)
    q_rope_offset: int | None = None     # Set to 512 on Track-2 mme_vla_exact
    hybrid_anchor_ratio: float = 0.875   # 28/32 time-uniform anchors + 4 top GDN-surprise/gripper peaks (#266!)
    hybrid_use_landmarks: bool = False   # Opt-in K=8 Multi-Scale GDN-Surprise Landmarks in HybridBridge KV (#275!)
    dense_prefix_frames: int = 0         # Auto-set to 32 on p2_m5/p2_m7 so t < 32 matches FrameSamplingBuffer 100%!
    hybrid_bridge_heads: int = 8         # Cross-attn heads fusing 2L-GDN + Wrist + <x,y>/Count pointers

    # ---- Unified Grounded-Hybrid-π0.5 (`p2_m7_grounded_hybrid`, `#313` P2-M7A & P2-M7B + `#395` P2-M11 Continuous VLM Grounding) ----
    graft_gate_init: float = 0.0         # Step-0 scalar gate on GraftedMemoryRMSNorm (`0.0` = bit-exact symbolic VLM!)
    subgoal_dg_prob: float = 0.65        # Prob of conditioning VLM prefix on DeltaGround v2_on100 predicted subgoal
    subgoal_oracle_prob: float = 0.15    # Prob of conditioning VLM prefix on jittered grounded_subgoal_online (rest = None)
    subgoal_jitter_px: int = 6           # Truncated Gaussian pixel jitter on training-only oracle subgoals
    use_continuous_vlm_ground: bool = False  # `#395(2)` `P2-M11`: Continuous VLM-Grounded Spatial Bridge in `HistoryBlock` KV (`max_token_len=64`!)
    cont_vlm_jitter_cells: float = 0.85      # Training spatial heatmap jitter (`~12 px` / `<1` cell) blocking seat memorization!
    use_reactive_precision_mode: bool = False  # `#405` `P2-M12`: Zero-oracle `StopCube` (`excl=1, stride=1`) + `InsertPeg` (`L=2 hop2` + climax landmarks)!
    reactive_stopcube_excl: int = 1          # Causal `t-1` live cube velocity & pass-count cutoff on `StopCube` (`0` robot copy-cat shortcut!)

    # ---- VLM-Prefix GDN-Surprise Reveal Keyframes (`p2_m8_vlmprefix_surprise_hybrid`, `#329` & `#331`) ----
    vlm_prefix_surprise_keyframes: int = 4     # K_vlm = 4 chronological GDN-Surprise reveal/highlight keyframes
    vlm_prefix_tokens_per_frame: int = 64      # 64 (8x8 -> 256t) or 16 (4x4 -> 64t, #331 gentle RoPE/softmax)
    vlm_prefix_demo_quota: int = 3             # demo_quota = 3 when demo > 0 (VideoUnmask*, VideoPlace*, VideoRepick)
    vlm_prefix_early_quota: int = 2            # early_quota = 2 pure visual s_norm when demo == 0 (PickHighlight, ButtonUnmask*)
    vlm_prefix_drop_prob: float = 0.15         # #331 15% whole-history extra_prefix_mask=False dropout in Stage B

    # ---- Flow-Matching Loss & Sampling (`#239`/`#240`/`#241` Clean Spatial + 32-D ODE + fp32 Master Fixes!) ----
    loss_on_valid_8d_only: bool = False  # Keep all 32 ODE dims supervised so x_t[:, :, 8:32] denoises cleanly!
    padded_dim_loss_weight: float = 0.25 # Weight dims 8..31 at 0.25x, dims 0..6 at 1.0x, dim 7 at 1.5x
    fm_loss_scale: float = 4.0           # Scale 32-D weighted mean up 4x so fm_loss dominates aux in Stage B
    gripper_dim_weight: float = 1.5      # Extra weight on dim 7 gripper actuation accuracy
    policy_weight_decay: float = 1e-10   # Matches upstream openpi/training/optimizer.py L73 (1e-10, NOT 1e-2!)
    stage_b_aux_scale: float = 0.02      # Keep aux weight tiny in Stage B so fm_loss dominates policy grad
    subgrid_crop_aug: bool = False       # NEVER affine-warp post-SigLIP position-baked visual tokens!
    subgrid_scale_min: float = 0.94
    subgrid_max_shift: float = 0.03
    feat_jitter_std: float = 0.0         # Zero feature noise into PaliGemma-2B visual tokens!
    ode_steps: int = 10
    ode_solver: Literal["euler", "midpoint"] = "euler"

    def __post_init__(self) -> None:
        if self.arch == "p2_b0_pi05_base":
            self.use_landmark_bank = False
            self.use_self_subgoal_tokens = False
            self.use_residual_hop2 = False
        elif self.arch == "p2_b1_t2mem":
            self.use_landmark_bank = False
            self.use_residual_hop2 = False
        elif self.arch == "p2_b2_framesamp":
            self.use_landmark_bank = False
            self.use_residual_hop2 = False
        elif self.arch == "p2_m1_gdn2l_ms":
            self.use_landmark_bank = False
            self.use_residual_hop2 = False
        elif self.arch == "p2_m2_deltamem":
            self.use_landmark_bank = True
            self.use_residual_hop2 = False
        elif self.arch == "p2_m3_deltamem_loop":
            self.use_landmark_bank = True
            self.use_residual_hop2 = True
        elif self.arch == "p2_m4_deltamem_aligned":
            self.use_landmark_bank = True
            self.modulator_mode = "mme_vla_exact"
            self.mem_cfg_dropout = 0.0
        elif self.arch == "p2_m5_hybrid_surprise_gdn":
            self.use_landmark_bank = bool(self.hybrid_use_landmarks)
            self.modulator_mode = "mme_vla_exact"
            self.mem_cfg_dropout = 0.0
            if self.dense_prefix_frames == 0:
                self.dense_prefix_frames = 32
        elif self.arch == "p2_m7_grounded_hybrid":
            self.use_landmark_bank = bool(self.hybrid_use_landmarks)
            if self.modulator_mode in ("modul_zero_init", "modul_residual"):
                self.modulator_mode = "grafted_residual"
            self.mem_cfg_dropout = 0.0
            if (not self.use_continuous_vlm_ground) and self.max_token_len < 128:
                self.max_token_len = 128
            if self.dense_prefix_frames == 0:
                self.dense_prefix_frames = 32
        elif self.arch == "p2_m8_vlmprefix_surprise_hybrid":
            self.use_landmark_bank = bool(self.hybrid_use_landmarks)
            self.modulator_mode = "mme_vla_exact"
            self.mem_cfg_dropout = 0.0
            if self.dense_prefix_frames == 0:
                self.dense_prefix_frames = 32

        if self.modulator_mode in ("mme_vla_exact", "grafted_residual"):
            # `mme_vla_exact` & `grafted_residual` align queries at `q_rope_offset = 512`!
            self.mem_cfg_dropout = 0.0
            if self.q_rope_offset is None:
                self.q_rope_offset = 512

        if self.train_stage == "stage_c":
            self.use_residual_hop2 = True

    @property
    def max_mem_tokens(self) -> int:
        """Maximum number of `1024-D` memory tokens exposed to `ActionExpert` (`3.4x` smaller than 512!)."""
        if self.arch == "p2_b0_pi05_base":
            return 0
        if self.arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid"):
            return int(self.hybrid_canvas_tokens)
        extra_sg = self.num_self_subgoal_tokens if (self.use_grounded_aux and self.use_self_subgoal_tokens) else 0
        if self.arch == "p2_b1_t2mem":
            return self.t2mem_num_query_tokens + extra_sg
        if self.arch == "p2_b2_framesamp":
            return self.framesamp_num_frames * self.framesamp_tokens_per_frame + extra_sg
        base = self.num_persistent_tokens + self.num_gdn_readout_tokens
        if self.use_landmark_bank:
            lm_tok = 16 if self.landmark_encoder == "multiscale_pyramid" else self.tokens_per_landmark
            base += self.num_landmarks * lm_tok
        return base + extra_sg

    @property
    def gdn_state_bytes(self) -> int:
        """Persistent `fp32` GDN state bytes (including fused `v=1` occupancy vector)."""
        if self.arch not in (
            "p2_m1_gdn2l_ms",
            "p2_m2_deltamem",
            "p2_m3_deltamem_loop",
            "p2_m4_deltamem_aligned",
            "p2_m5_hybrid_surprise_gdn",
            "p2_m7_grounded_hybrid",
            "p2_m8_vlmprefix_surprise_hybrid",
        ):
            return 0
        per_layer_floats = self.gdn_heads * (self.gdn_d_v + 1) * self.gdn_d_k
        return self.gdn_layers * per_layer_floats * 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pi05MemConfig":
        valid = set(cls.__dataclass_fields__.keys())
        filtered = {}
        for k, v in d.items():
            if k in valid:
                if k.endswith("_bounds") and isinstance(v, list):
                    v = (float(v[0]), float(v[1]))
                filtered[k] = v
        return cls(**filtered)

    @classmethod
    def tiny_test_preset(cls, arch: ArchPi05Mem = "p2_m2_deltamem", **overrides: Any) -> "Pi05MemConfig":
        """Fast CPU unit-test preset exercising 100% of code paths in milliseconds."""
        defaults: Dict[str, Any] = {
            "arch": arch,
            "vlm_width": 128,
            "vlm_mlp_dim": 256,
            "expert_width": 96,
            "expert_mlp_dim": 192,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 32,
            "vocab_size": 512,
            "mem_token_dim": 96,
            "mem_attn_heads": 4,
            "mem_attn_kv_heads": 1,
            "mem_attn_head_dim": 24,
            "vis_dim": 128,
            "front_tokens_hist": 64,
            "wrist_tokens_hist": 16,
            "front_tokens_curr": 64,
            "wrist_tokens_curr": 16,
            "max_token_len": 48 if arch == "p2_m7_grounded_hybrid" else 24,
            "d_mem": 96,
            "mem_enc_heads": 4,
            "num_write_slots": 4,
            "gdn_heads": 6,
            "gdn_d_k": 32,
            "gdn_d_v": 32,
            "num_gdn_readout_tokens": 9,
            "num_persistent_tokens": 2,
            "num_landmarks": 4,
            "tokens_per_landmark": 16,
            "demo_landmark_quota": 2,
            "hybrid_num_keyframes": 6,
            "hybrid_canvas_tokens": 96,
            "dense_prefix_frames": 6 if arch in ("p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid") else 0,
            "q_rope_offset": 128 if arch in ("p2_m4_deltamem_aligned", "p2_m5_hybrid_surprise_gdn", "p2_m7_grounded_hybrid", "p2_m8_vlmprefix_surprise_hybrid") else None,
            "vlm_prefix_surprise_keyframes": 2 if arch == "p2_m8_vlmprefix_surprise_hybrid" else 4,
            "vlm_prefix_tokens_per_frame": 16 if arch == "p2_m8_vlmprefix_surprise_hybrid" else 64,
            "vlm_prefix_demo_quota": 1 if arch == "p2_m8_vlmprefix_surprise_hybrid" else 3,
            "vlm_prefix_early_quota": 1 if arch == "p2_m8_vlmprefix_surprise_hybrid" else 2,
            "vlm_prefix_drop_prob": 0.0 if arch == "p2_m8_vlmprefix_surprise_hybrid" else 0.15,
            "hybrid_bridge_heads": 4,
            "framesamp_num_frames": 8,
            "framesamp_tokens_per_frame": 4,
            "t2mem_heads": 2,
            "t2mem_head_dim": 48,
            "t2mem_num_query_tokens": 8,
            "subgoal_latent_dim": 64,
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "hook_dropout": 0.0,
            "ode_steps": 4,
        }
        defaults.update(overrides)
        return cls(**defaults)
