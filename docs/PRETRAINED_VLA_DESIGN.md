# Pretrained VLA (`U6`) Architectural & Systems Blueprint: `SmolVLA-DeltaLoop`

> **Purpose:** Zero-delay, drop-in engineering specification for scaling our controlled memory and multi-hop readout suite (`M0, M1, M2, M5, M7`) onto a pretrained Robot Foundation Model (`RESEARCH_PLAN.md` v2.3 §3.7 `U6` & §5 Gate 3A).
> **Verified On:** Miyabi HPC (`/work/gj26/b20098/hub` + `/work/gj26/b20098/projects/deltanet-looped-vla/.venv`, `lerobot 0.6.1`, `transformers 5.5.4`, `fla 0.6.0`) and Local RTX 5080 (`external/robomme_benchmark/.venv`, `torch 2.9.1+cu128`, `transformers 5.22.2`, `safetensors 0.8.0` — **zero `pip install` required locally!**).

---

## 1. Executive Diagnosis & Backbone Selection (`smolvla_base` Primary vs. `pi05_base` Secondary)

### 1.1 Why Gate 3A Points Directly to a Pretrained Action Expert Prior
- **Cycle 2A (`15/15` = `2,400` matched 8-task `val` rollouts)** established that persistent GDN state compression outperforms memoryless policies (`m5 = 11.25% [18/160]` single-pass > `m6_adaptive = 7.50% [12/160]` looped; Holm-significant `H1b vs m1: p = 0.0010, p_holm = 0.0180`; nominal unadjusted `H1a vs m0: p = 0.0234`), alongside lower `k=3` distractor interference drift in `mechanistic_probes.json` (`cos = 0.945–0.974` vs `0.220` on `FrameSamp-32`).
- **Cycle 3A (`v3_mode=True` scratch SigLIP backbone, `1,760` 8-task `val` rollouts)** drove training Flow-Matching velocity error down `20×` (`m0 fm_loss = 0.00826`, `m2 fm_loss = 0.00480`, `m5 fm_loss = 0.00412`), with single-pass `m5` reaching `12.50%` (`20/160`, Holm-significant `H1c: p = 0.0002` vs `m0` `3.12%` and exceeding looped `m6` `8.12%` / `m7` `6.25%`), yet unseen 3D manipulation remained sample-limited on `100` demos/task relative to published end-to-end 3B $\pi_{0.5}$ (`17.93%` memoryless $\to$ `44.51%` with memory on the 16-task `test` split; published scratch `RoboMME/DP` scores `5.78%` on 16-task `test` in `arXiv:2603.04639` Table 2).
- **Physical Cause:** 100 demos/task sample only 100 discrete initial 3D object poses $(x, y, \theta)$. Generalizing millimeter-precise 7-DoF Panda grasps, peg insertions, and button presses to unseen `val` poses requires a **pretrained multi-robot visuomotor prior** in both the vision-language connector and the Flow-Matching Action Expert.

### 1.2 Head-to-Head Verification of Miyabi-Staged Pretrained VLAs

| Property | **`lerobot/smolvla_base` (PRIMARY)** | **`lerobot/pi05_base` (Secondary Scale-Up)** |
| :--- | :--- | :--- |
| **Miyabi Hub Snapshot (Verified)** | `/work/gj26/b20098/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205` (`906.71 MB` `model.safetensors`) | `/work/gj26/b20098/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (`14,467.17 MB` `model.safetensors`) |
| **Tokenizer / VLM Config Snapshot** | `/work/gj26/b20098/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467` (`4.9 MB` total) | `/work/gj26/b20098/hub/models--google--paligemma-3b-pt-224` |
| **Total / Active Eval Params** | **`450.05M` total / `402.74M` active** (`47.31M` `vlm.lm_head` unused & droppable!) | `3,616.76M` total / `2,826.79M` active (`789.97M` two `lm_head`s unused) |
| **Vision Tower & Spatial Tokens** | `12L` SigLIP-512 (`32×32 = 1024` patches of `768-D`) + **`4×4` PixelShuffle Connector (`12,288 -> 960`) = `64` lossless spatial tokens (`8×8`) / view** | `27L` SigLIP-So400m (`224×224` $\implies$ `256` tokens of `2048-D` / view) |
| **Truncated VLM Backbone** | `16` layers (`0..15` of `SmolLM2`), `d_vlm = 960`, GQA `15Q / 5KV`, `head_dim = 64` (frozen) | `18` layers (`PaliGemma-2B`), `d_vlm = 2048`, MQA `8Q / 1KV`, `head_dim = 256` |
| **Pretrained Flow Action Expert** | **`98.25M` params**: `16` layers (`0..15`), `d_exp = 720`, SwiGLU `2048`, even joint-SA / odd VLM-KV CA | **`424.78M` params**: `18` layers, `d_exp = 1024`, SwiGLU `4096`, `adaRMSNorm` `[3072, 1024]` |
| **1× GH200 Training Walltime (`15k` steps)** | **`14–18 min` (cached VLM tokens) / `28–35 min` (live `/dev/shm` pixel crop)**; fits on **`interact-mig` (24 GB)** AND **`interact-g` (98 GB)**! | `~2.5–3.5 hours` on full `interact-g` (`98 GB` required) |
| **Local RTX 5080 Eval VRAM / Worker** | **`~1.85 GB / worker`** (`806 MB` active `bf16` weights + `45 MB` DeltaLoop + `~250 MB` act + `~750 MB` SAPIEN) $\implies$ **5 parallel workers (`~9.3 GB < 16.3 GB`)!** | `~8.2–9.0 GB / worker` (`5.65 GB` `bf16` weights + act + SAPIEN) $\implies$ **max 1 worker** (`5×` slower eval!) |
| **Local `.venv` Compatibility** | **100% Native** (`SmolVLMForConditionalGeneration`, `LlamaModel`, `AutoTokenizer`, `safetensors` verified in `external/robomme_benchmark/.venv`) | Requires porting `pi_gemma` custom `adaRMSNorm` Gemma classes |

**Decision:** Use **`lerobot/smolvla_base`** as the primary Cycle 3B/4 backbone so all 5 controlled architectures (`M0, M1, M2, M5, M7`) train concurrently in **<35 minutes** on Miyabi's free interactive queues and evaluate **5× parallel** on our local RTX 5080.

---

## 2. Exact Safetensor Schema & Tensor Shapes of `lerobot/smolvla_base`

Inspection of `model.safetensors` (`500` tensor keys, `450,046,248` params) confirms exact dimensions:

| Submodule Key Prefix (in `model.safetensors`) | Param Count | Exact Shapes & Dtype | Role & Trainability |
| :--- | :---: | :--- | :--- |
| `model.vlm_with_expert.vlm.model.vision_model.*` | `85.8M` | `patch_embedding`: `[768, 3, 16, 16]` (`BF16`), `pos_emb`: `[1024, 768]`, `12` ViT layers (`0..11`) | **Frozen (`eval()`, `no_grad`)** |
| `model.vlm_with_expert.vlm.model.connector.*` | `11.8M` | `modality_projection.proj.weight`: **`[960, 12288]`** (`BF16`, $4\times 4 \times 768 \to 960$) | **Frozen (`eval()`, `no_grad`)** — outputs `(B, 64, 960)` per `512×512` view |
| `model.vlm_with_expert.vlm.model.text_model.*` | `205.3M` | `embed_tokens`: `[49280, 960]`, **`16` layers (`0..15` ONLY!)**, `q_proj [960, 960]`, `k/v_proj [320, 960]`, `mlp [2560, 960]` (`BF16`) | **Frozen (`eval()`)** — builds `16`-layer post-RoPE prefix KV cache |
| `model.vlm_with_expert.vlm.lm_head.weight` | `47.3M` | `[49280, 960]` (`BF16`) | **DROP / DO NOT LOAD** (saves `94.6 MB` VRAM; never called) |
| `model.vlm_with_expert.lm_expert.layers.{0..15}.*` | `98.25M` | **Even layers (`0,2,..,14` — Joint SA):** `q [960, 720]`, `k [320, 720]`, `v [320, 720]`, `o [720, 960]` (`BF16`)<br>**Odd layers (`1,3,..,15` — VLM-KV CA):** `q [960, 720]` (`BF16`), **`k [320, 320]` (`F32`)**, **`v [320, 320]` (`F32`)**, `o [720, 960]` (`BF16`)<br>**All 16 MLPs:** `gate/up [2048, 720]`, `down [720, 2048]` (`BF16`), `norm [720]` | **Trainable (`lr = 5e-5` full `bf16` OR LoRA `r=32`)** |
| `model.state_proj.{weight,bias}` | `31.7K` | `[960, 32]`, `[960]` (`F32`) | **Trainable (`lr = 1e-4`)** — maps `32`-padded `state_norm` to 1 VLM prefix token |
| `model.action_in_proj` / `model.action_out_proj` | `46.8K` | `[720, 32]` / `[32, 720]` (`F32`) | **Trainable (`lr = 1e-4`)** — token-wise (`chunk_size = 20` works natively!) |
| `model.action_time_mlp_{in,out}` | `1.56M` | `[720, 1440]` $\to$ SiLU $\to$ `[720, 720]` (`F32`) | **Trainable (`lr = 1e-4`)** — `min_period = 4e-3`, `max_period = 4.0` |

---

## 3. Six Critical Implementation Gotchas Caught Before Coding (READ FIRST!)

1. **NEVER Call `AutoProcessor.from_pretrained(...)` — Use `AutoTokenizer` + `AutoConfig` Only:**
   - `transformers.models.smolvlm.processing_smolvlm` imports `num2words`, which is **not installed** in local `external/robomme_benchmark/.venv` (would crash local RTX 5080 eval with `ModuleNotFoundError: No module named 'num2words'`).
   - `SmolVLA` (`add_image_special_tokens=False`) **never uses `AutoProcessor` on images** (images are pure `F.interpolate(..., (512, 512))` + `img * 2.0 - 1.0`). Using `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` + `AutoConfig.from_pretrained(...)` is **verified 100% working offline** on both Miyabi and local `.venv`!
   - **One-time `<5 MB` sync to local workstation** (so local RTX 5080 eval works 100% offline):
     ```bash
     rsync -azL miyabi-g1:/work/gj26/b20098/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct \
       ~/.cache/huggingface/hub/
     ```
2. **No `config.json` in `models--lerobot--smolvla_base` Snapshot + Strip `'model.'` Prefix:**
   - Miyabi's cached `smolvla_base` snapshot contains `model.safetensors` (`906.71 MB`) + processor JSONs, **without `config.json`**. Do **not** call `SmolVLAPolicy.from_pretrained(...)` (which also pulls in `lerobot` dependencies unavailable in local eval `.venv`).
   - Instead, build a self-contained PyTorch module `src/models/smolvla_deltaloop.py` (using only `torch`, `transformers`, `safetensors`, and `src/models/deltaloop_vla.py` components) and load `safetensors.torch.load_file(ckpt_path)` directly after stripping `"model."` and skipping `"model.vlm_with_expert.vlm.lm_head."`!
3. **Use `lerobot 0.6.1`'s Immutable `dict` KV Cache (Avoid `0.6.2`'s `DynamicCache.update` Bug):**
   - In `lerobot 0.6.2`, `DynamicCache.update(...)` inside `forward_attn_layer` mutates `past_key_values` in-place across the 10 Euler steps!
   - Follow Miyabi `0.6.1`: `prefill_vlm_kv(...)` returns an **immutable** `dict[int, tuple[Tensor, Tensor]]` mapping `layer_idx -> (k_prefix, v_prefix)` of shape `(B, L_prefix, 5, 64)` (post-RoPE). During Action Expert forward (`denoise_step` or multi-depth training), even layers compute `torch.cat([k_prefix, k_suffix], dim=1)` **out-of-place**, and odd layers cross-attend directly to `(k_prefix, v_prefix)`.
4. **Preserve `SmolVLA`'s Custom Split-Half `apply_rope(max_wavelength=10_000)`:**
   - Although `SmolVLM2` `config.json` says `"rope_theta": 100000`, `smolvla_base` was pretrained using `smolvlm_with_expert.py`'s custom split-half `apply_rope(x, positions, max_wavelength=10_000)` on **both** VLM layers `0..15` and `lm_expert` layers `0..15`. Copy `apply_rope` (`max_wavelength=10_000`) verbatim—never substitute `LlamaModel.forward`!
5. **Opposite Flow-Time Convention (`t = 1.0` Pure Noise $\to$ `t = 0.0` Clean Action!):**
   - Unlike Cycle 2A/3A (`τ = 0` noise $\to$ `τ = 1` clean), **`SmolVLA` pretrains `action_time_mlp` with `t = 1.0` = pure noise $\epsilon \sim \mathcal{N}(0, I)$ and `t = 0.0` = clean action $a$**:
     $$x_t = t \cdot \epsilon + (1 - t) \cdot a, \qquad u_t = \epsilon - a, \qquad t \sim 0.999\,\mathrm{Beta}(1.5, 1.0) + 0.001.$$
   - At inference, Euler integration runs for `num_steps = 10` (`dt = -0.1`) starting at $x_{1.0} = \epsilon$, $t_k = 1.0 - 0.1 k$, updating $x_{t - 0.1} = x_t - 0.1 \, v_\theta(x_t, t)$. **Keep this exact `t = 1 -> 0` convention** so `action_time_mlp` stays 100% in-distribution from step 0!
6. **Prompt Trailing `"\n"`, Camera Order, `chunk_size = 20`, and `MEAN_STD` `(20, 8)` Normalization:**
   - **Prompt:** Always append trailing newline `task_str = task.rstrip("\n") + "\n"`, tokenize with `max_length = 48, padding = "max_length", padding_side = "right", truncation = True`.
   - **Cameras:** Pass `[front_rgb, wrist_rgb]` (`2` views, `empty_cameras = 0`), resized to `(512, 512)` bilinear and scaled to `[-1.0, +1.0]` $\implies$ `64 + 64 = 128` image tokens + `48` lang tokens + `1` state token = **`L_prefix = 177` tokens**. Note image embeddings are multiplied by $\sqrt{960}$ and language embeddings by $\sqrt{960}$ in `embed_prefix` (`modeling_smolvla.py` L585, L611)!
   - **State & Action Scaling (`MEAN_STD` + `32`-Dim Zero-Padding):** `smolvla_base` (`policy_postprocessor.json`) was pretrained with **`MEAN_STD` (`z = (x - mean) / (std + 1e-6)`)**! Normalize `(8,)` state and `(20, 8)` delta-arm + abs-gripper action chunks to zero-mean unit-variance per-$(h, d)$ (using `robomme_siglip_v1` / `MME_VLA_NORM_STATS` `(20, 8)` chunk mean/std), zero-pad dims `8..31` to `32`, and apply `U1` horizon/gripper loss weights **strictly on `losses[:, :, :8]`** (`chunk_size = 20`, `exec_horizon = 16`).

---

## 4. Controlled `SmolVLA-DeltaLoop` Architecture & Zero-Init Hook Design

### 4.1 Direct Reuse of `src/models/deltaloop_vla.py` Memory & Multi-Hop Modules
Because `SmolVLM2`'s frozen `embed_image` outputs `(N, 64, 960)` (`8×8` spatial grid, `vis_dim = 960`) per camera view, **100% of our verified memory and halting modules from `src/models/deltaloop_vla.py` reuse verbatim** (`d_mem = 384`, `N_c = 4` Coda tokens, `N_w = 8` write slots, `H = 8`, `d_k = 128`, `d_v = 128`, `d_v_half = 64` $\implies$ **`520.0 KB` (`532,480 B` incl. key occupancy) exact iso-state**):
- **Current Readout Context $\mathbf{E}_t$ (for `Prelude` & `RecurrentCore` in `M1, M5, M7`):** Lightweight projection of current `front_vlm_t` `(B, 64, 960)` + `wrist_vlm_t` `(B, 16, 960)` (`2×2` avg-pooled from `8×8`) + `U3` `SpatialKeypointBranch` (`kp_front=32, kp_wrist=16`) + `obs_horizon=2` `[s_{t-1}, s_t, 5 Δs_t]` + language tokens into `(B, N_ctx, 384)`.
- **Historical Write Stream (`M2, M5, M7`):** Exact `A3` video-demo-preserving cutoff (`max(exec_start_idx, step_idx - 16 + 1)`), `U5` adaptive demo striding (`stride = 8` when `exec_start_idx > 256`, `write_stride = 4` during execution), `PerceiverSlotResampler` (`vis_dim = 960 -> d_mem = 384`, `CausalConv1D(w=4)` along frame axis), and `U5` visual-motion-gated `SlotGDNMemoryWriter` (`v3_beta_bias = -2.5`, `motion_beta_proj` gain `3.5`) via `run_dual_backend_gdn` (`V <= 64` Hopper `fla` + IEEE `fp32` PyTorch fallback).

### 4.2 Zero-Init Action-Expert Memory Hook (`SmolVLAExpertMemoryHook`)
Inside `SmolVLMWithExpertModel`, the 16 Action Expert layers (`d_exp = 720`) alternate between joint prefix-suffix self-attention (`even` $\ell \in \{0, 2, \dots, 14\}$) and VLM-KV cross-attention (`odd` $\ell \in \{1, 3, \dots, 15\}$). We attach a **dual zero-init memory hook** to all **8 odd cross-attention layers** $\ell \in \{1, 3, 5, 7, 9, 11, 13, 15\}$ (adding only **`4.15M` parameters** total across all 8 hooks while preserving **exact bit-for-bit `M0` equivalence at Step 0**):

```python
class SmolVLAExpertMemoryHook(nn.Module):
    """Zero-init dual memory injection into a SmolVLA Action Expert layer (d_exp=720, d_mem=384).

    Guarantees exact 0.0 output delta at initialization while avoiding the `0 * random_W`
    saddle point via `zero_init_out=True` + `(1.0 + tanh(gamma_ctx))` (`#63` verified fix).
    """

    def __init__(self, d_exp: int = 720, d_mem: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        # Path 1: Zero-init Cross-Attention from action tokens (B, 20, 720) to Coda C_t (B, N_c, 384)
        self.mem_q_norm = nn.RMSNorm(d_exp)
        self.mem_kv_norm = nn.RMSNorm(d_mem)
        self.mem_kv_proj = nn.Linear(d_mem, 2 * d_exp, bias=False)
        self.mem_q_proj = nn.Linear(d_exp, d_exp, bias=False)
        self.mem_o_proj = nn.Linear(d_exp, d_exp, bias=False)
        nn.init.zeros_(self.mem_o_proj.weight)  # Exact 0.0 at step 0, full 1.0x grad on step 1!
        self.gamma_ctx = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.num_heads = num_heads
        self.head_dim = d_exp // num_heads  # 120

        # Path 2: Zero-init FiLM scale/shift + time-awareness before Expert SwiGLU MLP
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_mem + d_exp, 2 * d_exp, bias=True),
        )
        nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)

    def forward(
        self,
        u_ell: torch.Tensor,            # (B, H_act=20, 720) post-VLM-attn residual
        post_attn_norm: nn.Module,      # pretrained expert_layer.post_attention_layernorm
        mlp: nn.Module,                 # pretrained expert_layer.mlp
        t_emb: torch.Tensor,            # (B, 720) sinusoidal flow-time embedding
        mem_tokens: torch.Tensor | None,# (B, N_c, 384) Coda slots C_t^{(l)}
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mem_tokens is None:
            return u_ell + mlp(post_attn_norm(u_ell))

        dt = u_ell.dtype
        B, T_act, D = u_ell.shape
        mem_t = mem_tokens.to(dt)

        # 1. Gated Memory Cross-Attention
        q = self.mem_q_proj(self.mem_q_norm(u_ell)).view(B, T_act, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.mem_kv_proj(self.mem_kv_norm(mem_t)).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        attn_mask = mem_mask[:, None, None, :] if mem_mask is not None else None
        ca = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca = self.mem_o_proj(ca.transpose(1, 2).reshape(B, T_act, D))
        gate = (1.0 + torch.tanh(self.gamma_ctx)).to(dt)
        if mem_mask is not None:
            gate = gate * mem_mask.any(dim=-1).view(B, 1, 1).to(dt)
        u_ell = u_ell + gate * ca

        # 2. FiLM-Modulated Pretrained Expert MLP
        if mem_mask is not None:
            mf = mem_mask.to(dt).unsqueeze(-1)
            m_pool = (mem_t * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
            has_m = mem_mask.any(dim=-1, keepdim=True).to(dt)
        else:
            m_pool = mem_t.mean(dim=1)
            has_m = 1.0
        scale, shift = self.film_proj(torch.cat([m_pool, t_emb.to(dt)], dim=-1)).chunk(2, dim=-1)
        h_norm = post_attn_norm(u_ell) * (1.0 + (scale * has_m).unsqueeze(1)) + (shift * has_m).unsqueeze(1)
        return u_ell + mlp(h_norm)
```

### 4.3 Controlled 5-Model Pretrained Ablation Matrix (100% Matched Pretrained Backbone)

| Model ID | Memory State Size | Loop Depth | Exact Modules Active on Frozen `SmolVLM2` + Pretrained `98.25M` Expert | Scientific Hypothesis Tested |
| :--- | :---: | :---: | :--- | :--- |
| **`smolvla_m0_base`** | `0 B` | $L = 1$ | Pretrained `SmolVLA` (`mem_tokens = None`, + shared `U1/U4` gripper & training-only `<x,y>` heads) | Pretrained Markov VLA baseline (replaces `17.93%` $\pi_{0.5}$ baseline) |
| **`smolvla_m1_rdvla_loop4`** | `0 B` | $L = 4$ | `M0` + Memoryless `Prelude -> Weight-Tied Core (L=4) -> Coda` + 8 `ExpertMemoryHooks` | `H0a` & `H1b`: Tests whether latent looping *without* history memory can solve occlusion/counting |
| **`smolvla_m2_framesamp`** | `24 KB` (`32` slots) | $L = 1$ | `M0` + `PerceiverSlotResampler(1)` + `32`-slot demo+exec buffer + `framesamp_refine` + `Coda` + 8 hooks | `H3a` & `H4`: Pretrained `FrameSamp+Modulator` baseline (and $k=3$ distractor vulnerability) |
| **`smolvla_m5_gdn2l_l1`** | **`520.0 KB`** (`2L` iso-state) | $L = 1$ | `M0` + `PerceiverSlotResampler(8)` + **2-Layer Motion-Gated $\beta \in (0,2)$ GDN** (`d_v_half=64`) + single-pass `Readout1+2` ($L=1$) + 8 hooks | `H1c` & Sign-Flip free 2-layer Householder write expressivity without multi-hop readout |
| **`smolvla_m7_deltaloop` (Ours)** | **`520.0 KB`** (`2L` iso-state) | **$L = 1..4$ (CRN Gate)** | `M5` + **Weight-Tied Dynamic Multi-Hop Readout ($L=1..4$)** + **Stage-II `M_CRN=8` Halting Gate** | **`H1a/b` + `H5a/b`**: Full `DeltaLoop-VLA` (intra-ckpt `L=1 -> L=2 -> CRN/Conv` + adaptive compute) |

---

## 5. Fast GH200 Sequence Training Pipeline (`< 20–35 min` / Model) & Prefix KV Reuse

### 5.1 Solving the Episode-History $\times$ VLM Bottleneck via `robomme_smolvla_v1`
Running `512×512` `SmolVLM2` `embed_image` live on ~`1,600` historical episode frames per batch would take `>1.2 s/step`. Because `vlm.vision_model` + `vlm.connector` are **100% frozen (`no_grad`)**, we decouple vision extraction once:
1. **Step 0 — One-Time `SmolVLM2` Visual Token Extraction (`4.5 min` for 8 Cycle 3A tasks / `9.0 min` for all 16 tasks on 1× GH200):**
   - Read `/work/gj26/b20098/datasets/robomme_data_lerobot` (`batch_size = 256` images, `bf16` SDPA) through frozen `vlm.embed_image` (`512×512 -> (64, 960)`).
   - Store per-task `{TaskName}.pt` in `/work/gj26/b20098/datasets/robomme_smolvla_v1/`:
     * `front_vlm`: `(N_frames, 64, 960)` `bfloat16` (`8×8` PixelShuffled tokens)
     * `wrist_vlm`: `(N_frames, 64, 960)` `bfloat16` (`8×8` for VLM prefix; pool `2×2 -> (16, 960)` on-the-fly for `PerceiverSlotResampler` if desired; total **~`94 GiB` for 8 tasks** / `189 GiB` for 16 tasks, loads in `<0.5 s` via `mmap=True` on `/work/gj26` with `9.8 TiB` free!)
     * Pre-tokenized `lang_tokens`: `(100, 48)` `int64` & `lang_masks`: `(100, 48)` `bool` (with trailing `"\n"`!)
     * Copy `state`, `actions`, `is_demo`, `ep_offsets`, `exec_start_idx`, `simple_subgoals`, `grounded_subgoals` directly from `robomme_siglip_v1`.
2. **Current-Frame Spatial Augmentation (`Tier 1` Cached Sub-Grid Affine vs. `Tier 1+` Live `/dev/shm` Pixel Crop):**
   - **Default Fast Mode (`~55–75 ms/step` = `14–19 min` per `15,000`-step model on 1× GH200!):** Apply `analysis/subgoal_grounding.py`'s synchronized `apply_subgrid_affine_to_tokens` (`scale ∈ [0.90, 1.0]`, shift `±0.04`, transforming `<row_y, col_x>` `U4` subgoal targets identically!) + `feat_noise = 0.015` + `patch_dropout = 0.05` directly on `front_vlm_t (B, 64, 960)` before multiplying by $\sqrt{960}$ and feeding the frozen 16-layer `vlm.text_model`.
   - **Optional Live-Pixel Crop Mode (`~115 ms/step` = `~29 min` per model on 1× GH200):** Stage active task Parquet files into `/dev/shm/robomme_parquet` (`<15 s` copy into `480 GB` Grace RAM), apply pixel-level `RandomResizedCrop(512, scale=(0.90, 1.0))` + `ColorJitter` ONLY to the `B = 64` **current-step** `front_rgb_t` frames (`with torch.no_grad(): vlm.embed_image`), while reading all historical episode frames (`W_e`) from `robomme_smolvla_v1` in `0 ms`!

### 5.2 Single-Prefill `past_key_values` Reuse Across Multi-Hop Depths & `M_CRN=8` Halting
Crucially, `past_key_values` depends **only** on current `(front_t, wrist_t, lang, state_t)`—never on Coda $\mathbf{C}_t^{(l)}$, noise $\epsilon$, or flow time $t$! Therefore, in every training step (`B_ep = 16` episodes $\times 4$ chunks = `B = 64` chunks):
1. **Prefill VLM Prefix Once (`1` call, `L_prefix = 177`):** Run frozen `vlm.text_model` layers `0..15` (`with torch.set_grad_enabled(train_state_proj):`—note tokens `0..175` have `att_mask=0` and never attend to `state_tok` at `176`!) to obtain `past_key_values` (`~7 ms` on GH200).
2. **Packed Episode GDN Scan (`1` call):** Encode `B_ep = 16` episodes via `PerceiverSlotResampler` + `compute_packed_gdn_states` (`fla` Hopper `chunk_gated_delta_rule`) $\to$ `(S1_past, occ1_past, S2_past, occ2_past)` (`~14 ms` on GH200).
3. **Multi-Hop Readout & Shared-CRN Expert Supervision (`l = 0..3`):**
   - Run `Prelude` + `L_max = 4` weight-tied `Readout -> Core -> Coda` hops (`< 3 ms` total!) to get `[C_t^{(0)}, C_t^{(1)}, C_t^{(2)}, C_t^{(3)}]`.
   - Supervise `U4` training-only `<row_y, col_x>` + `simple_subgoal` grounding head at **every** hop $l=0..3$.
   - Sample CRN `(noise ε, time t)` and run ONLY the lightweight `98.25M` `lm_expert` (`H_act = 20` tokens, reusing `past_key_values`!) on either 2 stratified CRN depths (`l=0` + `l ~ U{1,2,3}` for `2×` faster Stage-I) or all 4 depths (`~8 ms/depth` on GH200!).
   - **Stage-II Halting (`1,000` steps, `< 3 min`):** Freeze all weights except `CRNHaltingGate`, prefill `past_key_values` **once** per batch, and evaluate `M_CRN = 8 × 4 = 32` expert-only passes (`20` tokens) in a single fused batch `B_fused = B * 8`!

### 5.3 Optimizer, Differential Learning Rates & Fine-Tuning Hyperparameters
- **Steps & Batch:** `12,000–15,000` Stage-I steps + `1,000` Stage-II CRN gate steps, `B_ep = 16` episodes $\times 4$ chunks = **`B = 64` chunks/step**, cosine decay after `500`-step warmup.
- **Differential AdamW (`betas=(0.9, 0.95)`):**
  1. `lm_expert.layers.*` (pretrained `98.25M` in `bf16`, or LoRA `r=32` on `q,k,v,o,gate,up,down`): **`lr = 5e-5`**, `weight_decay = 1e-4` (prevents catastrophic forgetting of the pretrained manipulation manifold on 100 demos!).
  2. `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_{in,out}`, `gripper_head`: **`lr = 1e-4`**, `weight_decay = 1e-4`.
  3. New `DeltaLoop` memory, loop, `SmolVLAExpertMemoryHook`s, and `U4` grounding heads: **`lr = 2e-4`**, `weight_decay = 1e-3`, with **separate gradient clipping** (`clip_write = 1.0`, `clip_other = 1.0`).

---

## 6. Local RTX 5080 Evaluation Wrapper (`StatefulPolicy`) & Secondary `pi05_base` Mapping

### 6.1 `SmolVLARoboMMEWrapper` (`< 1.9 GB` VRAM/Worker, 5× Parallel on 16 GB RTX 5080)
`SmolVLARoboMMEWrapper` implements `src/robomme_eval/interface.py:StatefulPolicy` with **zero extra vision encoder** in VRAM:
- **`reset()`:** Clears streaming GDN states `(S1, occ1, S2, occ2)`, raw `SmolVLM2` token history ring/list, `prev_state_norm`, and `exec_start_idx`.
- **`ingest_demo_frames(front_rgb_list, wrist_rgb_list, state_list)` & `ingest_step_history(...)`:** Encodes **only the strided written frames** (`write_stride = 4`, or `long_demo_stride = 8` when `N_demo > 256` — i.e., only **4 frames = 8 images per 16-step chunk**!) through `vlm.embed_image` (`bf16`, `batch <= 32`), caching `(64, 960)` tokens so A3's 16-frame exclusion window commits eligible slots incrementally into `(S1, S2)` via `recurrent_gated_delta_rule` in `< 6 ms`!
- **`predict_chunk(front_rgb, wrist_rgb, state, prompt)`:**
  1. Encodes current `(front_rgb, wrist_rgb)` (`2` images $\to$ `(1, 64, 960)` each) + cached tokenized `prompt + "\n"` + `state_proj(pad32(state_norm))` $\to$ `prefill_vlm_kv` (`1` pass, `~9 ms`).
  2. Reads committed `(S1_past, S2_past)` via `Prelude -> Readout1+2 -> Core` (`L_t` chosen online by `CRNHaltingGate.step_hazard >= 0.5`, `~1.5 ms`).
  3. Runs `10`-step Euler integration (`t = 1.0 -> 0.0`, `dt = -0.1`, reusing `past_key_values` + `C_t^{(L_t-1)}` across all 10 steps, `~28 ms`), slices `actions[:, :16, :8]`, unnormalizes via `(20, 8)` `MEAN_STD` + `state[:7]` delta-to-absolute joint conversion (`gripper_head` sign at dim 7), and returns `PolicyChunkOutput` in **`~45–60 ms/chunk` (`< 1.9 GB` VRAM/worker)**!

### 6.2 Optional Scale-Up Mapping for `lerobot/pi05_base` (`3.62B`)
If a single-flagship `3.6B` scale-up run (`M0` vs `M7` on `regular-g` / `interact-g`) is desired after `SmolVLA`:
- `lerobot/pi05_base` (`modeling_pi05.py`) already has native `adaRMSNorm` (`input_layernorm.dense` & `post_attention_layernorm.dense` of shape `[3072, 1024]`) on all `18` `gemma_expert` layers (`d_exp = 1024`).
- The **exact same** `SmolVLAExpertMemoryHook(d_exp=1024, d_mem=384, num_heads=8)` attaches to `gemma_expert.model.layers.{1,3,..,17}` (`adarms_cond[1] = time_emb + zero_init_mem_proj(m_pool)`), using `QUANTILES` normalization (`MME_VLA_NORM_STATS`) and `224×224` `resize_with_pad_torch`.
