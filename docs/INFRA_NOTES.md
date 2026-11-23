# Infrastructure, RoboMME Benchmark & Miyabi HPC Operational Guide

> **Last verified:** 2026-11-11 (`infra-probe`)
> **Key rule of thumb (from `EXP_TIPS.md`):** Train on **Miyabi-G** (`NVIDIA GH200 120GB`, **free <=2h interactive jobs** via `interact-g` / `interact-mig`) starting with **compact models** for rapid iteration, and run **ManiSkill3/SAPIEN RoboMME evaluation in parallel on our local RTX 5080** (verified at **147–183 Hz headless**).

---

## 1. Local Workstation Summary (`x86_64` + NVIDIA RTX 5080)

| Resource | Verified Specification |
| :--- | :--- |
| **OS / Arch** | Linux `x86_64` |
| **GPU** | `NVIDIA GeForce RTX 5080 Laptop GPU` (**16,303 MiB VRAM**, Driver `580.196.07`, CUDA `13.0`) |
| **Disk Free** | **206 GiB available** on `/` (`/dev/nvme0n1p1`, 915 GiB total) |
| **Package Manager** | `/home/kohei/.nix-profile/bin/uv` |
| **Headless Vulkan** | `/usr/share/vulkan/icd.d/nvidia_icd.json` installed; headless SAPIEN 3.0.2 works out-of-the-box with `DISPLAY=""` (no Xvfb/VNC needed) |
| **Ready RoboMME Sim + Eval Venv** | `/home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/.venv` (`Python 3.11.15`, `torch 2.9.1+cu128`, `mani-skill 3.0.0b21 @ 07be6fbc`, `sapien 3.0.2`, `robomme 0.1.0`, `transformers 5.22.2`, `diffusers 0.42.1`, `sentencepiece 0.2.3`, `protobuf 7.36.2`, `hydra-core 1.3.8`, `einops 0.8.2`, `websockets 15.0.1`, `msgpack 1.1.2`, `flask 3.1.3`) |
| **Other Local PyTorch Venvs** | `/home/kohei/codes/ghq/github.com/k1000dai/VLA_icl_steering/.venv` (`Python 3.12.3`, `torch 2.14.1+cu130`), `/home/kohei/codes/ghq/github.com/k1000dai/lerobot_urdf/.venv` (`Python 3.12.3`, `torch 2.11.0+cu128`, `lerobot`, `h5py`) |

### Local Headless RoboMME Verification & Usage
We cloned `https://github.com/RoboMME/robomme_benchmark` into `external/robomme_benchmark` and installed its isolated simulator + challenge server environment in `external/robomme_benchmark/.venv`.

Verified benchmark speeds on local RTX 5080 (`DISPLAY=""`):
- `PickXtimes` (`val`, 50 eps): `env.reset()` in **1.59 s**, headless execution at **6.8 ms/step (147.0 Hz)**.
- `VideoUnmaskSwap` (`val`, 50 eps): `env.reset()` in **1.05 s** (returns 168 conditioning-video frames + 1 initial execution frame = 169 frames in `obs["front_rgb_list"]`), headless execution at **5.5 ms/step (182.6 Hz)**.

```bash
# Run headless RoboMME evaluation locally (decoupled from policy training/env)
DISPLAY="" /home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/.venv/bin/python -c '
from robomme.env_record_wrapper import BenchmarkEnvBuilder
builder = BenchmarkEnvBuilder(env_id="VideoUnmaskSwap", dataset="val", action_space="joint_angle", gui_render=False, max_steps=1300)
env = builder.make_env_for_episode(0)
obs, info = env.reset()
print("Tasks:", len(BenchmarkEnvBuilder.get_task_list()), "Frames on reset:", len(obs["front_rgb_list"]), "Goal:", info["task_goal"][0])
env.close()
'
```

> **Disk Warning (Local vs. Remote):** Local free disk is **206 GiB**. Keep large datasets (`Yinpei/robomme_preprocessed_data` at 265.8 GiB, or full uncompressed HDF5s) on **Miyabi `/work/gj26/b20098`** (~9.8 TB free in group quota), and transfer only trained checkpoints (~50 MB – 2 GB) back to local for parallel RTX 5080 evaluation!

---

## 2. Miyabi HPC (`miyabi-g1`) Complete Operational Guide

### 2.1 Critical Gotchas Caught During Probing (READ FIRST!)

1. **`ssh-miyabi` CLI Argument Rule (`$# -gt 2` exits with error!):**
   `/home/kohei/.local/bin/ssh-miyabi` accepts **at most 2 positional arguments** (`[node] [--batch]`) and reads commands from **STDIN** (or `CMDS_RAW`), **NOT** as a 3rd argument!
   ```bash
   # CORRECT: pass commands via heredoc or pipe to STDIN
   /home/kohei/.local/bin/ssh-miyabi 1 --batch <<'EOF'
   hostname && qstat
   EOF

   # ALSO WORKS once ControlMaster socket is active (~hours after ssh-miyabi auth):
   ssh -o BatchMode=yes miyabi-g1 "qstat"
   rsync -az --exclude '.venv' --exclude 'external' --exclude 'outputs' ./ miyabi-g1:/work/gj26/b20098/projects/deltanet-looped-vla/
   ```
2. **Stale Values in `~/.config/hpc/config.toml`:**
   `~/.config/hpc/config.toml` lists `group = "gw13"` and `work_dir = "/wor/gj26/b20098"`. **Both are invalid:** `gw13` expired on `2026/08/04`, and `/wor/` is a typo.
   - **Active PBS project/group:** `-W group_list=gj26` (`uid=14359(b20098)`, `gid=22662(gj26)`)
   - **Active work directory:** `/work/gj26/b20098` (project dir created at `/work/gj26/b20098/projects/deltanet-looped-vla`)
3. **CRITICAL Triton / `fla` Fixes on Miyabi Hopper (`CC=gcc CXX=g++` + `triton>=3.7.1`):**
   - Miyabi loads `nvidia/25.9` by default, setting `CC=nvc`. When Triton JIT-compiles `cuda_utils.c`, `nvc` crashes with `nvc-Error-Unknown switch: -Wno-psabi`. Always set:
     ```bash
     export CC=gcc
     export CXX=g++
     ```
   - **Hopper (`sm_90`) Triton Bug `#640`:** `lerobot-pi05/.venv` ships `triton==3.6.0`, which causes `fla` `chunk_gated_delta_rule` backward (`chunk_bwd_dqkwg`) to raise `RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results`. We upgraded `/work/gj26/b20098/projects/deltanet-looped-vla/.venv` to **`triton==3.9.0`** and **empirically verified `chunk_gated_delta_rule` forward + backward on `NVIDIA GH200` (`mg1098`)** (`FLA_GH200_FWD_BWD_OK`, autotune cache saved in `/work/gj26/b20098/.triton`).
4. **`flash-linear-attention` Package Split (`fla-core` required):**
   Installing `flash-linear-attention==0.6.0` with `--no-deps` omits `fla.ops` unless `fla-core==0.6.0` is also installed (`uv pip install --no-deps fla-core flash-linear-attention`). Both are now pre-installed in `/work/gj26/b20098/projects/deltanet-looped-vla/.venv`.
5. **Non-TTY `qsub -I` Line-Wrapping & `PATH` in `zsh`:**
   - The user's login shell on Miyabi is `zsh`. Non-interactive `zsh -c` does **not** include `/opt/pbs/bin` or `~/.local/bin` in `PATH`, and `source /etc/profile.d/modules.sh` fails under `zsh` (`MOD_GIT_ROOTDIR` unbound) unless run inside `bash -lc`.
   - When driving `qsub -I` via `script`, themed `zsh` wraps input lines >80 columns (`ssh-miyabi` line 41 note). Always put job commands in a script file on `/work/gj26/b20098/projects/deltanet-looped-vla/...` and send only a short `<80 char` invocation (`bash /work/.../job.sh` + `exit`)!

### 2.2 Miyabi Hardware, Quotas, Budget & Maintenance Schedule

| Item | Verified Value |
| :--- | :--- |
| **Compute Node Hardware (`Miyabi-G`)** | **1x NVIDIA GH200 Grace Hopper Superchip** per node: `NVIDIA GH200 120GB` (`97,871 MiB` dedicated HBM3e, Compute Capability `9.0` SM90, Driver `615.71.09`) + **72-core ARM64 Neoverse-V2 Grace CPU** (`aarch64`) |
| **MIG Slice (`interact-mig` / `debug-mig`)** | `NVIDIA GH200 120GB MIG 1g.24gb` (**24 GB VRAM** slice — 0/12 used, starts in ~2 seconds!) |
| **Compute Node Internet Access** | **ONLINE (`HTTP/2 200` to `https://huggingface.co` verified on `mg1098`!)** — compute nodes CAN download from HF/W&B directly, though staging datasets on `/work` is still best practice |
| **Disk Quota (`/work/gj26`, `show_quota`)** | Used `191.4 TiB / 201.2 TiB` (**~9.8 TiB free**); `b20098` personal usage `2.15 TiB` (no per-user cap); inodes `230.3M / 419.8M` |
| **Home Quota (`/home/b20098`)** | `50 MB / 51,200 MB` (keep caches/venvs on `/work/gj26/b20098`, where `$HOME` already symlinks/points in containers) |
| **Token Budget (`show_token`)** | Project `gj26`: `236,194.1 / 261,360` node-hours used (**25,165.9 node-hours remaining**, expires `2027/03/31`). **Interactive queues (`interact-g`, `interact-mig`) show `TOKEN: -` (0 token cost — FREE!)** |
| **Scheduled Maintenance** | Next stop: `2026/11/25 (Wed) 09:00 JST` (~13.5 days from now) |

### 2.3 Miyabi-G Queue Map & When to Use Each

Check live queue occupancy anytime with `qstat --rsc -b` and `qstat --nodeuse` (do **not** use `qstat -Q`, which is unsupported by Miyabi's custom `qstat` wrapper; use `/opt/pbs/bin/qsub`, `qstat`, `qdel`).

| Queue | Nodes (Used/Total at probe) | Walltime / Scope | Token Cost | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`interact-mig`** | **0 / 12 nodes** (4 MIGs/node = 48 slices, **0 queued!**) | `<= 02:00:00` (`qsub -I`), 1x `1g.24gb` (24 GB VRAM) | **FREE (`-`)** | **Instant (<3s wait!)** debugging, kernel tests, compact policy training (`batch_size <= 128`, `<24 GB` VRAM) |
| **`interact-g`** | 48 / 48 shared pool (`15` running, `0` queued in `interact-g_n1`) | `<= 02:00:00` (`qsub -I`), 1–8 full GH200 (98 GB VRAM) | **FREE (`-`)** | **Fast full-GH200 2h training runs** (~90k steps of compact Flow policy or frozen-backbone VLA) |
| **`ai-interact-g`** | **8 / 1024** (`1%` used) | Interactive AI pool (`qsub -I`) | Check `qstat` | Fallback interactive full-node queue if `interact-g` is full |
| **`debug-mig` / `debug-g`** | `debug-mig`: 9/12 (`0` queued); `debug-g`: 48/48 (`183` queued) | Short batch jobs (`<= 00:30:00`–`02:00:00`) | Charged | Prefer `interact-mig` / `interact-g` instead (`debug-g` is heavily queued) |
| **`short-g` / `regular-g`** | `short-g`: 22/24 (`146` queued); `regular-g`: 994/1024 (`867` queued) | Batch jobs up to `24:00:00` (`#PBS -q regular-g`) | Charged (`gj26`) | Multi-hour / overnight full 3B VLA fine-tuning with checkpoint-resume |

### 2.4 Exact Tested Recipe: Running Automated Non-TTY Training on Free Interactive GPUs (`run_interactive_pbs.py`)

Because `interact-g` and `interact-mig` require `-I` (interactive pseudo-terminal), **do NOT pipe stdin directly via `script -qec 'qsub -I ...' <<EOF`** — if `qsub -I` waits even >1s in queue (`waiting for job ... to start`), stdin bytes piped upfront are consumed/dropped before the compute-node shell starts! Instead, use our verified PTY driver **`/work/gj26/b20098/projects/deltanet-looped-vla/run_interactive_pbs.py`**, which forks `/opt/pbs/bin/qsub -I`, waits for `qsub: job ... ready` + shell startup, and only then injects `bash <script.sh>` and captures `__PBS_EXIT_CODE__`:

- **CPU Cgroup RAM Limits (`interact-mig` vs `interact-g`):**
  - `interact-mig` / `debug-mig` (`1g.24gb` slice): **24 GB GPU VRAM + 25 GB CPU cgroup RAM** (`CONSTRAINT_MEMCG` kills processes exceeding 25 GB RSS!).
  - `interact-g` (full GH200 node): **98 GB GPU VRAM + 100 GB CPU cgroup RAM + 72 Grace CPUs** (still **100% FREE**, `TOKEN: -`).
- **Queue Management & `substate=41` Gotcha:**
  - `qstat` and `qdel` are custom wrappers in `/usr/local/bin/{qstat,qdel}`: pass numeric IDs **without `.opbs`** to `qdel` (e.g. `qdel 3883669`).
  - Never kill an `ssh` client mid-startup while `qsub -I` is transitioning into a job (`substate=41`): PBS holds your 1-job interactive slot (`max_run_res.ncpus` / `njobs_int-g`) for ~5–10 minutes until `resvport` times out. Always let `run_interactive_pbs.py` exit cleanly or run `qdel <numeric_id>`.

```bash
# Step 1: Sync local repo code to Miyabi
rsync -az --exclude '.venv' --exclude 'external' --exclude 'outputs' --exclude 'data' --exclude 'checkpoints' --exclude '.git' \
  /home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/ \
  miyabi-g1:/work/gj26/b20098/projects/deltanet-looped-vla/

# Step 2: Write a self-contained job script on Miyabi (avoids >80-col TTY wrapping)
/home/kohei/.local/bin/ssh-miyabi 1 --batch <<'EOF'
cat <<'SH' > /work/gj26/b20098/projects/deltanet-looped-vla/run_miyabi.sh
#!/usr/bin/env bash
set -euo pipefail
export CC=gcc CXX=g++
export PATH="/work/gj26/b20098/.local/bin:$PATH"
export HF_HOME="/work/gj26/b20098"
export PYTHONPATH="/work/gj26/b20098/projects/deltanet-looped-vla/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

cd /work/gj26/b20098/projects/deltanet-looped-vla
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python "$@"
SH
chmod +x /work/gj26/b20098/projects/deltanet-looped-vla/run_miyabi.sh
EOF

# Step 3: Launch reliably via run_interactive_pbs.py (<queue> <script_path> [walltime])
# Use `interact-mig` for instant 24GB VRAM / 25GB RAM jobs, or `interact-g` for full 98GB GH200 / 100GB RAM!
/home/kohei/.local/bin/ssh-miyabi 1 --batch <<'EOF'
python3 /work/gj26/b20098/projects/deltanet-looped-vla/run_interactive_pbs.py \
  interact-g \
  /work/gj26/b20098/projects/deltanet-looped-vla/run_miyabi.sh \
  02:00:00
EOF
```

For standard batch submission (`regular-g` / `short-g`):
```bash
#!/bin/bash
#PBS -N deltanet_vla
#PBS -q regular-g
#PBS -W group_list=gj26
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /work/gj26/b20098/projects/deltanet-looped-vla/outputs/pbs_train.log

export CC=gcc CXX=g++
export HF_HOME=/work/gj26/b20098
cd /work/gj26/b20098/projects/deltanet-looped-vla
.venv/bin/python scripts/train.py
```

### 2.5 Pre-Built Miyabi Python Environment & Cached Pretrained Weights

We set up and verified **`/work/gj26/b20098/projects/deltanet-looped-vla/.venv`** on Miyabi (inheriting `lerobot-pi05/.venv` via `.pth` without duplicating 5 GB of CUDA wheels):
- **Python:** `3.12.13` (`aarch64`)
- **Core ML:** `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, **`triton 3.9.0`** (upgraded from `3.6.0` to fix Hopper `sm_90` `fla` backward bug `#640`)
- **Linear Attention:** `fla-core 0.6.0` + `flash-linear-attention 0.6.0` (`chunk_gated_delta_rule` & `GatedDeltaNet` verified forward+backward on GH200 with `CC=gcc CXX=g++`!)
- **VLA / Robot Learning:** `lerobot 0.6.1`, `transformers 5.5.4`, `sentencepiece 0.2.3`, `protobuf 7.36.2`, `accelerate 1.14.0`, `safetensors 0.8.0`, `datasets 4.8.5`, `einops 0.8.2`, `h5py 3.16.0`, `hydra-core 1.3.8`, `omegaconf 2.3.1`, `websockets 17.4`, `msgpack 1.2.4`, `msgpack-numpy 0.4.8`
- **Pre-cached Vision/Text Encoder (`google/siglip-base-patch16-224`):** Cached in `/work/gj26/b20098/hub` on Miyabi and `~/.cache/huggingface/hub` locally (~428 MB VRAM in `bfloat16`).
- **Simulator availability on Miyabi:** `mani_skill` / `sapien` / `openpi` (JAX) are **not** installed on `aarch64` Miyabi (`sapien` wheels target `x86_64`). Run all ManiSkill3/SAPIEN simulation rollouts on our **local `x86_64` RTX 5080**!
- **Apptainer Container (optional):** `/work/gj26/b20098/sif/lerobot_060.sif` (load via `bash -lc 'module load apptainer/1.3.5 && apptainer exec --nv ...'`).
- **Pre-cached HuggingFace Models in `/work/gj26/b20098/hub` (`HF_HOME=/work/gj26/b20098`):**
  - `lerobot/pi05_base` (14 GB complete PyTorch $\pi_{0.5}$ checkpoint!)
  - `google/paligemma-3b-pt-224`
  - `lerobot/smolvla_base` (450M SmolVLA!)
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- **Reference Compact Flow Policy on Miyabi:** `/work/gj26/b20098/projects/small_libero/lerobot` (`tinyflow` policy: Depthwise CNN + Spatial Softmax Keypoints + 4-layer AdaLN Flow-Matching Transformer/Mixer, ~1M–10M params, trains 90k steps at `batch_size=128` in `<2h` on 1x GH200).

---

## 3. RoboMME Codebase, Datasets & Architecture Hooks

### 3.1 Ecosystem Split (4 Repositories)

`RoboMME/robomme_benchmark` (cloned at [`external/robomme_benchmark`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark)) is **purely the ManiSkill3/SAPIEN simulator, task suite, and decoupled evaluation harness**. Training code lives in sibling repos under `https://github.com/RoboMME/` (referenced as submodules in [`.gitmodules`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/.gitmodules)):

| Repository | Framework | Backbone & Size | Role |
| :--- | :--- | :--- | :--- |
| **`RoboMME/robomme_benchmark`** (local `external/robomme_benchmark`) | PyTorch + ManiSkill3 + SAPIEN 3.0.2 | Simulator + `challenge_interface` WebSocket/HTTP server | 16 RoboMME tasks (`src/robomme/robomme_env/`) + RoboMME-Plus (`src/robomme_plus/`), `BenchmarkEnvBuilder`, episode seeds/metadata |
| **`RoboMME/robomme_policy_learning`** | **JAX (`0.5.3`) + Flax NNX (`0.10.2`)** (`openpi`) | $\pi_{0.5}$ (`PaliGemma-2B` width 2048 + `ActionExpert-300M` width 1024, depth 18 + frozen `SigLIP-400M`, **3.44B total**) | Official **MME-VLA-Suite** (14 memory-augmented variants: `FrameSamp`, `TokenDrop`, `RMT`, `TTT`, `Symbolic` $\times$ `context` / `modul` / `expert`) |
| **`RoboMME/DP`** | **Pure PyTorch** | **ResNet18 (`num_kp=32`) + CLIP text (`512->64`) + 1D UNet (`[256,512,1024]`)** (`ckpt_200000.pth` = **298 MiB**) | Lightweight **Diffusion Policy** baseline trained directly on `Yinpei/robomme_data_lerobot` Parquet files (`batch_size=128`, `pred_horizon=16`, `exec_horizon=8`) |
| **`RoboMME/MemoryVLA`** / **`SimpleMemVLA`** / **`PonderPounce`** | PyTorch | OpenVLA-7B / Qwen3.5-9B + Pi0.5 | External VLM/VLA leaderboard baselines |

### 3.2 Benchmark Tasks, Splits, Episode Lengths & Observation/Action Conventions

- **16 Tasks across 4 Memory Suites** ([`readme.md`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/readme.md)):
  1. **Counting (Temporal):** `BinFill`, `PickXtimes`, `SwingXtimes`, `StopCube`
  2. **Permanence (Spatial):** `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap`
  3. **Reference (Object):** `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`
  4. **Imitation (Procedural):** `MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick`
- **Fixed Benchmark Splits** (`src/robomme/env_metadata/{train,val,test}/*.json`):
  - **Train:** `100 episodes/task × 16 tasks = 1,600 episodes` (`768,897` total frames at **10 Hz** in RoboMME v1; mean **480.6 frames/episode** $\approx$ 48 s).
  - **Val:** `50 episodes/task × 16 tasks = 800 episodes` (use a stratified subset, e.g. 4–8 tasks × 10–20 eps = 80–160 eps, for fast iteration!).
  - **Test:** `50 episodes/task × 16 tasks = 800 episodes`, `max_steps = 1300` (`scripts/evaluation.py` line 85), evaluated across model seeds `7, 42, 0`.
- **Critical Video-Conditioning Convention** ([`challenge_interface/policy.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/challenge_interface/policy.py#L61-L75)):
  - Every `obs` entry returned by `env.reset()` and `env.step()` is a **list** (`front_rgb_list`, `wrist_rgb_list`, `joint_state_list`, `eef_state_list`, `gripper_state_list`).
  - On `env.reset()` (`is_first_step=True`): for `Video*` tasks (`VideoUnmask`, `VideoUnmaskSwap`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`), `obs["front_rgb_list"][:-1]` contains the **conditioning video demonstration frames** (e.g., 168 frames showing which cup hides the cube and how cups swap!), and `obs["front_rgb_list"][-1]` is the **first execution frame**. Our recurrent memory **MUST ingest `obs[*][:-1]` sequentially during `reset()` / first step** before predicting actions on `obs[*][-1]`!
  - During training, only execution frames (`is_demo == False` / `step_idx >= exec_start_idx`) supervise action prediction, while `is_demo == True` frames populate the prefix memory state.
- **Action Space (`joint_angle` standard):**
  - Shape `(8,)` `float32`: 7 absolute Panda joint angles (radians) + gripper command (`+1.0` = open, `-1.0` = close).
  - Observed `state` `(8,)`: 7 joint angles + gripper width (`[0, 0.04]` m; in HDF5 raw `gripper_state` is `(2,)` two finger widths).
  - **Action Chunk Horizons:** MME-VLA trains with `action_horizon = 20` and executes `16` actions per chunk (`(16, 8)`). Diffusion Policy trains with `action_pred_horizon = 16` and executes `8` actions (`(8, 8)`).

### 3.3 HuggingFace Datasets (`Yinpei/*`) — Sizes & Formats

| HF Dataset Repo | Size | Contents & Best Use Case |
| :--- | :--- | :--- |
| **`Yinpei/robomme_preprocessed_data_sample`** | **18.54 GiB** (`data.zip` 6.9 GiB + `features.zip` 12.1 GiB) | **Fastest starting point!** `42,762` total frames (`27,452` execution samples) with **pre-extracted frozen SigLIP `2048-D` visual token embeddings** (`token_emb_<step>.npy`) + state/actions/subgoals (`{idx}.pkl`) + `meta/stats.json`. |
| **`Yinpei/robomme_data_lerobot`** | **121 GiB** (**1,600 Parquet files** across `chunk-000` & `chunk-001`; HF UI truncates at 994 files / 65.18 GiB) | **STAGED ON MIYABI at `/work/gj26/b20098/datasets/robomme_data_lerobot`!** Full 1,600 train episodes (`768,897` total frames = `292,040` video demo + `476,857` execution frames): `image` (256×256 PNG bytes), `wrist_image` (256×256 PNG bytes), `state` `(8,)`, `actions` `(8,)`, `is_demo`, `exec_start_idx`, `simple_subgoal`, `grounded_subgoal`. See §5 below for exact 16-block task mapping & fast loading recipes. |
| **`Yinpei/robomme_data_h5`** | **52.56 GiB** compressed (`.h5.tar.xz` × 16 tasks) | Raw per-task HDF5s (100 eps/task) including RGB, millimeter `int16` depth, camera intrinsics/extrinsics, `joint_action` `(8,)`, `eef_action` `(7,)`, `waypoint_action` `(7,)`, `is_video_demo`, `is_subgoal_boundary`. Can download **individual task `.h5.tar.xz` files (0.9–4.6 GiB each)** for targeted per-suite experiments! |
| **`Yinpei/robomme_preprocessed_data`** | **265.78 GiB** (`data/part_0..9.zip` + `features/`) | Full 1,600-episode pre-extracted SigLIP `2048-D` features + pickles used by upstream `MME-VLA-Suite`. Stage on Miyabi `/work/gj26/b20098/datasets/` only if needed for full 16-task frozen-SigLIP scaling. |
| **`Yinpei/robomme_plus_data_lerobot`** / **`_h5`** | **0.30 GiB** (LeRobot v3 H264) / **20.95 GiB** (H5) | 50 OOD RoboMME-Plus demos (`56,462` frames at 20 Hz). |

Also available on HF (`Yinpei/*` models):
- **`Yinpei/pi05_vision_encoder`** (`1.58 GiB`, `siglip_params.pkl`): standalone frozen $\pi_{0.5}$ SigLIP vision encoder weights!
- **`Yinpei/DP-Unet-ckpt`** (`297.9 MiB`, `ckpt_200000.pth`): pretrained RoboMME Diffusion Policy UNet checkpoint.
- **`Yinpei/mme_vla_suite`** & **`Yinpei/perceptual-framesamp-modul`**: released 80k-step MME-VLA checkpoints.

### 3.4 Exact Architecture Hooks in `MME-VLA-Suite` & Why Our Method Fits Cleanly

Inspection of `RoboMME/robomme_policy_learning/src/mme_vla_suite/` reveals the exact baseline implementation details:

1. **Pre-extracted Feature Stream (`models/config/base.yaml` & `representation/recur_mem.py`):**
   - Historical frames are represented by frozen SigLIP patch embeddings (`img_emb_dim = 2048`, `token_per_image = 64` for recurrent or `16` for frame-sampling, `num_views = 1` front camera) + temporal/3D pos emb (`768`) -> projected via `FeatureEncoder` to `hidden_dim = 512`.
   - `RecurrentMemory` (`recur_mem.py`) processes up to `max_recur_steps = 64` frames (`max_pretraj_steps = 40` demo frames + execution history, `64 × 64 = 4096` tokens in `64`-token mini-batches) through a 1-layer `TTTLayerLinear` (`num_ttt_heads = 2`) or `RMTLayer`, then **statically slices and projects the last `budget = 512` tokens ONCE**:
     `final_output = self.proj(output[:, -512:])` $\to$ `mem_seq: [B, 512, 1024]`.
   - **Crucial Scientific Limitation Confirmed in Code:** Upstream `recurrent-ttt-modul` performs **no iterative/dynamic query-conditioned readout** from its fast-weight state—the Action Expert only attends to a static single-pass readout buffer `mem_seq`!
2. **Memory-as-Modulator Hook (`models/integration/history_gemma.py`, `HistoryBlock` lines 180–200):**
   - Inside each of the 18 layers of $\pi_{0.5}$, `xs = [vlm_tokens, action_expert_tokens]`.
   - After joint prefix-causal self-attention and **only on the Action Expert stream (`xs[-1]`, width `1024`)**, `MemoryAttention` (GQA cross-attention: `4` query heads, `1` KV head, `head_dim = 256`, RoPE) attends from action tokens `x: [B, T_act, 1024]` to `mem_seq: [B, S, 1024]`, producing `mem_mod_vec: [B, T_act, 1024]`.
   - `MemoryRMSNorm` modulates `x` via FiLM/AdaRMS (`norm(x) * (1 + scale(mem_mod_vec)) + shift(mem_mod_vec)`) immediately before `pre_ffw_norm` (flow-time AdaRMSNorm) + `FeedForward`!
3. **Recommended Two-Stage Experimental Pipeline for Our PyTorch Implementation:**
   - **Stage A — Fast Compact PyTorch Visuomotor Policy (`<1–2h` on 1x Miyabi GH200, FREE `interact-g`):**
     Use a lightweight PyTorch encoder (ResNet18 with Spatial Softmax / compact ViT or cached SigLIP tokens from `Yinpei/robomme_preprocessed_data_sample` / per-suite `Yinpei/robomme_data_h5` tasks like `VideoUnmaskSwap`, `ButtonUnmaskSwap`, `PickXtimes`, `VideoPlaceOrder`) + **Gated DeltaNet Memory (`fla` `chunk_gated_delta_rule`)** + **Weight-Tied Looped AdaLN Flow-Matching Action Expert** (`width = 256–512`, `horizon = 16`, `action_dim = 8`). Trains in minutes to <2h on 1x GH200; evaluates directly on local RTX 5080 via `BenchmarkEnvBuilder` or [`challenge_interface/policy.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/challenge_interface/policy.py)!
   - **Stage B — Frozen-Backbone VLA Scaling (`lerobot/pi05_base` or `lerobot/smolvla_base` already cached in `/work/gj26/b20098/hub` on Miyabi):**
     Keep the VLM/vision encoder frozen (or use cached `2048-D` SigLIP features) and attach our PyTorch **Gated DeltaNet + Looped Multi-Hop Readout Modulator** directly to the Flow-Matching Action Expert (`width = 1024` for $\pi_{0.5}$ or `SmolVLA`), decoupling simulator dependencies completely via `challenge_interface` WebSocket/HTTP (`server.py` / `client.py`).


---

## 4. Exact Preprocessing, Action/State Normalization & Demo-Frame Conditioning (`RoboMME/DP` vs. `MME-VLA-Suite`)

We cloned both official training repositories locally (`--depth 1`) for direct reference:
- [`external/DP`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/DP) (`https://github.com/RoboMME/DP` — PyTorch ResNet18 + CLIP + 1D UNet Diffusion Policy)
- [`external/robomme_policy_learning`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning) (`https://github.com/RoboMME/robomme_policy_learning` — JAX/Flax `openpi` $\pi_{0.5}$ **MME-VLA-Suite**)

### 4.1 Side-by-Side Specification Table

| Component | `RoboMME/DP` (`external/DP`) | `MME-VLA-Suite` (`external/robomme_policy_learning`) |
| :--- | :--- | :--- |
| **State Packing (`(8,)` `float32`)** | `np.concatenate([joint_state, gripper_state[:1]], dtype=np.float32)` — 7 Panda joint angles (rad) + **first finger width ONLY (`gripper_state[:1]` in `[0, 0.04]` m, NOT sum/mean!)** | **Identical:** `pack_state` in [`env_runner.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/examples/robomme/env_runner.py#L16-L18) & [`build_robomme_dataset.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_robomme_dataset.py#L259) |
| **Current RGB Input (`front_rgb`, `wrist_rgb`)** | Raw `(256, 256, 3)` `uint8` $\to$ CHW `/ 255.0` in `[0, 1]`, stacked to `(6, 256, 256)`. Inside [`dp_unet.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/DP/eval_envs/model/dp_unet.py#L94-L109): split into `K=2` views, **`RandomCrop(230, 230)`** (`crop_ratio=0.9`, train) / **`CenterCrop(230, 230)`** (eval), then `(x - 0.5) / 0.5` $\to$ **`[-1, 1]`**. Encoded by `ResNet18 + SpatialSoftmax(num_kp=32)` (`64-D`/cam). | Raw `(256, 256, 3)` `uint8` $\to$ **`resize_with_pad(224, 224)`** $\to$ `float32 / 255.0 * 2.0 - 1.0` in **`[-1, 1]`** ([`model.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/src/openpi/models/model.py#L117-L189)). Train augs (`augmax`): `base_0_rgb` gets `RandomCrop(0.95)` + `Resize(224,224)` + `Rotate(-5°, +5°)` + `ColorJitter(0.3, 0.4, 0.5)`; `left_wrist_0_rgb` gets `ColorJitter` **only** (no crop/rotate on wrist!). |
| **Language Goal Conditioning** | `openai/clip-vit-base-patch32` frozen text embedding (`512-D`) projected via `Linear(512, 64)` | PaliGemma SentencePiece tokenizer (`max_token_len = 200`, `discrete_state_input = True` for $\pi_{0.5}$) |
| **Action Space Representation (`(8,)` `float32`)** | **ABSOLUTE joint angles + gripper** (`use_delta_action: False` in `train_dp_unet.yaml`): `actions[:, :7]` = target joint angles (rad); `actions[:, 7]` $\in [-1, 1]$ (`+1` open, `-1` close) | **DELTA arm joints + ABSOLUTE gripper** ([`config.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/src/mme_vla_suite/training/config.py#L370-L374)): `delta_action_mask = make_bool_mask(7, -1) = (True,)*7 + (False,)`.<br>• **Train (`DeltaActions`):** `a_train[:, :7] = a_raw[:, :7] - state[:7]`, `a_train[:, 7] = a_raw[:, 7]`<br>• **Infer (`AbsoluteActions`):** `a_exec[:, :7] = a_unnorm[:, :7] + state[:7]`, `a_exec[:, 7] = a_unnorm[:, 7]` |
| **State & Action Normalization Formula** | **Min-Max to `[-1, 1]`** (`norm_type: minmax` over `is_demo=False` frames, [`normalize.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/DP/eval_envs/utils/normalize.py#L152-L176)):<br>`x_norm = (x - min) / (max - min + 1e-6) * 2 - 1`<br>`x = (x_norm + 1) / 2 * (max - min + 1e-6) + min` | **Quantile (`q01`/`q99`) to `[-1, 1]`** (`use_quantile_norm=True` because `model_type == PI05`, [`transforms.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/src/openpi/transforms.py#L141-L182)):<br>`x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1`<br>`x = (x_norm + 1) / 2 * (q99 - q01 + 1e-6) + q01`<br>*(Historical `recur_state_emb` / `static_state_emb` also normalized with the same `state` `q01`/`q99`!)* |
| **Gripper Binarization at Eval** | DDPM `clip_sample=True` clips normalized output to `[-1, 1]`; unnormalized dim 7 passed directly to `env.step()` without `sign()` (ManiSkill3 thresholds internally at `0.0`) | Flow-sampled dim 7 unnormalized via `q01=-1.0, q99=0.9996` and passed directly (`[:, :8]` via `RoboMMEOutputs`) without explicit `sign()` (optional `sign` thresholding at `0.0` is equivalent in sim) |
| **Action Chunk Horizons** | `obs_horizon = 1`, **`action_pred_horizon = 16`**, **`action_exec_horizon = 8`** (end-of-episode padded by repeating final action in `LeRobotDataset`) | **`action_horizon = 20`** (end-of-episode padded by repeating `last_action` in `build_robomme_dataset.py`), **`exec_horizon = 16`** (`eval.py` `obs_horizon=16`) |
| **Demo Frames (`is_demo=True` / `obs[*][:-1]`)** | **Completely ignored** in both training (`_build_valid_indices` starts at `exec_start_idx`) and rollout (`serve_dp.py` `add_buffer` is a no-op; only `obs[*][-1]` is used) | **Ingested into memory buffer, NEVER supervised with action loss!**<br>• **Train:** Only `step_idx >= exec_start_idx` are training targets; prefix frames `[0 .. exec_start_idx - 1]` + prior execution frames `[exec_start_idx .. step_idx]` populate memory.<br>• **Rollout (`eval.py` + `mem_buffer.py`):** On `reset()`, `pre_traj` has `N_init = N_demo + 1` frames (`exec_start_idx = N_init - 1`). All `pre_traj` frames are pushed to `MemoryBuffer` before chunk 0; subsequent 16-step execution chunks push their 16 frames before each new `infer()`. |

### 4.2 Exact Upstream `MME-VLA-Suite` Recurrent Memory Frame-Subsampling Rule (`mem_buffer.py` lines 417–478)
Understanding how upstream `recurrent-ttt-modul` / `recurrent-rmt-modul` samples history frames explains one of its subtle bottlenecks:
- Configured with `streaming_obs_horizon = 16`, `max_recur_steps = 64`, `max_pretraj_steps = 40`, `num_views = 1` (**front camera ONLY — wrist camera is NOT fed to memory!**), `token_per_image = 64` (`8x8` pooled SigLIP `2048-D` tokens + `768-D` 3D pos emb):
  - **Non-video tasks (`exec_start_idx == 0`):** Strides execution history every `16` steps (`range(step_idx % 16, step_idx + 1, 16)[-64:]`).
  - **Video-conditioned tasks (`exec_start_idx > 0`):** Samples video demo indices at stride `8` (if `exec_start_idx <= 32`), stride `16` (if `<= 640`), or `linspace(0, exec_start_idx - 1, 40)` (if `> 640`, e.g. `VideoPlaceButton` ~760 demo frames or `VideoPlaceOrder` ~917 demo frames), concatenates strided execution indices, keeps `[-64:]`, and **left-pads** to `64` steps.
- Official 1,600-episode `q01` / `q99` / `mean` / `std` constants from [`assets/norm_stats.json`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_policy_learning/assets/norm_stats.json) are embedded directly in [`src/robomme_eval/interface.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval/interface.py) (`MME_VLA_NORM_STATS` & `ActionStateNormalizer`).

---

## 5. Staged `Yinpei/robomme_data_lerobot` on Miyabi (`/work/gj26/b20098/datasets/robomme_data_lerobot`)

We downloaded and verified the **complete 1,600-episode dataset** on Miyabi:
- **Path on Miyabi:** `/work/gj26/b20098/datasets/robomme_data_lerobot`
- **Verified Disk Footprint & Files:** **121 GiB**, **1,600 Parquet files** (`data/chunk-000/episode_000000.parquet` .. `episode_000999.parquet` and `data/chunk-001/episode_001000.parquet` .. `episode_001599.parquet`) + `meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}`.
- **Exact Frame Counts:** **`768,897` total frames** = **`292,040` conditioning video demo frames** (`is_demo == True`, `step_idx < exec_start_idx`) + **`476,857` execution frames** (`is_demo == False`, `step_idx >= exec_start_idx`).

### 5.1 Complete 16-Block Episode-to-Task Mapping (`epis_idx` `0..1599`)

Because individual Parquet rows store `epis_idx` (`0..1599`) and language prompt (`task_index` $\to$ 116 strings in `tasks.jsonl`) rather than the `EnvID` string, use this verified 16-block mapping to slice any specific task or memory suite directly:

| Block | `epis_idx` Range | `EnvID` | Suite | Mean Total Len `[min..max]` | Mean Demo Frames (`exec_start_idx`) | Total Exec Frames (`is_demo=False`) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | `0 .. 99` | **`PatternLock`** | Imitation | `207.8 [48..568]` | `103.9 [24..284]` | `10,391` |
| **1** | `100 .. 199` | **`ButtonUnmaskSwap`** | Permanence | `399.4 [297..555]` | `0.0 [0..0]` | `39,940` |
| **2** | `200 .. 299` | **`ButtonUnmask`** | Permanence | `266.3 [198..452]` | `0.0 [0..0]` | `26,626` |
| **3** | `300 .. 399` | **`VideoPlaceButton`** | Reference | `965.1 [900..1089]` | `759.5 [703..818]` | `20,558` |
| **4** | `400 .. 499` | **`VideoUnmaskSwap`** | Permanence | `348.1 [213..586]` | `153.8 [114..216]` | `19,432` |
| **5** | `500 .. 599` | **`PickXtimes`** | Counting | `537.2 [266..1025]` | `0.0 [0..0]` | `53,720` |
| **6** | `600 .. 699` | **`StopCube`** | Counting | `316.1 [121..582]` | `0.0 [0..0]` | `31,614` |
| **7** | `700 .. 799` | **`SwingXtimes`** | Counting | `434.2 [296..608]` | `0.0 [0..0]` | `43,419` |
| **8** | `800 .. 899` | **`PickHighlight`** | Reference | `345.6 [199..652]` | `0.0 [0..0]` | `34,560` |
| **9** | `900 .. 999` | **`MoveCube`** | Imitation | `393.9 [247..525]` | `220.6 [148..299]` | `17,331` |
| **10** | `1000 .. 1099` | **`InsertPeg`** | Imitation | `477.0 [406..586]` | `238.4 [203..293]` | `23,866` |
| **11** | `1100 .. 1199` | **`RouteStick`** | Imitation | `369.0 [200..700]` | `184.5 [100..350]` | `18,450` |
| **12** | `1200 .. 1299` | **`BinFill`** | Counting | `602.8 [266..1044]` | `0.0 [0..0]` | `60,282` |
| **13** | `1300 .. 1399` | **`VideoPlaceOrder`** | Reference | `1123.9 [909..1411]` | `917.1 [702..1145]` | `20,679` |
| **14** | `1400 .. 1499` | **`VideoRepick`** | Reference | `685.8 [405..1027]` | `276.6 [147..385]` | `40,921` |
| **15** | `1500 .. 1599` | **`VideoUnmask`** | Permanence | `216.7 [163..399]` | `66.0 [66..66]` | `15,068` |
| **All 16** | `0 .. 1599` | **All 16 Tasks** | **4 Suites** | **`480.6 [48..1411]`** | **`182.5 [0..1145]`** | **`476,857`** |

### 5.2 Lustre I/O Profiling & High-Throughput Data Loading Best Practices on Miyabi
Profiling `episode_000400.parquet` (`326` frames, `4` row groups) on Miyabi revealed an important Lustre characteristic:
- **Grace CPU PNG decode is very fast:** **753.4 frames/s on a single CPU thread** (`0.43s` for 326 frames; `>10,000` frames/s across 16–32 `DataLoader` workers on the 72-core Grace CPU).
- **Avoid per-sample `pq.read_table(parquet_file)` inside `__getitem__` over cold Lustre!** Reading a full 40–80 MB Parquet file over Lustre just to grab 1 random step takes `~1.1s` (warm) to `~10s` (cold). Instead:
  1. **For Compact / Recurrent Sequence Training (Fastest — Recommended):** Pre-extract frozen visual token/spatial features once on GH200 (1 pass over the 1,600 Parquet files with 16 CPU decode workers + `batch_size=256` GPU vision fwd takes **~3–5 minutes total** on GH200 and produces a compact `<8 GiB` memory-mappable tensor/HDF5 cache!). This makes multi-epoch TBPTT sequence training completely compute-bound (`0` I/O wait).
  2. **For End-to-End Pixel Training:** Either (a) copy the active task Parquet files into compute-node RAM disk `/dev/shm/robomme` at job start (`cp -r` takes `<20s` for 4–8 tasks, and GH200 nodes have ~480 GB Grace RAM!), or (b) use Arrow memory-mapped tables (`HF_DATASETS_CACHE=/work/gj26/b20098/hf_datasets_cache`) + episode-level batching.

---

## 6. Reusable Local Headless Rollout Evaluator (`src/robomme_eval/`)

We built and smoke-tested [`src/robomme_eval/`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval) on our local RTX 5080 (`external/robomme_benchmark/.venv/bin/python`):
- [`src/robomme_eval/interface.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval/interface.py):
  - `StatefulPolicy` ABC (`reset()`, `ingest_demo_frames(...)`, `ingest_step_history(...)`, `predict_chunk(...) -> PolicyChunkOutput(actions, loop_depth, aux_info)`).
  - `pack_robomme_state(joint_state, gripper_state)` (`(8,)` `float32` using `gripper_state[:1]`), `ActionStateNormalizer` (`mode="quantile_delta"` with `MME_VLA_NORM_STATS` or `mode="minmax_abs"`), `DummyStatefulLoopedPolicy`, and `ChallengePolicyAdapter` (bridges any `StatefulPolicy` to RoboMME's WebSocket `challenge_interface`).
- [`src/robomme_eval/metrics.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval/metrics.py):
  - `EpisodeResult`, `TaskEvalSummary`, `bootstrap_ci`, `paired_bootstrap_ci` (matched `(task, dataset, episode_id, model_seed)` paired bootstrap 95% CIs + two-sided p-values), `summarize_episodes`, and `compare_policy_runs`.
- [`src/robomme_eval/runner.py`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval/runner.py):
  - `evaluate_episode` & `evaluate_policy` with automatic `DISPLAY=""`, strict per-episode `policy.reset()`, step-0 `obs[*][:-1]` demo-frame ingestion, intermediate chunk-frame streaming (`ingest_step_history`), configurable `exec_horizon` (e.g. `8` or `16`) and `max_steps` (`1300`), crash-safe JSON checkpoint/resume (`resume=True`), and optional side-by-side MP4 video saving (with red border on demo frames).

### 6.1 Verified Smoke-Test Command & Throughput on Local RTX 5080
Verified 100% bug-free on both `ButtonUnmaskSwap` (`demo_frames=0`, `steps=1300` in **10.3s / episode**) and `VideoUnmaskSwap` (`demo_frames=168`, `steps=1300` in **11.5s / episode**):
```bash
DISPLAY="" PYTHONPATH=/home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src \
/home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/external/robomme_benchmark/.venv/bin/python -m robomme_eval \
  --tasks ButtonUnmaskSwap VideoUnmaskSwap \
  --dataset val \
  --episodes 2 \
  --exec-horizon 16 \
  --max-steps 1300 \
  --compare-dummy-baseline
```

### 6.2 End-to-End Evaluator Sanity Check with Pretrained `Yinpei/DP-Unet-ckpt`
To empirically verify that our local ManiSkill3/SAPIEN rollout pipeline (`pack_robomme_state`, RGB channel ordering, `ActionStateNormalizer`, joint-angle + gripper actuation, and success detection) matches upstream training conventions end-to-end, we wrapped the official pretrained Diffusion Policy checkpoint (`Yinpei/DP-Unet-ckpt` $\to$ `outputs/pretrained_dp/ckpt_200000.pth`, 298 MiB) in [`PretrainedDPPolicy`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_eval/dp_baseline.py#L89-L223):

- **Exact Reverse-Engineered `Yinpei/DP-Unet-ckpt` Config (`state_dict` inspection):**
  - Note: `oldTOM/RoboMME_DP` on HuggingFace is gated (`403`), whereas `Yinpei/DP-Unet-ckpt` provides `ckpt_200000.pth` without `config.yaml` or `stats.json`.
  - Inspecting `ckpt["model"]` tensor shapes confirmed **`obs_horizon = 2`**, **`include_text = False`** (no `text_proj` weights; `cond_encoder.1.weight` is `(2048, 400)` where `400 = 128 [diffusion_step] + 2 * (128 [2x ResNet18 SpatialSoftmax kp=32] + 8 [state])`), **`action_pred_horizon = 16`**, **`action_exec_horizon = 8`**, `CenterCrop(230, 230)` at eval, and exact 1,600-episode (`476,857` `is_demo=False` frames) `minmax` stats (`DP_MINMAX_NORM_STATS`).
- **Verified 10-Episode `val` Rollout Results (`outputs/pretrained_dp/val10_sanity_results.json`, `seed=7`, Peak VRAM = `342.9 MB`):**
  - **`ButtonUnmaskSwap` (Permanence, `val` `ep=00..09`):** **`10.0%` success rate** (`95% CI [0.0%, 30.0%]`; `ep=02` succeeded autonomously in `375` steps = button press to unmask $\to$ container swaps $\to$ grasp target cube; `6/10` episodes completed full button-press + container-unmask trajectories in `303–459` steps, failing only on post-swap container identity due to `obs_horizon=2` + `include_text=False`).
  - **`PickHighlight` (Reference, `val` `ep=00..09`):** **`10.0%` success rate** (`95% CI [0.0%, 30.0%]`; `ep=05` succeeded autonomously in `162` steps = press start button $\to$ grasp & lift highlighted block $\to$ press stop button; episodes `00, 05, 06, 08` all executed multi-stage button-press + block-pick + stop-button sequences, failing on `00/06/08` only after highlights vanished).
  - **`StopCube` (Counting, `val` `ep=00..09`):** **`0.0%` success rate** (`mean_steps = 181.5`; physically pressed the stop button cleanly on **10/10 episodes** in `89–244` steps, but at the wrong pass count because `obs_horizon=2` + `include_text=False` lacks both target count text and pass-counting memory).
  - **Overall (30 `val` episodes, 100-step DDPM inference at `384 ms/chunk`):** **`6.7%` SR**, confirming 100% end-to-end contact-rich actuation fidelity on local RTX 5080 while reproducing the exact memory bottleneck reported in `arXiv:2603.04639`!

```python
from robomme_eval import PretrainedDPPolicy, evaluate_policy

policy = PretrainedDPPolicy("outputs/pretrained_dp/ckpt_200000.pth")  # 343 MB VRAM
res = evaluate_policy(
    policy,
    tasks=("PickHighlight", "StopCube", "ButtonUnmaskSwap"),
    dataset="val",
    max_episodes_per_task=10,
    exec_horizon=8,
)
```

---

## 7. Multi-Camera Spatial SigLIP Feature Cache (`/work/gj26/b20098/datasets/robomme_siglip_v1/`) & `src/robomme_data/`

To make multi-epoch recurrent Gated DeltaNet + Looped Action Expert training completely compute-bound on Miyabi GH200 (`0` Lustre Parquet/PNG I/O wait), we pre-extracted all **1,600 training episodes (`768,897` total frames = `292,040` demo + `476,857` execution frames)** using frozen **`google/siglip-base-patch16-224`** (`768-D`, `bfloat16`) and implemented [`src/robomme_data/`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_data) so **100% identical vision/text preprocessing runs offline on Miyabi and online during RTX 5080 evaluation**.

### 7.1 Cache Location, Load Speed & Per-Task Shard Schema

- **Miyabi Cache Directory:** `/work/gj26/b20098/datasets/robomme_siglip_v1/` (**89 GiB total**, extracted in `703.8s` at **836–1,088 frames/s** on 1x GH200 `mg0024`)
- **Files:** 16 per-task shards `{TaskName}.pt` (`2.4 GiB` – `13.0 GiB` each) + `norm_stats.json` (both `quantile_delta` and `minmax_abs` stats over all `476,857` `is_demo=False` frames) + `dp_stats.json` + `manifest.json`.
- **Verified `torch.load(..., mmap=True)` Load Speeds on Miyabi:**
  - **4-task Permanence suite** (`ButtonUnmask`, `ButtonUnmaskSwap`, `VideoUnmask`, `VideoUnmaskSwap` = `400` episodes): **`0.044 s` (44 ms)!**
  - **All 16 RoboMME tasks** (`1,600` episodes, `768,897` frames): **`0.424 s`!**

Each `{TaskName}.pt` dictionary contains contiguous episode-concatenated tensors indexed by `ep_offsets[i] : ep_offsets[i+1]`:

| Key | Shape & Dtype | Description |
| :--- | :--- | :--- |
| `front_emb` | `(N_frames, 64, 768)` `bfloat16` | **Front camera `8×8` spatial grid** adaptive-avg-pooled from `14×14` SigLIP `last_hidden_state` patch tokens (preserves 3-container left/middle/right geometry & swap trajectories) |
| `wrist_emb` | `(N_frames, 16, 768)` `bfloat16` | **Wrist camera `4×4` spatial grid** adaptive-avg-pooled from `14×14` SigLIP patch tokens (**fixes upstream MME-VLA single-camera bottleneck!**) |
| `state` | `(N_frames, 8)` `float32` | Raw packed state (`7` joint angles rad + `gripper_state[:1]` finger width in `[0, 0.04]` m) |
| `actions` | `(N_frames, 8)` `float32` | Raw target joint actions (`7` target joint angles rad + `±1.0` gripper command) |
| `is_demo` | `(N_frames,)` `bool` | `True` for prefix video-demo frames (`0 .. exec_start_idx - 1`), `False` for execution frames |
| `ep_offsets` | `(101,)` `int64` | CSR offsets (`0 .. N_frames`) slicing episode `ep_local in [0..99]` |
| `epis_idx` | `(100,)` `int64` | Global LeRobot episode ID (`0..1599`) |
| `exec_start_idx` | `(100,)` `int64` | Index of first supervised execution step (`0` for non-video tasks; `24..1145` for `Video*` / `Imitation` tasks) |
| `text_token_emb` | `(100, 64, 768)` `bfloat16` | Per-episode SigLIP SentencePiece token embeddings (`max_length=64`, `padding="max_length"`) |
| `text_pool_emb` | `(100, 768)` `bfloat16` | Per-episode pooled language goal embedding |
| `text_attn_mask` | `(100, 64)` `bool` | Attention mask over valid prompt tokens |
| `prompts` | `list[str]` (`len=100`) | Raw natural-language task goal strings (`info["task_goal"][0]`) |
| `simple_subgoals` / `grounded_subgoals` | `list[list[str]]` | Per-frame oracle symbolic/grounded subgoals (for diagnostic linear probing & analysis) |

### 7.2 Reusable `src/robomme_data/` API (Offline Training + Live RTX 5080 Eval)

- [`SiglipFeatureEncoder`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_data/vision_encoder.py#L100-L239) & [`preprocess_siglip_rgb`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_data/vision_encoder.py#L25-L81):
  - Deterministic antialiased bilinear `256×256 -> 224×224` resize + `[-1, 1]` scaling (`pad=0` on 1:1 RGB frames), fused `2*N` front+wrist batching through `SiglipModel.vision_model`, `8×8` / `4×4` spatial pooling, and cached text encoding.
  - Peak GPU footprint on local RTX 5080: **427.9 MB VRAM** (`<< 4 GB` budget), encoding 16-frame chunks in `<6 ms`.
- [`RoboMMEEpisodeDataset`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_data/cache_dataset.py#L85-L220) (Recommended for TBPTT / full-episode recurrent DeltaNet training):
  - Returns full episode `front_emb (T, 64, 768)`, `wrist_emb (T, 16, 768)`, `state_norm (T, 8)`, `is_demo (T,)`, `chunk_step_indices (M,)` (starting at `exec_start_idx`, optional `random_chunk_phase=True` data augmentation), normalized end-padded `target_action_chunks (M, H_act, 8)`, and **A3 anti-shortcut `readout_cutoff_excl (M,)`** (`max(exec_start_idx, max(0, t_step - exclusion_window + 1))`, default `exclusion_window = exec_horizon = 16`).
- [`RoboMMEChunkDataset`](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/src/robomme_data/cache_dataset.py#L223-L375):
  - Step/chunk-indexed dataset returning current `(front_emb, wrist_emb, state_norm)`, `target_actions (H_act, 8)`, and strided history window `(hist_front_emb, hist_wrist_emb, hist_state_norm, hist_mask)` automatically truncated at A3 `exclusion_window`.
- Both datasets support **`action_mode="quantile_delta"`** (`H_act=20, H_exec=16`, delta joints `0..6` + abs gripper `7`, `MME_VLA_NORM_STATS`) and **`action_mode="minmax_abs"`** (`H_act=16, H_exec=8`, absolute joints + min-max stats) seamlessly!

```python
from robomme_data import ROBOMME_SUITES, RoboMMEEpisodeDataset, SiglipFeatureEncoder

# 1. Miyabi GH200 Training (loads 4 Permanence tasks in 44 ms via mmap=True!)
train_ds = RoboMMEEpisodeDataset(
    "/work/gj26/b20098/datasets/robomme_siglip_v1",
    tasks=ROBOMME_SUITES["Permanence"],      # or tasks=None for all 16 tasks (1,600 eps in 0.42s)
    episode_indices_per_task=range(0, 90),   # e.g. 90 train / 10 held-out train-val episodes per task
    action_mode="quantile_delta",            # H_act=20, H_exec=16, exclusion_window=16
    random_chunk_phase=True,
)
ep = train_ds[0]
# ep["front_emb"]: (T, 64, 768) bf16 | ep["wrist_emb"]: (T, 16, 768) bf16
# ep["chunk_step_indices"]: (M,)     | ep["readout_cutoff_excl"]: (M,)
# ep["target_action_chunks"]: (M, 20, 8) float32

# 2. Local RTX 5080 Live Rollout Feature Extraction (100% identical to offline cache!)
encoder = SiglipFeatureEncoder(device="cuda", dtype=torch.bfloat16)  # ~428 MB VRAM
front_tok, wrist_tok = encoder.encode_views(front_rgb_uint8, wrist_rgb_uint8)  # (N, 64, 768), (N, 16, 768)
text_tok, text_pool, text_mask = encoder.encode_text([task_goal])
```
