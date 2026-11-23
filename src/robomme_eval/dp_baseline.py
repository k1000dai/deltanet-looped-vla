"""Pretrained RoboMME Diffusion Policy (`Yinpei/DP-Unet-ckpt`) wrapper implementing `StatefulPolicy`.

Architecture & normalization verified directly against `ckpt_200000.pth` state_dict shapes and
`external/DP/{serve_dp.py, eval_envs/model/dp_unet.py, eval_envs/config/train_dp_unet_obs2.yaml}`:
- `obs_horizon = 2`, `action_pred_horizon = 16`, `action_exec_horizon = 8`, `include_text = False`
- ResNet18 + `SpatialSoftmax(num_kp=32)` (`64-D` x 2 cameras = `128-D`) + `8-D` `minmax`-normalized state
  -> `2 * 136 = 272-D` + `128-D` diffusion step embedding = `400-D` UNet conditioning (`[256, 512, 1024]`)
- `CenterCrop(230, 230)` + `Normalize([0.5]*3, [0.5]*3)` inside `DiffusionPolicyUnet.eval()`
- Exact 1,600-episode (`476,857` `is_demo=False` execution frames) `minmax` stats embedded below.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from .interface import ActionStateNormalizer, PolicyChunkOutput, StatefulPolicy, to_uint8_rgb

# Exact 1,600-episode (`476,857` `is_demo=False` execution frames) stats computed over
# `Yinpei/robomme_data_lerobot` matching `RoboMME/DP` `RoboMMEDataset._compute_stats`.
DP_MINMAX_NORM_STATS: dict[str, dict[str, tuple[float, ...]]] = {
    "state": {
        "min": (
            -0.6679782867431641,
            -0.5672716498374939,
            -0.7225731015205383,
            -3.0687756538391113,
            -0.5970768928527832,
            1.5707963705062866,
            -1.5199806690216064,
            0.0,
        ),
        "max": (
            0.6571396589279175,
            1.3945214748382568,
            0.6758602261543274,
            -0.5983060598373413,
            0.711341917514801,
            3.2234132289886475,
            2.8839821815490723,
            0.03999999910593033,
        ),
    },
    "actions": {
        "min": (
            -0.6683933734893799,
            -0.5712698698043823,
            -0.7256401777267456,
            -3.06937575340271,
            -0.6007632613182068,
            1.5707963705062866,
            -1.5224655866622925,
            -1.0,
        ),
        "max": (
            0.658385694026947,
            1.3966323137283325,
            0.678263247013092,
            -0.5900649428367615,
            0.7127829790115356,
            3.2284109592437744,
            2.8888230323791504,
            1.0,
        ),
    },
}


def _strip_ddp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    clean: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k = k.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        while k.startswith("module."):
            k = k[len("module.") :]
        clean[k] = v
    return clean


def _to_chw_float01(rgb_uint8: np.ndarray) -> np.ndarray:
    arr = to_uint8_rgb(rgb_uint8)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0


class PretrainedDPPolicy(StatefulPolicy):
    """Wraps `Yinpei/DP-Unet-ckpt` (`DiffusionPolicyUnet`) into `src/robomme_eval`."""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        dp_repo_dir: str | Path | None = None,
        device: str = "cuda",
        obs_horizon: int = 2,
        action_pred_horizon: int = 16,
        action_exec_horizon: int = 8,
        num_inference_steps: int = 100,
    ) -> None:
        if dp_repo_dir is None:
            dp_repo_dir = Path(__file__).resolve().parents[2] / "external" / "DP"
        dp_repo_str = str(Path(dp_repo_dir).resolve())
        if dp_repo_str not in sys.path:
            sys.path.insert(0, dp_repo_str)

        from eval_envs.model.dp_unet import DiffusionPolicyUnet

        self.device = torch.device(device)
        self.obs_horizon = obs_horizon
        self.action_pred_horizon = action_pred_horizon
        self.action_exec_horizon = action_exec_horizon

        shape_meta = SimpleNamespace(
            image=SimpleNamespace(shape=[6, 256, 256]),
            state=SimpleNamespace(shape=[8]),
            action=SimpleNamespace(shape=[8]),
        )
        obs_encoder_cfg = SimpleNamespace(name="resnet18", output_dim=64, num_kp=32)
        noise_sched_cfg = SimpleNamespace(
            num_train_timesteps=100,
            num_inference_steps=num_inference_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        image_aug_cfg = SimpleNamespace(crop_ratio=0.9)

        self.model = DiffusionPolicyUnet(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder_cfg,
            noise_scheduler=noise_sched_cfg,
            obs_horizon=obs_horizon,
            action_exec_horizon=action_exec_horizon,
            action_pred_horizon=action_pred_horizon,
            image_aug=image_aug_cfg,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            diffusion_step_embed_dim=128,
            include_text=False,
            device=str(self.device),
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(_strip_ddp_prefix(ckpt["model"]), strict=True)
        self.model.to(self.device).eval()

        self.normalizer = ActionStateNormalizer(mode="minmax_abs", stats=DP_MINMAX_NORM_STATS)
        self._img_buf: list[np.ndarray] = []
        self._state_buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._img_buf.clear()
        self._state_buf.clear()

    def _push_obs(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray) -> None:
        fused = np.concatenate([_to_chw_float01(front_rgb), _to_chw_float01(wrist_rgb)], axis=0)
        self._img_buf.append(fused)
        self._state_buf.append(np.asarray(state, dtype=np.float32).copy())
        if len(self._img_buf) > self.obs_horizon:
            self._img_buf.pop(0)
            self._state_buf.pop(0)

    def ingest_demo_frames(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        # Upstream `serve_dp.py` explicitly ignores demo frames (`DP is non-recurrent`).
        return None

    def ingest_step_history(
        self,
        front_rgb_list: Sequence[np.ndarray],
        wrist_rgb_list: Sequence[np.ndarray],
        state_list: Sequence[np.ndarray],
        task_goal: str,
    ) -> None:
        # Push trailing step observations so `_img_buf` holds `[frame_{t-1}, frame_t]` (`dt = 1/fps`)
        for fr, wr, st in zip(
            front_rgb_list[-self.obs_horizon :],
            wrist_rgb_list[-self.obs_horizon :],
            state_list[-self.obs_horizon :],
        ):
            self._push_obs(fr, wr, st)

    @torch.inference_mode()
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
        self._push_obs(front_rgb, wrist_rgb, state)
        n = len(self._img_buf)
        if n < self.obs_horizon:
            pad = self.obs_horizon - n
            imgs = np.stack([self._img_buf[0]] * pad + self._img_buf, axis=0)
            states = np.stack([self._state_buf[0]] * pad + self._state_buf, axis=0)
        else:
            imgs = np.stack(self._img_buf[-self.obs_horizon :], axis=0)
            states = np.stack(self._state_buf[-self.obs_horizon :], axis=0)

        states_norm = self.normalizer.normalize_state(states)
        obs_dict = {
            "image": torch.from_numpy(imgs[None, ...]).to(self.device, dtype=torch.float32),
            "state": torch.from_numpy(states_norm[None, ...]).to(self.device, dtype=torch.float32),
        }
        out_dict, _ = self.model.predict_action(obs_dict, policy_info={})
        act_norm = out_dict["action"][0].detach().cpu().numpy()  # (exec_horizon, 8)
        act_abs = self.normalizer.decode_actions(act_norm)
        return PolicyChunkOutput(actions=act_abs, loop_depth=1)
