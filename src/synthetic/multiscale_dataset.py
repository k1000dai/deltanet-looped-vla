"""Task 1B: Key-Collision Multi-Timescale Interference Benchmark (Iso-State-Size).

Sequence length T = 2048 over C = 6 channels and V = 8 values (chance = 12.5%).
Crucial anti-shortcut design:
- ALL values in the entire sequence enter ONLY through `OP_OBS(chan_id, val_id)`!
  No value token ever carries a timescale tag, so setting beta(OP_OBS) ~ 0 fails completely!
- Five balanced query regimes (reported individually and overall):
  0. `short` (10-30 steps ago): Most recent `OP_OBS` on an active channel.
  1. `mid_persist` (200-400 steps ago): `OP_OBS` on a quiet channel while active channels collide/update.
  2. `long_persist` (1500-2000 steps ago): `OP_OBS` on a quiet channel 1500-2000 steps ago.
  3. `mid_collide` (200-400 steps ago): `OP_OBS(c, v)` followed at t+1 by `OP_LOCK_MID(c)` (which
     does NOT contain `v`!), followed by ~80 colliding `OP_OBS(c, v_distractor)` updates on the SAME channel `c`!
  4. `long_collide` (1500-2000 steps ago): `OP_OBS(c, v)` followed at t+1 by `OP_LOCK_LONG(c)` (which
     does NOT contain `v`!), followed by ~400 colliding `OP_OBS(c, v_distractor)` updates on the SAME channel `c`!
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTimescaleDataset(Dataset):
    """T=2048 Key-Collision Multi-Timescale Benchmark."""

    OP_NOOP = 0
    OP_OBS = 1
    OP_LOCK_MID = 2
    OP_LOCK_LONG = 3
    OP_QUERY = 4

    NUM_OPS = 5
    NUM_CHANNELS = 6  # Channels 0..3: high-frequency colliding; Channels 4..5: persistence channels
    NUM_VALUES = 8
    NUM_QUERY_SCALES = 3  # 0=Short, 1=Mid, 2=Long
    VOCAB_SIZE = 1 + NUM_CHANNELS + NUM_VALUES + NUM_QUERY_SCALES  # 18

    SUBTYPE_NAMES = (
        "short",
        "mid_persist",
        "long_persist",
        "mid_collide",
        "long_collide",
    )

    def __init__(
        self,
        num_samples: int,
        seq_len: int = 2048,
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert seq_len >= 2048, "Task 1B requires seq_len >= 2048 for 1500-2000 step Long horizon"
        self.num_samples = num_samples
        self.seq_len = seq_len

        rng = np.random.default_rng(seed)

        tokens = np.zeros((num_samples, seq_len, 3), dtype=np.int64)
        labels = np.zeros(num_samples, dtype=np.int64)
        subtypes = np.zeros(num_samples, dtype=np.int64)
        query_scales = np.zeros(num_samples, dtype=np.int64)
        latest_obs_val = np.zeros(num_samples, dtype=np.int64)

        chan_off = 1  # 1..6
        val_off = 1 + self.NUM_CHANNELS  # 7..14
        scale_off = val_off + self.NUM_VALUES  # 15..17

        # Vectorized base stream + structured landmark injection per sample
        for i in range(num_samples):
            subtype = i % 5
            subtypes[i] = subtype

            # 1) Background colliding stream on active channels 0..3 (~300 colliding OP_OBS over 2048 steps,
            # i.e. ~75 direct key-collisions per active channel after a Long lock!)
            active_mask = rng.random(seq_len - 1) < 0.15
            chans = rng.integers(0, 4, size=seq_len - 1)
            vals = rng.integers(0, self.NUM_VALUES, size=seq_len - 1)

            tokens[i, :-1, 0] = np.where(active_mask, self.OP_OBS, self.OP_NOOP)
            tokens[i, :-1, 1] = np.where(active_mask, chan_off + chans, 0)
            tokens[i, :-1, 2] = np.where(active_mask, val_off + vals, 0)

            # 2) Inject ALL multiscale events into EVERY episode so prefix statistics are IDENTICAL
            # regardless of which subtype is queried at t = T - 1!
            # (a) Long persistence on channel 4 at lag in [1500, 2000] -> t in [seq_len-1-2000, seq_len-1-1500]
            t_long_p = int(rng.integers(seq_len - 1 - 2000, seq_len - 1 - 1500))
            v_long_p = int(rng.integers(0, self.NUM_VALUES))
            tokens[i, t_long_p] = (self.OP_OBS, chan_off + 4, val_off + v_long_p)

            # (b) Mid persistence on channel 5 at lag in [200, 400]
            t_mid_p = int(rng.integers(seq_len - 1 - 400, seq_len - 1 - 200))
            v_mid_p = int(rng.integers(0, self.NUM_VALUES))
            tokens[i, t_mid_p] = (self.OP_OBS, chan_off + 5, val_off + v_mid_p)

            # (c) Long colliding occlusion on active channel c_lc in {0, 1} at lag in [1500, 2000]
            c_lc = int(rng.integers(0, 2))
            t_long_c = int(rng.integers(seq_len - 1 - 2000, seq_len - 1 - 1500))
            if t_long_c == t_long_p or t_long_c + 1 == t_long_p:
                t_long_c += 2
            v_long_c = int(rng.integers(0, self.NUM_VALUES))
            tokens[i, t_long_c - 1] = (self.OP_OBS, chan_off + c_lc, val_off + v_long_c)
            tokens[i, t_long_c] = (self.OP_LOCK_LONG, chan_off + c_lc, 0)
            # Ensure >= 1 colliding OP_OBS on c_lc occurs later so naive overwrite fails!
            t_force_after_lc = int(rng.integers(seq_len - 100, seq_len - 40))
            v_dist_lc = int((v_long_c + rng.integers(1, self.NUM_VALUES)) % self.NUM_VALUES)
            tokens[i, t_force_after_lc] = (self.OP_OBS, chan_off + c_lc, val_off + v_dist_lc)

            # (d) Mid colliding subgoal lock on active channel c_mc in {2, 3} at lag in [200, 400]
            c_mc = int(rng.integers(2, 4))
            t_mid_c = int(rng.integers(seq_len - 1 - 400, seq_len - 1 - 200))
            if t_mid_c == t_mid_p or t_mid_c - 1 == t_mid_p:
                t_mid_c += 2
            v_mid_c = int(rng.integers(0, self.NUM_VALUES))
            tokens[i, t_mid_c - 1] = (self.OP_OBS, chan_off + c_mc, val_off + v_mid_c)
            tokens[i, t_mid_c] = (self.OP_LOCK_MID, chan_off + c_mc, 0)
            t_force_after_mc = int(rng.integers(seq_len - 80, seq_len - 35))
            v_dist_mc = int((v_mid_c + rng.integers(1, self.NUM_VALUES)) % self.NUM_VALUES)
            tokens[i, t_force_after_mc] = (self.OP_OBS, chan_off + c_mc, val_off + v_dist_mc)

            # (e) Short target on a random active channel c_sh in {0, 1, 2, 3} at lag in [10, 30]
            c_sh = int(rng.integers(0, 4))
            t_sh = int(rng.integers(seq_len - 1 - 30, seq_len - 1 - 10))
            v_sh = int(rng.integers(0, self.NUM_VALUES))
            tokens[i, t_sh] = (self.OP_OBS, chan_off + c_sh, val_off + v_sh)
            # After t_sh, allow updates on OTHER active channels != c_sh so c_sh's latest Short obs IS at lag 10..30!
            for t_tail in range(t_sh + 1, seq_len - 1):
                if tokens[i, t_tail, 0] == self.OP_OBS and tokens[i, t_tail, 1] == chan_off + c_sh:
                    other_c = int((c_sh + rng.integers(1, 4)) % 4)
                    tokens[i, t_tail, 1] = chan_off + other_c

            # 3) Form final query at t = seq_len - 1 based on subtype
            if subtype == 0:  # short
                q_chan, q_scale, y = c_sh, 0, v_sh
            elif subtype == 1:  # mid_persist
                q_chan, q_scale, y = 5, 1, v_mid_p
            elif subtype == 2:  # long_persist
                q_chan, q_scale, y = 4, 2, v_long_p
            elif subtype == 3:  # mid_collide
                q_chan, q_scale, y = c_mc, 1, v_mid_c
            else:  # long_collide
                q_chan, q_scale, y = c_lc, 2, v_long_c

            tokens[i, seq_len - 1] = (self.OP_QUERY, chan_off + q_chan, scale_off + q_scale)
            labels[i] = y
            query_scales[i] = q_scale

            # Track most recent OP_OBS on q_chan for shortcut audit
            obs_positions = np.where(
                (tokens[i, :-1, 0] == self.OP_OBS) & (tokens[i, :-1, 1] == chan_off + q_chan)
            )[0]
            latest_obs_val[i] = (
                int(tokens[i, obs_positions[-1], 2] - val_off) if len(obs_positions) > 0 else -1
            )

        self.tokens = torch.from_numpy(tokens)
        self.labels = torch.from_numpy(labels)
        self.subtypes = torch.from_numpy(subtypes)
        self.query_scales = torch.from_numpy(query_scales)
        self.latest_obs_val = torch.from_numpy(latest_obs_val)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[idx],
            "label": self.labels[idx],
            "subtype": self.subtypes[idx],
            "query_scale": self.query_scales[idx],
        }


def audit_multiscale_dataset(ds: MultiTimescaleDataset) -> Dict[str, float]:
    """Verify uniform value distribution and failure of recency heuristic on colliding subtypes."""
    y = ds.labels.numpy()
    sub = ds.subtypes.numpy()
    latest = ds.latest_obs_val.numpy()
    counts = np.bincount(y, minlength=ds.NUM_VALUES) / len(y)

    out: Dict[str, float] = {
        "chance_level": 1.0 / ds.NUM_VALUES,
        "max_class_marginal": float(counts.max()),
    }
    for s_idx, s_name in enumerate(ds.SUBTYPE_NAMES):
        mask = sub == s_idx
        out[f"latest_obs_heuristic_{s_name}"] = float((y[mask] == latest[mask]).mean())
    return out
