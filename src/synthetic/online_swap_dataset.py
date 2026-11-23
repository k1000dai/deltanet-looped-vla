"""Task 1A-Online: Shortcut-Free Online Container Swap State Tracking Benchmark.

Tests Opus 5.5's critique directly:
- M = 6 distinct objects placed into 6 of C = 8 distinct cups at t = 0..5.
- Target object m* is NOT revealed until t = T - 1, forcing simultaneous tracking of all 6 objects.
- Fixed total swap count `total_swaps = 12` in EVERY episode (both ID k in {1..4} and OOD k in {5, 6}),
  with k relevant swaps moving m* and (12 - k) distractor swaps between cups NOT holding m*,
  interleaved uniformly at random so neither total swap count nor the final swap token leaks k or y!
- Final cup distribution is uniform across all C = 8 cups (chance = 12.5%).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class OnlineSwapDataset(Dataset):
    """Pre-generated shortcut-free Online Container Swap dataset."""

    OP_NOOP = 0
    OP_PLACE = 1
    OP_SWAP = 2
    OP_QUERY = 3

    NUM_OPS = 4
    NUM_OBJECTS = 6
    NUM_CUPS = 8
    VOCAB_SIZE = 1 + NUM_OBJECTS + NUM_CUPS  # 15

    def __init__(
        self,
        num_samples: int,
        k_values: Sequence[int] = (1, 2, 3, 4),
        seq_len: int = 40,
        total_swaps: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.k_values = list(k_values)
        self.seq_len = seq_len
        self.total_swaps = total_swaps

        rng = np.random.default_rng(seed)

        tokens = np.zeros((num_samples, seq_len, 3), dtype=np.int64)
        labels = np.zeros(num_samples, dtype=np.int64)
        k_hops = np.zeros(num_samples, dtype=np.int64)
        init_cups = np.zeros(num_samples, dtype=np.int64)
        last_swap_a = np.zeros(num_samples, dtype=np.int64)
        last_swap_b = np.zeros(num_samples, dtype=np.int64)

        obj_offset = 1
        cup_offset = 1 + self.NUM_OBJECTS  # 7..14

        for i in range(num_samples):
            k = self.k_values[i % len(self.k_values)]
            assert k <= total_swaps

            placed_cups = rng.choice(self.NUM_CUPS, size=self.NUM_OBJECTS, replace=False)
            obj_order = rng.permutation(self.NUM_OBJECTS)

            for step_idx, obj_id in enumerate(obj_order):
                tokens[i, step_idx, 0] = self.OP_PLACE
                tokens[i, step_idx, 1] = obj_offset + obj_id
                tokens[i, step_idx, 2] = cup_offset + placed_cups[obj_id]

            target_obj = int(rng.integers(0, self.NUM_OBJECTS))
            cur_cup = int(placed_cups[target_obj])
            init_cups[i] = cur_cup

            avail_slots = seq_len - 1 - self.NUM_OBJECTS
            assert total_swaps <= avail_slots

            event_times = np.sort(
                rng.choice(avail_slots, size=total_swaps, replace=False) + self.NUM_OBJECTS
            )
            rel_mask = np.zeros(total_swaps, dtype=bool)
            rel_positions = rng.choice(total_swaps, size=k, replace=False)
            rel_mask[rel_positions] = True

            for ev_idx, t_step in enumerate(event_times):
                other_cups = [c for c in range(self.NUM_CUPS) if c != cur_cup]
                if rel_mask[ev_idx]:
                    next_cup = int(rng.choice(other_cups))
                    c1, c2 = cur_cup, next_cup
                    cur_cup = next_cup
                else:
                    d_pair = rng.choice(other_cups, size=2, replace=False)
                    c1, c2 = int(d_pair[0]), int(d_pair[1])

                if rng.random() < 0.5:
                    c1, c2 = c2, c1

                tokens[i, t_step, 0] = self.OP_SWAP
                tokens[i, t_step, 1] = cup_offset + c1
                tokens[i, t_step, 2] = cup_offset + c2

                if ev_idx == total_swaps - 1:
                    last_swap_a[i] = c1
                    last_swap_b[i] = c2

            tokens[i, seq_len - 1, 0] = self.OP_QUERY
            tokens[i, seq_len - 1, 1] = obj_offset + target_obj
            tokens[i, seq_len - 1, 2] = 0

            labels[i] = cur_cup
            k_hops[i] = k

        self.tokens = torch.from_numpy(tokens)
        self.labels = torch.from_numpy(labels)
        self.k_hops = torch.from_numpy(k_hops)
        self.init_cups = torch.from_numpy(init_cups)
        self.last_swap_a = torch.from_numpy(last_swap_a)
        self.last_swap_b = torch.from_numpy(last_swap_b)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[idx],
            "label": self.labels[idx],
            "k": self.k_hops[idx],
        }


def audit_online_swap_dataset(ds: OnlineSwapDataset) -> Dict[str, float]:
    """Verify absence of label bias, initial-cup shortcut, and last-swap arg1/non-arg shortcuts."""
    y = ds.labels.numpy()
    counts = np.bincount(y, minlength=ds.NUM_CUPS) / len(y)
    init_acc = float((y == ds.init_cups.numpy()).mean())
    last_swap_arg1_acc = float((y == ds.last_swap_a.numpy()).mean())
    in_last_swap = float(((y == ds.last_swap_a.numpy()) | (y == ds.last_swap_b.numpy())).mean())
    # Bayes-optimal single-cup guess conditioned ONLY on the last swap token {a, b}:
    bayes_last_swap_acc = max(in_last_swap / 2.0, (1.0 - in_last_swap) / (ds.NUM_CUPS - 2))
    return {
        "chance_level": 1.0 / ds.NUM_CUPS,
        "max_class_marginal": float(counts.max()),
        "heuristic_init_cup_acc": init_acc,
        "heuristic_last_swap_arg1_acc": last_swap_arg1_acc,
        "bayes_optimal_from_last_swap_only": bayes_last_swap_acc,
    }
