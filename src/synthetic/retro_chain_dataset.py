"""Task 1A-Retro: Shortcut-Free Retroactive Multi-Hop Relational Pointer Chaining.

Models occluded post-cue VLA relational reasoning:
- Over N = 32 intermediate entities/containers and Z = 8 Terminal Zones, episodes contain
  multiple disjoint chains (matched chain structure + distractor chains) and stale edge overwrites.
- All chains in an episode follow a matched temporal ordering distribution (`forward`, `reverse`,
  or `mixed`), eliminating any correlation between the last zone token in the stream and the
  queried chain's target zone!
- Target query `Query(StartNode)` is ONLY revealed at t = T - 1 (without k!).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class RetroChainDataset(Dataset):
    """Shortcut-free Retroactive Multi-Hop Pointer Chaining benchmark."""

    OP_NOOP = 0
    OP_BIND = 1
    OP_QUERY = 2

    NUM_OPS = 3
    NUM_NODES = 32
    NUM_ZONES = 8
    VOCAB_SIZE = 1 + NUM_NODES + NUM_ZONES  # 41

    ORDER_FORWARD = 0
    ORDER_REVERSE = 1
    ORDER_MIXED = 2
    ORDER_SINGLE = 3

    def __init__(
        self,
        num_samples: int,
        k_values: Sequence[int] = (1, 2, 3, 4),
        seq_len: int = 44,
        num_overwrites: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.k_values = list(k_values)
        self.seq_len = seq_len
        self.num_overwrites = num_overwrites
        self.max_k = max(self.k_values)

        rng = np.random.default_rng(seed)

        tokens = np.zeros((num_samples, seq_len, 3), dtype=np.int64)
        labels = np.zeros(num_samples, dtype=np.int64)
        k_hops = np.zeros(num_samples, dtype=np.int64)
        order_types = np.zeros(num_samples, dtype=np.int64)
        hop_timesteps = np.full((num_samples, self.max_k), -1, dtype=np.int64)

        node_offset = 1  # 1..32
        zone_offset = 1 + self.NUM_NODES  # 33..40

        for i in range(num_samples):
            target_k = self.k_values[i % len(self.k_values)]

            # Always build 6 chains: the queried chain of length target_k PLUS 5 distractor chains,
            # INCLUDING a twin chain of the exact same length `target_k` (when 2*target_k + 6 <= 18)
            # and all 6 chains assigned distinct or uniform terminal zones so "last zone in stream"
            # is near 1/6..1/8 (~14%)!
            if set(self.k_values) == {1, 2, 3, 4}:
                # 6 chains: [1, 2, 3, 4, 4, 3] = 17 edges <= 32 nodes!
                chain_lengths = [1, 2, 3, 4, 4, 3]
                queried_chain_idx = [1, 2, 3, 4].index(target_k)
            else:
                # OOD (target_k in {5, 6}): [target_k, target_k, 2, 2, 1] <= 17 edges!
                chain_lengths = [target_k, target_k, 2, 2, 1]
                queried_chain_idx = 0

            total_true_edges = sum(chain_lengths)
            chosen_nodes = rng.permutation(self.NUM_NODES)[:total_true_edges]

            # Assign distinct random zones across the chains so all chains cover different zones!
            chain_zones = rng.choice(self.NUM_ZONES, size=len(chain_lengths), replace=False)
            target_zone = int(chain_zones[queried_chain_idx])

            chains_edges: List[List[tuple[int, int]]] = []
            cursor = 0
            start_node_sym = 0
            for c_idx, c_len in enumerate(chain_lengths):
                c_nodes = chosen_nodes[cursor : cursor + c_len]
                cursor += c_len
                z_id = int(chain_zones[c_idx])
                edges_c: List[tuple[int, int]] = []
                for h_idx in range(c_len):
                    src_sym = node_offset + int(c_nodes[h_idx])
                    dst_sym = (
                        node_offset + int(c_nodes[h_idx + 1])
                        if h_idx < c_len - 1
                        else zone_offset + z_id
                    )
                    edges_c.append((src_sym, dst_sym))
                if c_idx == queried_chain_idx:
                    start_node_sym = edges_c[0][0]
                chains_edges.append(edges_c)

            if target_k == 1:
                ord_mode = self.ORDER_SINGLE
                ep_ord = i % 3
            else:
                ep_ord = i % 3
                ord_mode = (
                    self.ORDER_FORWARD
                    if ep_ord == 0
                    else (self.ORDER_REVERSE if ep_ord == 1 else self.ORDER_MIXED)
                )

            all_true_flat: List[tuple[int, int, int, int]] = []
            for c_idx, edges_c in enumerate(chains_edges):
                for h_idx, (s_sym, d_sym) in enumerate(edges_c):
                    all_true_flat.append((s_sym, d_sym, c_idx, h_idx))

            # Order EVERY chain in the episode using the same ordering regime `ep_ord`
            # so the queried chain's terminal edge has NO positional bias over distractor chains!
            perm = rng.permutation(len(all_true_flat)).tolist()
            for c_idx, c_len in enumerate(chain_lengths):
                if c_len <= 1:
                    continue
                c_pos = [
                    pos for pos, idx_e in enumerate(perm) if all_true_flat[idx_e][2] == c_idx
                ]
                c_e_idx = [
                    idx_e for idx_e in range(len(all_true_flat)) if all_true_flat[idx_e][2] == c_idx
                ]
                if ep_ord == 0:
                    ord_c = c_e_idx
                elif ep_ord == 1:
                    ord_c = list(reversed(c_e_idx))
                else:
                    if c_len == 2:
                        if rng.random() < 0.5:
                            ord_c = c_e_idx
                            if c_idx == queried_chain_idx:
                                ord_mode = self.ORDER_FORWARD
                        else:
                            ord_c = list(reversed(c_e_idx))
                            if c_idx == queried_chain_idx:
                                ord_mode = self.ORDER_REVERSE
                    else:
                        while True:
                            cand = rng.permutation(c_e_idx).tolist()
                            if cand != c_e_idx and cand != list(reversed(c_e_idx)):
                                ord_c = cand
                                break
                for pos_p, e_id in zip(c_pos, ord_c):
                    perm[pos_p] = e_id

            events_chronological: List[tuple[int, int, int, int, bool]] = [
                (*all_true_flat[e_idx], True) for e_idx in perm
            ]

            ow_edge_indices = rng.choice(
                len(all_true_flat),
                size=min(num_overwrites, len(all_true_flat)),
                replace=False,
            ).tolist()
            for ow_idx in ow_edge_indices:
                s_sym, true_d_sym, _, _ = all_true_flat[ow_idx]
                true_pos = next(
                    p for p, ev in enumerate(events_chronological) if ev[4] and ev[0] == s_sym
                )
                while True:
                    wrong_d = int(rng.integers(1, self.VOCAB_SIZE))
                    if wrong_d != true_d_sym and wrong_d != s_sym:
                        break
                ins_pos = int(rng.integers(0, true_pos + 1))
                events_chronological.insert(ins_pos, (s_sym, wrong_d, -1, -1, False))

            total_events = len(events_chronological)
            avail_slots = seq_len - 1
            assert total_events <= avail_slots
            event_slots = np.sort(rng.choice(avail_slots, size=total_events, replace=False))

            for slot_t, (s_sym, d_sym, c_idx, h_idx, is_true) in zip(
                event_slots, events_chronological
            ):
                tokens[i, slot_t, 0] = self.OP_BIND
                tokens[i, slot_t, 1] = s_sym
                tokens[i, slot_t, 2] = d_sym
                if is_true and c_idx == queried_chain_idx:
                    hop_timesteps[i, h_idx] = slot_t

            tokens[i, seq_len - 1, 0] = self.OP_QUERY
            tokens[i, seq_len - 1, 1] = start_node_sym
            tokens[i, seq_len - 1, 2] = 0

            labels[i] = target_zone
            k_hops[i] = target_k
            order_types[i] = ord_mode

        self.tokens = torch.from_numpy(tokens)
        self.labels = torch.from_numpy(labels)
        self.k_hops = torch.from_numpy(k_hops)
        self.order_types = torch.from_numpy(order_types)
        self.hop_timesteps = torch.from_numpy(hop_timesteps)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[idx],
            "label": self.labels[idx],
            "k": self.k_hops[idx],
            "order_type": self.order_types[idx],
            "hop_timesteps": self.hop_timesteps[idx],
        }


def audit_retro_chain_dataset(ds: RetroChainDataset) -> Dict[str, float]:
    """Verify uniform zone distribution and failure of last-token / stream-end shortcuts."""
    y = ds.labels.numpy()
    toks = ds.tokens.numpy()
    counts = np.bincount(y, minlength=ds.NUM_ZONES) / len(y)
    zone_offset = 1 + ds.NUM_NODES

    last_zone_hits = 0
    for i in range(len(y)):
        bind_zones = [
            toks[i, t, 2] - zone_offset
            for t in range(ds.seq_len - 1)
            if toks[i, t, 0] == ds.OP_BIND and toks[i, t, 2] >= zone_offset
        ]
        pred = bind_zones[-1] if bind_zones else -1
        if pred == y[i]:
            last_zone_hits += 1

    return {
        "chance_level": 1.0 / ds.NUM_ZONES,
        "max_class_marginal": float(counts.max()),
        "heuristic_last_zone_in_stream_acc": last_zone_hits / len(y),
    }
