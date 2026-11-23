"""Iso-State-Size Multi-Timescale Gated DeltaNet with Sparse Beta Consolidation.

Implements all 7+1 variants for Cycle 1 Task 1B under strictly identical total
state size (sum_i H_i * d_k * d_v = 1536 floats):
  - "1_single_mh": Single Multi-Head GDN (H=6, d_k=16, d_v=16, diverse head decay init)
  - "1b_stacked_3layer": 3-Layer Stacked GDN without band constraints / sparse beta gate
  - "2_3bank_same_fixed": 3-Bank (3 x H=2), identical fixed decay (hl = 15)
  - "3_3bank_diverse_fixed": 3-Bank (3 x H=2), diverse fixed decays (hl = 10..2500)
  - "4_3bank_learnable": 3-Bank (3 x H=2), band-constrained learnable decays in fp32
  - "5a_3bank_consolidation": (4) + beta-Gated Sparse Consolidation reading S_{t-1}^{(<i)}
  - "5b_3bank_routing": (4) + Scale-Routed Readout
  - "5c_3bank_full": Full (4) + beta-Gated Sparse Consolidation + Scale-Routed Readout
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from .gated_deltanet import GatedDeltaNetLayer, l2_normalize
from .readout import MemoryReadout, ScaleRoutedReadout


Variant1B = Literal[
    "1_single_mh",
    "1b_stacked_3layer",
    "2_3bank_same_fixed",
    "3_3bank_diverse_fixed",
    "4_3bank_learnable",
    "5a_3bank_consolidation",
    "5b_3bank_routing",
    "5c_3bank_full",
]


def halflife_to_alpha(hl: float) -> float:
    return float(2.0 ** (-1.0 / hl))


class MultiTimescaleGDN(nn.Module):
    """Unified Iso-State-Size Multi-Timescale Memory + Readout Module."""

    def __init__(
        self,
        d_model: int = 96,
        variant: Variant1B = "5c_3bank_full",
        d_k: int = 16,
        d_v: int = 16,
        beta_max: float = 1.0,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.variant = variant
        self.d_k = d_k
        self.d_v = d_v
        self.chunk_size = chunk_size

        self.use_consolidation = variant in ("5a_3bank_consolidation", "5c_3bank_full")
        self.use_routing = variant in ("5b_3bank_routing", "5c_3bank_full")

        if variant == "1_single_mh":
            self.single_bank = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=6,
                d_k=d_k,
                d_v=d_v,
                beta_max=beta_max,
                decay_mode="learnable",
                min_halflife=10.0,
                max_halflife=2500.0,
                chunk_size=chunk_size,
            )
            self.single_readout = MemoryReadout(
                d_model=d_model, num_heads=6, d_k=d_k, d_v=d_v
            )
            self.banks = None
            self.multi_readout = None

        elif variant == "1b_stacked_3layer":
            self.banks = nn.ModuleList(
                [
                    GatedDeltaNetLayer(
                        d_model=d_model,
                        num_heads=2,
                        d_k=d_k,
                        d_v=d_v,
                        beta_max=beta_max,
                        decay_mode="learnable",
                        min_halflife=10.0,
                        max_halflife=2500.0,
                        use_seq_readout=(i < 2),
                        chunk_size=chunk_size,
                    )
                    for i in range(3)
                ]
            )
            self.stack_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(2)])
            self.multi_readout = ScaleRoutedReadout(
                d_model=d_model,
                num_banks=3,
                heads_per_bank=2,
                d_k=d_k,
                d_v=d_v,
                use_routing=False,
            )
            self.single_bank = None
            self.single_readout = None

        else:
            self.single_bank = None
            self.single_readout = None

            if variant == "2_3bank_same_fixed":
                a_same = halflife_to_alpha(15.0)
                bank_specs = [
                    dict(decay_mode="fixed", fixed_alphas=(a_same, a_same)),
                    dict(decay_mode="fixed", fixed_alphas=(a_same, a_same)),
                    dict(decay_mode="fixed", fixed_alphas=(a_same, a_same)),
                ]
            elif variant == "3_3bank_diverse_fixed":
                bank_specs = [
                    dict(
                        decay_mode="fixed",
                        fixed_alphas=(halflife_to_alpha(10.0), halflife_to_alpha(25.0)),
                    ),
                    dict(
                        decay_mode="fixed",
                        fixed_alphas=(halflife_to_alpha(200.0), halflife_to_alpha(400.0)),
                    ),
                    dict(
                        decay_mode="fixed",
                        fixed_alphas=(halflife_to_alpha(1500.0), halflife_to_alpha(2500.0)),
                    ),
                ]
            else:
                bank_specs = [
                    dict(
                        decay_mode="band",
                        alpha_bounds=(halflife_to_alpha(4.0), halflife_to_alpha(40.0)),
                    ),
                    dict(
                        decay_mode="band",
                        alpha_bounds=(halflife_to_alpha(80.0), halflife_to_alpha(600.0)),
                    ),
                    dict(
                        decay_mode="band",
                        alpha_bounds=(halflife_to_alpha(1000.0), halflife_to_alpha(6000.0)),
                    ),
                ]

            self.banks = nn.ModuleList(
                [
                    GatedDeltaNetLayer(
                        d_model=d_model,
                        num_heads=2,
                        d_k=d_k,
                        d_v=d_v,
                        beta_max=beta_max,
                        chunk_size=chunk_size,
                        **spec,
                    )
                    for spec in bank_specs
                ]
            )

            if self.use_consolidation:
                self.cons_q_mid = nn.Linear(d_model, 2 * d_k, bias=False)
                self.cons_o_mid = nn.Linear(2 * d_v, d_model, bias=False)
                self.cons_gate_mid = nn.Linear(2 * d_model, 2, bias=True)

                self.cons_q_long_from_short = nn.Linear(d_model, 2 * d_k, bias=False)
                self.cons_q_long_from_mid = nn.Linear(d_model, 2 * d_k, bias=False)
                self.cons_o_long = nn.Linear(4 * d_v, d_model, bias=False)
                self.cons_gate_long = nn.Linear(2 * d_model, 2, bias=True)

                nn.init.constant_(self.cons_gate_mid.bias, -2.0)
                nn.init.constant_(self.cons_gate_long.bias, -2.0)

            self.multi_readout = ScaleRoutedReadout(
                d_model=d_model,
                num_banks=3,
                heads_per_bank=2,
                d_k=d_k,
                d_v=d_v,
                use_routing=self.use_routing,
            )

    @property
    def total_state_numel(self) -> int:
        if self.single_bank is not None:
            return self.single_bank.state_numel
        assert self.banks is not None
        return sum(b.state_numel for b in self.banks)

    def write_sequence(
        self,
        z: torch.Tensor,
        non_noop_gate: Optional[torch.Tensor] = None,
        read_from: Literal["S_prev", "S_curr"] = "S_prev",
    ) -> Dict[str, object]:
        """Process full sequence z: (B, T, d_model) via fast chunkwise Gated Delta Rule."""
        B, T, _ = z.shape
        if self.single_bank is not None:
            out = self.single_bank(
                z,
                beta_gate=non_noop_gate,
                read_from=read_from,
                return_penultimate_state=(read_from == "S_prev"),
            )
            S_mem = out["S_penult"] if read_from == "S_prev" else out["S_final"]
            n_mem = out["n_penult"] if read_from == "S_prev" else out["n_final"]
            return {
                "S_single": S_mem,
                "n_single": n_mem,
                "mean_beta": out["beta"].mean(dim=(0, 1)).detach(),
                "mean_alpha": torch.exp(out["g"]).mean(dim=(0, 1)).detach(),
            }

        assert self.banks is not None
        if self.variant == "1b_stacked_3layer":
            S_list: List[torch.Tensor] = []
            n_list: List[Optional[torch.Tensor]] = []
            betas = []
            alphas = []
            cur = z
            for i, bank in enumerate(self.banks):
                res = bank(
                    cur,
                    beta_gate=non_noop_gate,
                    read_from="S_prev",
                    return_penultimate_state=(read_from == "S_prev"),
                )
                S_i = res["S_penult"] if read_from == "S_prev" else res["S_final"]
                n_i = res["n_penult"] if read_from == "S_prev" else res["n_final"]
                assert S_i is not None
                S_list.append(S_i)
                n_list.append(n_i)
                betas.append(res["beta"].mean(dim=(0, 1)).detach())
                alphas.append(torch.exp(res["g"]).mean(dim=(0, 1)).detach())
                if i < 2 and res["seq_out"] is not None:
                    cur = self.stack_norms[i](cur + res["seq_out"])
            return {
                "S_banks": S_list,
                "n_banks": n_list,
                "mean_beta": torch.cat(betas, dim=0),
                "mean_alpha": torch.cat(alphas, dim=0),
            }

        if not self.use_consolidation:
            S_list = []
            n_list = []
            betas = []
            alphas = []
            for bank in self.banks:
                res = bank(
                    z,
                    beta_gate=non_noop_gate,
                    read_from=read_from,
                    return_penultimate_state=(read_from == "S_prev"),
                )
                S_i = res["S_penult"] if read_from == "S_prev" else res["S_final"]
                n_i = res["n_penult"] if read_from == "S_prev" else res["n_final"]
                assert S_i is not None
                S_list.append(S_i)
                n_list.append(n_i)
                betas.append(res["beta"].mean(dim=(0, 1)).detach())
                alphas.append(torch.exp(res["g"]).mean(dim=(0, 1)).detach())
            return {
                "S_banks": S_list,
                "n_banks": n_list,
                "mean_beta": torch.cat(betas, dim=0),
                "mean_alpha": torch.cat(alphas, dim=0),
            }

        # Single fused Bank 0 pass computing BOTH Mid and Long consolidation queries on S_{t-1}^{(0)}!
        q_mid_0 = l2_normalize(self.cons_q_mid(z).view(B, T, 2, self.d_k), dim=-1)
        q_long_0 = l2_normalize(self.cons_q_long_from_short(z).view(B, T, 2, self.d_k), dim=-1)
        q_both_0 = torch.stack([q_mid_0, q_long_0], dim=2)  # (B, T, 2, 2, d_k)

        res0 = self.banks[0](
            z,
            external_q=q_both_0,
            beta_gate=non_noop_gate,
            read_from="S_prev",
            return_penultimate_state=(read_from == "S_prev"),
        )
        raw0_both = res0["raw_out"]  # (B, T, 2, 2, d_v)
        assert raw0_both is not None
        raw_0_to_1 = raw0_both[:, :, 0]
        raw_0_to_2 = raw0_both[:, :, 1]

        m_cons_mid = self.cons_o_mid(raw_0_to_1.reshape(B, T, 2 * self.d_v))
        gate_mid = torch.sigmoid(self.cons_gate_mid(torch.cat([z, m_cons_mid], dim=-1)))
        if non_noop_gate is not None:
            gate_mid = gate_mid * non_noop_gate
        z_mid = z + m_cons_mid

        q_long_1 = l2_normalize(self.cons_q_long_from_mid(z).view(B, T, 2, self.d_k), dim=-1)
        res1 = self.banks[1](
            z_mid,
            external_q=q_long_1,
            beta_gate=gate_mid,
            read_from="S_prev",
            return_penultimate_state=(read_from == "S_prev"),
        )
        raw_1_to_2 = res1["raw_out"]
        assert raw_1_to_2 is not None

        m_cons_long = self.cons_o_long(
            torch.cat([raw_0_to_2, raw_1_to_2], dim=-1).reshape(B, T, 4 * self.d_v)
        )
        gate_long = torch.sigmoid(self.cons_gate_long(torch.cat([z, m_cons_long], dim=-1)))
        if non_noop_gate is not None:
            gate_long = gate_long * non_noop_gate
        z_long = z + m_cons_long

        res2 = self.banks[2](
            z_long,
            beta_gate=gate_long,
            read_from=read_from,
            return_penultimate_state=(read_from == "S_prev"),
        )

        S_list = [
            res0["S_penult"] if read_from == "S_prev" else res0["S_final"],
            res1["S_penult"] if read_from == "S_prev" else res1["S_final"],
            res2["S_penult"] if read_from == "S_prev" else res2["S_final"],
        ]
        n_list = [
            res0["n_penult"] if read_from == "S_prev" else res0["n_final"],
            res1["n_penult"] if read_from == "S_prev" else res1["n_final"],
            res2["n_penult"] if read_from == "S_prev" else res2["n_final"],
        ]
        betas = [
            res0["beta"].mean(dim=(0, 1)).detach(),
            res1["beta"].mean(dim=(0, 1)).detach(),
            res2["beta"].mean(dim=(0, 1)).detach(),
        ]
        alphas = [
            torch.exp(res0["g"]).mean(dim=(0, 1)).detach(),
            torch.exp(res1["g"]).mean(dim=(0, 1)).detach(),
            torch.exp(res2["g"]).mean(dim=(0, 1)).detach(),
        ]
        return {
            "S_banks": S_list,  # type: ignore[dict-item]
            "n_banks": n_list,
            "mean_beta": torch.cat(betas, dim=0),
            "mean_alpha": torch.cat(alphas, dim=0),
            "cons_gate_mid_mean": gate_mid.mean().detach(),
            "cons_gate_long_mean": gate_long.mean().detach(),
        }

    def read_step(
        self,
        h: torch.Tensor,
        mem_state: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        if self.single_readout is not None:
            return self.single_readout(
                h,
                mem_state["S_single"],  # type: ignore[arg-type]
                mem_state["n_single"],  # type: ignore[arg-type]
            )
        assert self.multi_readout is not None
        return self.multi_readout(
            h,
            mem_state["S_banks"],  # type: ignore[arg-type]
            mem_state["n_banks"],  # type: ignore[arg-type]
        )
