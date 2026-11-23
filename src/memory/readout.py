"""Readout modules for single-bank, parallel multi-query, and multi-timescale GDN."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_deltanet import l2_normalize


class MemoryReadout(nn.Module):
    """Dynamic (or static-cached) single-step multi-head readout from GDN state S."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.g_proj = nn.Linear(d_model, num_heads * d_v, bias=True)
        self.o_proj = nn.Linear(num_heads * d_v, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        S: torch.Tensor,
        n_state: Optional[torch.Tensor] = None,
        fixed_q: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read from S: (B, H, d_v, d_k) given latent state h: (B, d_model)."""
        B = h.shape[0]
        if fixed_q is not None:
            q = fixed_q
        else:
            q = l2_normalize(self.q_proj(h).view(B, self.num_heads, self.d_k), dim=-1)

        raw = torch.einsum("bhvk,bhk->bhv", S.float(), q.float()).to(h.dtype)
        gate = torch.sigmoid(self.g_proj(h)).view(B, self.num_heads, self.d_v)
        m = self.o_proj((gate * raw).reshape(B, self.num_heads * self.d_v))

        r_norm = torch.linalg.vector_norm(raw.float(), ord=2, dim=-1)  # (B, H)
        if n_state is not None:
            coverage = torch.sum(q.float() * n_state.float(), dim=-1)  # (B, H)
        else:
            coverage = torch.zeros_like(r_norm)

        return {
            "m": m,
            "q": q,
            "raw": raw,
            "r_norm": r_norm,
            "coverage": coverage,
        }


class ParallelMultiQueryReadout(nn.Module):
    """Issues K queries in parallel at l=0 (L=1) without sequential conditioning."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        num_parallel_queries: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.K = num_parallel_queries

        # Bottleneck query/output projections so total parameter budget matches DeltaLoop
        self.q_proj = nn.Linear(d_model, self.K * num_heads * d_k, bias=False)
        self.g_proj = nn.Linear(d_model, self.K * num_heads * d_v, bias=True)
        self.o_proj = nn.Linear(self.K * num_heads * d_v, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        S: torch.Tensor,
        n_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        q = l2_normalize(
            self.q_proj(h).view(B, self.K, self.num_heads, self.d_k), dim=-1
        )
        # S is (B, H, V, D_k), q is (B, K, H, D_k) -> raw is (B, K, H, V)
        raw = torch.einsum("bhvd,bkhd->bkhv", S.float(), q.float()).to(h.dtype)
        gate = torch.sigmoid(self.g_proj(h)).view(B, self.K, self.num_heads, self.d_v)
        m = self.o_proj((gate * raw).reshape(B, self.K * self.num_heads * self.d_v))
        r_norm = torch.linalg.vector_norm(raw.float(), ord=2, dim=-1).mean(dim=1)
        if n_state is not None:
            coverage = torch.sum(q.float() * n_state.float().unsqueeze(1), dim=-1).mean(dim=1)
        else:
            coverage = torch.zeros_like(r_norm)
        return {
            "m": m,
            "q": q[:, 0],
            "q_all": q,
            "raw": raw,
            "r_norm": r_norm,
            "coverage": coverage,
        }


class ScaleRoutedReadout(nn.Module):
    """Multi-bank readout with optional query-conditioned Softmax scale routing."""

    def __init__(
        self,
        d_model: int,
        num_banks: int = 3,
        heads_per_bank: int = 2,
        d_k: int = 16,
        d_v: int = 16,
        use_routing: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        self.heads_per_bank = heads_per_bank
        self.d_k = d_k
        self.d_v = d_v
        self.use_routing = use_routing

        self.bank_readouts = nn.ModuleList(
            [
                MemoryReadout(
                    d_model=d_model,
                    num_heads=heads_per_bank,
                    d_k=d_k,
                    d_v=d_v,
                )
                for _ in range(num_banks)
            ]
        )
        if use_routing:
            self.router = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, num_banks),
            )
        else:
            self.router = None

    def forward(
        self,
        h: torch.Tensor,
        S_banks: List[torch.Tensor],
        n_banks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert len(S_banks) == self.num_banks
        B = h.shape[0]
        m_list = []
        r_norms = []
        coverages = []
        qs = []
        for i in range(self.num_banks):
            n_i = n_banks[i] if n_banks is not None else None
            ro = self.bank_readouts[i](h, S_banks[i], n_i)
            m_list.append(ro["m"])
            r_norms.append(ro["r_norm"])
            coverages.append(ro["coverage"])
            qs.append(ro["q"])

        m_stack = torch.stack(m_list, dim=1)  # (B, num_banks, d_model)
        if self.use_routing and self.router is not None:
            pi = F.softmax(self.router(h), dim=-1)  # (B, num_banks)
        else:
            pi = torch.full(
                (B, self.num_banks),
                1.0 / self.num_banks,
                device=h.device,
                dtype=h.dtype,
            )

        # Multiply by num_banks when unrouted vs routed? Keep convex sum scaled consistently:
        # Both routed and uniform convex combinations sum to 1 across banks.
        m = torch.sum(pi.unsqueeze(-1) * m_stack, dim=1)
        return {
            "m": m,
            "pi": pi,
            "q_banks": torch.stack(qs, dim=1),
            "r_norm": torch.cat(r_norms, dim=-1),
            "coverage": torch.cat(coverages, dim=-1),
        }
