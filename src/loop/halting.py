"""Adaptive Halting Head (2-Stage Frozen-Policy PonderNet + Latent Convergence)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHaltingHead(nn.Module):
    """Computes loop stopping probabilities from latent state, memory readout, and coverage."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        max_loops: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_loops = max_loops

        # Input features: h^{(l)}, m^{(l-1)}, ||dh||_2, r_norm (H), coverage (H), loop one-hot (max_loops)
        in_dim = 2 * d_model + 1 + 2 * num_heads + max_loops
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        # Initialize slightly negative so early exploration examines multiple depths
        nn.init.constant_(self.net[-1].bias, -1.0)

    def geometric_prior(
        self, L_max: int, lambda_p: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Truncated geometric prior over depths l = 1..L_max."""
        idx = torch.arange(L_max, device=device, dtype=torch.float32)
        probs = ((1.0 - lambda_p) ** idx) * lambda_p
        probs[-1] = probs[-1] + (1.0 - probs.sum()).clamp(min=0.0)
        probs = probs / probs.sum()
        return probs.to(dtype)

    def forward(
        self,
        h_list: List[torch.Tensor],
        m_list: List[torch.Tensor],
        dh_norms: List[torch.Tensor],
        r_norms: List[torch.Tensor],
        coverages: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute stick-breaking halting probabilities p_tilde^{(l)} for l = 1..L_max.

        Args:
            h_list: length L_max list of h^{(l)} after loop step l-1, each (B, d_model).
            m_list: length L_max list of m^{(l-1)}, each (B, d_model).
            dh_norms: length L_max list of ||h^{(l)} - h^{(l-1)}||_2, each (B, 1).
            r_norms: length L_max list of per-head ||S q^{(l-1)}||_2, each (B, H).
            coverages: length L_max list of per-head q^{(l-1)} . n, each (B, H).
        """
        L_max = len(h_list)
        B = h_list[0].shape[0]
        device = h_list[0].device

        p_stops = []
        for l_idx in range(L_max):
            if l_idx == L_max - 1:
                p_l = torch.ones(B, device=device, dtype=torch.float32)
            else:
                step_oh = F.one_hot(
                    torch.full((B,), min(l_idx, self.max_loops - 1), device=device),
                    num_classes=self.max_loops,
                ).float()
                feat = torch.cat(
                    [
                        h_list[l_idx].float(),
                        m_list[l_idx].float(),
                        dh_norms[l_idx].float(),
                        r_norms[l_idx].float(),
                        coverages[l_idx].float(),
                        step_oh,
                    ],
                    dim=-1,
                )
                p_l = torch.sigmoid(self.net(feat).squeeze(-1))
            p_stops.append(p_l)

        p_stop = torch.stack(p_stops, dim=1)  # (B, L_max)
        # Stick-breaking survival: prod_{j < l} (1 - p_stop[:, j])
        one_minus = (1.0 - p_stop).clamp(min=1e-7, max=1.0)
        survival = torch.cat(
            [
                torch.ones(B, 1, device=device, dtype=torch.float32),
                torch.cumprod(one_minus[:, :-1], dim=1),
            ],
            dim=1,
        )
        p_tilde = p_stop * survival
        p_tilde = p_tilde / p_tilde.sum(dim=1, keepdim=True).clamp(min=1e-7)

        depths = torch.arange(1, L_max + 1, device=device, dtype=torch.float32)
        expected_loops = torch.sum(p_tilde * depths.unsqueeze(0), dim=1)  # (B,)

        return {
            "p_stop": p_stop,
            "p_tilde": p_tilde,
            "expected_loops": expected_loops,
        }

    def compute_stage2_loss(
        self,
        halt_out: Dict[str, torch.Tensor],
        per_depth_losses_detached: torch.Tensor,
        ponder_cost: float = 0.04,
        kl_weight: float = 0.02,
        lambda_prior: float = 0.35,
    ) -> torch.Tensor:
        """2-Stage Frozen-Policy Halting Objective.

        Args:
            halt_out: Output dict from forward().
            per_depth_losses_detached: (B, L_max) detached task cross-entropy at each depth.
            ponder_cost: Linear per-loop step penalty encouraging stopping as soon as solved.
            kl_weight: PonderNet KL regularization weight toward geometric prior.
            lambda_prior: Geometric prior parameter.
        """
        p_tilde = halt_out["p_tilde"]  # (B, L_max)
        L_max = p_tilde.shape[1]
        expected_task_loss = torch.sum(p_tilde * per_depth_losses_detached.detach(), dim=1).mean()
        step_penalty = ponder_cost * halt_out["expected_loops"].mean()

        prior = self.geometric_prior(
            L_max=L_max,
            lambda_p=lambda_prior,
            device=p_tilde.device,
            dtype=p_tilde.dtype,
        )
        kl = torch.sum(
            p_tilde * (torch.log(p_tilde.clamp(min=1e-7)) - torch.log(prior.unsqueeze(0).clamp(min=1e-7))),
            dim=1,
        ).mean()
        return expected_task_loss + step_penalty + kl_weight * kl

    @staticmethod
    def select_halt_step(
        halt_out: Dict[str, torch.Tensor],
        logits_per_depth: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Deterministic early-exit decision at inference: first l where cumulative p_tilde >= threshold."""
        p_tilde = halt_out["p_tilde"]  # (B, L_max)
        B, L_max, C = logits_per_depth.shape
        cdf = torch.cumsum(p_tilde, dim=1)
        hit = cdf >= threshold
        hit[:, -1] = True  # Always halt by L_max
        chosen_idx = torch.argmax(hit.int(), dim=1)  # (B,) in 0..L_max-1
        gather_idx = chosen_idx.view(B, 1, 1).expand(B, 1, C)
        chosen_logits = torch.gather(logits_per_depth, dim=1, index=gather_idx).squeeze(1)
        return {
            "logits": chosen_logits,
            "halt_loops": (chosen_idx + 1).float(),
        }
