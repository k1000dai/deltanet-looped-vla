"""Exact Gated DeltaNet cell and chunkwise WY parallel forward in PyTorch.

Implements:
- Scalar per-head log-decay parameterized in fp32: g_t = log(alpha_t) <= 0
  (avoiding bf16 rounding near alpha = 1.0).
- Configurable beta range:
  * beta_max = 1.0: standard Gated DeltaNet, eigenvalues in (0, 1).
  * beta_max = 2.0: negative-eigenvalue Householder Gated DeltaNet
    (Grazzi et al., ICLR 2025), eigenvalues in (-1, 1).
- Key L2-normalization in fp32 + fused v=1 Delta-Rule Key Occupancy state:
  s_occ_t = alpha_t * (I - beta_t k_t k_t^T) s_occ_{t-1} + beta_t k_t
  (consistent with both erase-then-write and Householder reflections!).
- Readout from S_{t-1} (strictly causal, preventing trivial self-retrieval of
  current step t) vs S_t (inclusive ablation).
- Exact equivalence between step-by-step recurrent execution and chunkwise
  parallel WY representation (unit lower-triangular solve).
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize tensor along `dim` strictly in fp32 and cast back."""
    x_f32 = x.float()
    norm = torch.sqrt(torch.sum(x_f32 * x_f32, dim=dim, keepdim=True) + eps * eps)
    return (x_f32 / norm).to(x.dtype)


def chunk_gated_delta_rule(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_norm_state: Optional[torch.Tensor] = None,
    read_from: Literal["S_prev", "S_curr"] = "S_prev",
    chunk_size: int = 64,
    track_norm_state: bool = True,
    return_penultimate_state: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    """Chunkwise parallel Gated Delta Rule via WY representation in fp32.

    Supports both single query stream q: (B, T, H, d_k) -> out: (B, T, H, d_v)
    and multi-query stream q: (B, T, M, H, d_k) -> out: (B, T, M, H, d_v).
    Tracks fused v=1 Delta Occupancy vector `n_t` (solving the exact same Gated Delta
    recurrence with scalar target 1.0 as an extra virtual row of S!).
    """
    B, T, H, d_k = k.shape
    d_v = v.shape[-1]
    orig_dtype = v.dtype

    if return_penultimate_state and T >= 2:
        out_prefix, S_penult, n_penult, _ = chunk_gated_delta_rule(
            k=k[:, :-1],
            v=v[:, :-1],
            g=g[:, :-1],
            beta=beta[:, :-1],
            q=q[:, :-1] if q is not None else None,
            initial_state=initial_state,
            initial_norm_state=initial_norm_state,
            read_from=read_from,
            chunk_size=chunk_size,
            track_norm_state=track_norm_state,
            return_penultimate_state=False,
        )
        k_last = k[:, -1].float()
        v_last = v[:, -1].float()
        alpha_last = torch.exp(g[:, -1].float())
        beta_last = beta[:, -1].float()

        S_prev_f = S_penult.float()
        Sk = torch.einsum("bhvk,bhk->bhv", S_prev_f, k_last)
        p_last = beta_last.unsqueeze(-1) * (v_last - alpha_last.unsqueeze(-1) * Sk)
        S_last = alpha_last[:, :, None, None] * S_prev_f + torch.einsum(
            "bhv,bhk->bhvk", p_last, k_last
        )
        n_last = None
        if track_norm_state and n_penult is not None:
            n_prev_f = n_penult.float()
            nk = torch.sum(n_prev_f * k_last, dim=-1)  # (B, H)
            p_occ = beta_last * (1.0 - alpha_last * nk)
            n_last = alpha_last.unsqueeze(-1) * n_prev_f + p_occ.unsqueeze(-1) * k_last

        out_all = None
        if q is not None and out_prefix is not None:
            q_last = q[:, -1].float()
            target_S = S_prev_f if read_from == "S_prev" else S_last
            if q_last.dim() == 3:
                o_last = torch.einsum("bhvk,bhk->bhv", target_S, q_last).to(orig_dtype)
            else:
                o_last = torch.einsum("bhvk,bmhk->bmhv", target_S, q_last).to(orig_dtype)
            out_all = torch.cat([out_prefix, o_last.unsqueeze(1)], dim=1)

        return (
            out_all,
            S_last.to(orig_dtype),
            n_last.to(orig_dtype) if n_last is not None else None,
            (S_penult, n_penult),
        )

    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()

    # Fuse v=1 occupancy channel as (d_v + 1)-th value coordinate so it gets exact
    # Delta erase/Householder treatment inside the single triangular solve!
    if track_norm_state:
        ones_v = torch.ones(B, T, H, 1, device=k.device, dtype=torch.float32)
        v_aug = torch.cat([v_f, ones_v], dim=-1)  # (B, T, H, d_v + 1)
    else:
        v_aug = v_f
    d_v_aug = v_aug.shape[-1]

    q_multi = False
    M_q = 1
    q_f = None
    if q is not None:
        if q.dim() == 5:
            q_multi = True
            M_q = q.shape[2]
            q_f = q.float()  # (B, T, M_q, H, d_k)
        else:
            q_f = q.float().unsqueeze(2)  # (B, T, 1, H, d_k)

    C = min(chunk_size, T) if T > 0 else 1
    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        k_f = F.pad(k_f, (0, 0, 0, 0, 0, pad_len))
        v_aug = F.pad(v_aug, (0, 0, 0, 0, 0, pad_len))
        g_f = F.pad(g_f, (0, 0, 0, pad_len), value=0.0)
        beta_f = F.pad(beta_f, (0, 0, 0, pad_len), value=0.0)
        if q_f is not None:
            q_f = F.pad(q_f, (0, 0, 0, 0, 0, 0, 0, pad_len))

    T_pad = k_f.shape[1]
    num_chunks = T_pad // C

    k_c = k_f.view(B, num_chunks, C, H, d_k).permute(0, 3, 1, 2, 4).contiguous()
    v_c = v_aug.view(B, num_chunks, C, H, d_v_aug).permute(0, 3, 1, 2, 4).contiguous()
    g_c = g_f.view(B, num_chunks, C, H).permute(0, 3, 1, 2).contiguous()
    beta_c = beta_f.view(B, num_chunks, C, H).permute(0, 3, 1, 2).contiguous()
    q_c = (
        q_f.view(B, num_chunks, C, M_q, H, d_k).permute(0, 4, 1, 3, 2, 5).contiguous()
        if q_f is not None
        else None
    )  # (B, H, N_c, M_q, C, d_k)

    gamma = torch.cumsum(g_c, dim=-1)  # (B, H, N_c, C)
    idx = torch.arange(C, device=k.device)
    strict_lower = idx[:, None] > idx[None, :]
    incl_lower = idx[:, None] >= idx[None, :]

    diff_strict = (gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).masked_fill(~strict_lower, 0.0)
    decay_strict = torch.exp(diff_strict) * strict_lower.float()

    kk = torch.matmul(k_c, k_c.transpose(-1, -2))
    L = beta_c.unsqueeze(-1) * decay_strict * kk
    eye = torch.eye(C, device=k.device, dtype=torch.float32)
    A = eye + L

    rhs_u = beta_c.unsqueeze(-1) * v_c  # (B, H, N_c, C, d_v_aug)
    exp_gamma = torch.exp(gamma)
    rhs_w = (beta_c * exp_gamma).unsqueeze(-1) * k_c
    rhs = torch.cat([rhs_u, rhs_w], dim=-1)
    uw = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    U = uw[..., :d_v_aug]
    W = uw[..., d_v_aug:]

    if q_c is not None:
        # q_c: (B, H, N_c, M_q, C, d_k), k_c.unsqueeze(3): (B, H, N_c, 1, C, d_k)
        qk = torch.matmul(q_c, k_c.unsqueeze(3).transpose(-1, -2))  # (B, H, N_c, M_q, C, C)
        if read_from == "S_curr":
            diff_incl = (gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).masked_fill(~incl_lower, 0.0)
            M_read = (torch.exp(diff_incl) * incl_lower.float()).unsqueeze(3) * qk
            q_scale = exp_gamma.unsqueeze(3)  # (B, H, N_c, 1, C)
        else:
            gamma_prev = F.pad(gamma[..., :-1], (1, 0), value=0.0)
            diff_prev = (gamma_prev.unsqueeze(-1) - gamma.unsqueeze(-2)).masked_fill(
                ~strict_lower, 0.0
            )
            M_read = (torch.exp(diff_prev) * strict_lower.float()).unsqueeze(3) * qk
            q_scale = torch.exp(gamma_prev).unsqueeze(3)

    S0_val = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(B, H, d_v, d_k, device=k.device, dtype=torch.float32)
    )
    if track_norm_state:
        n0_val = (
            initial_norm_state.float()
            if initial_norm_state is not None
            else torch.zeros(B, H, d_k, device=k.device, dtype=torch.float32)
        )
        S_aug = torch.cat([S0_val, n0_val.unsqueeze(-2)], dim=-2)  # (B, H, d_v + 1, d_k)
    else:
        S_aug = S0_val

    out_chunks = [] if q_c is not None else None

    for c_idx in range(num_chunks):
        U_i = U[:, :, c_idx]  # (B, H, C, d_v_aug)
        W_i = W[:, :, c_idx]  # (B, H, C, d_k)
        k_i = k_c[:, :, c_idx]  # (B, H, C, d_k)
        gamma_i = gamma[:, :, c_idx]
        gamma_end = gamma_i[..., -1:]

        P = U_i - torch.matmul(W_i, S_aug.transpose(-1, -2))  # (B, H, C, d_v_aug)

        if q_c is not None and out_chunks is not None:
            P_val = P[..., :d_v]  # (B, H, C, d_v)
            S_val = S_aug[:, :, :d_v, :]  # (B, H, d_v, d_k)
            q_i = q_c[:, :, c_idx]  # (B, H, M_q, C, d_k)
            M_i = M_read[:, :, c_idx]  # (B, H, M_q, C, C)
            qs_i = q_scale[:, :, c_idx].unsqueeze(-1)  # (B, H, M_q, C, 1)
            o_i = qs_i * torch.matmul(q_i, S_val.unsqueeze(2).transpose(-1, -2)) + torch.matmul(
                M_i, P_val.unsqueeze(2)
            )
            out_chunks.append(o_i)  # (B, H, M_q, C, d_v)

        decay_to_end = torch.exp(gamma_end - gamma_i)
        S_aug = torch.exp(gamma_end).unsqueeze(-1) * S_aug + torch.matmul(
            P.transpose(-1, -2), decay_to_end.unsqueeze(-1) * k_i
        )

    S_final = S_aug[:, :, :d_v, :]
    n_final = S_aug[:, :, d_v, :] if track_norm_state else None

    out_tensor = None
    if out_chunks is not None:
        # Stack along chunk dim -> (B, H, N_c, M_q, C, d_v) -> (B, N_c, C, M_q, H, d_v)
        out_full = (
            torch.stack(out_chunks, dim=2)
            .permute(0, 2, 4, 3, 1, 5)
            .reshape(B, T_pad, M_q, H, d_v)[:, :T]
        )
        if not q_multi:
            out_full = out_full.squeeze(2)
        out_tensor = out_full.contiguous().to(orig_dtype)

    penult_pack = None
    if return_penultimate_state and T == 1:
        penult_pack = (
            S0_val.to(orig_dtype),
            n0_val.to(orig_dtype) if track_norm_state else None,
        )

    return (
        out_tensor,
        S_final.to(orig_dtype),
        n_final.to(orig_dtype) if n_final is not None else None,
        penult_pack,
    )


def recurrent_gated_delta_rule(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_norm_state: Optional[torch.Tensor] = None,
    read_from: Literal["S_prev", "S_curr"] = "S_prev",
    track_norm_state: bool = True,
    return_penultimate_state: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    """Reference step-by-step fp32 recurrent Gated Delta Rule (with v=1 Delta Occupancy)."""
    B, T, H, d_k = k.shape
    d_v = v.shape[-1]
    orig_dtype = v.dtype

    S = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(B, H, d_v, d_k, device=k.device, dtype=torch.float32)
    )
    n_state = (
        (
            initial_norm_state.float()
            if initial_norm_state is not None
            else torch.zeros(B, H, d_k, device=k.device, dtype=torch.float32)
        )
        if track_norm_state
        else None
    )

    outputs = [] if q is not None else None
    S_penult: Optional[torch.Tensor] = None
    n_penult: Optional[torch.Tensor] = None

    for t in range(T):
        if return_penultimate_state and t == T - 1:
            S_penult = S.clone().to(orig_dtype)
            n_penult = n_state.clone().to(orig_dtype) if n_state is not None else None

        k_t = k[:, t].float()
        v_t = v[:, t].float()
        alpha_t = torch.exp(g[:, t].float())
        beta_t = beta[:, t].float()

        if q is not None and outputs is not None and read_from == "S_prev":
            q_t = q[:, t].float()
            if q_t.dim() == 3:
                outputs.append(torch.einsum("bhvk,bhk->bhv", S, q_t))
            else:
                outputs.append(torch.einsum("bhvk,bmhk->bmhv", S, q_t))

        Sk = torch.einsum("bhvk,bhk->bhv", S, k_t)
        S = alpha_t[:, :, None, None] * S + torch.einsum(
            "bhv,bhk->bhvk",
            beta_t.unsqueeze(-1) * (v_t - alpha_t.unsqueeze(-1) * Sk),
            k_t,
        )
        if track_norm_state and n_state is not None:
            nk = torch.sum(n_state * k_t, dim=-1)
            n_state = (
                alpha_t.unsqueeze(-1) * n_state
                + (beta_t * (1.0 - alpha_t * nk)).unsqueeze(-1) * k_t
            )

        if q is not None and outputs is not None and read_from == "S_curr":
            q_t = q[:, t].float()
            if q_t.dim() == 3:
                outputs.append(torch.einsum("bhvk,bhk->bhv", S, q_t))
            else:
                outputs.append(torch.einsum("bhvk,bmhk->bmhv", S, q_t))

    out_tensor = (
        torch.stack(outputs, dim=1).to(orig_dtype) if outputs is not None else None
    )
    penult_pack = (S_penult, n_penult) if return_penultimate_state else None
    return (
        out_tensor,
        S.to(orig_dtype),
        n_state.to(orig_dtype) if n_state is not None else None,
        penult_pack,
    )


class GatedDeltaNetLayer(nn.Module):
    """Reusable Multi-Head Gated DeltaNet Temporal Memory Layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        beta_max: float = 1.0,
        decay_mode: Literal["learnable", "fixed", "band"] = "learnable",
        min_halflife: float = 16.0,
        max_halflife: float = 512.0,
        alpha_bounds: Optional[Tuple[float, float]] = None,
        fixed_alphas: Optional[Tuple[float, ...]] = None,
        use_seq_readout: bool = False,
        chunk_size: int = 64,
        track_norm_state: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.beta_max = float(beta_max)
        self.decay_mode = decay_mode
        self.use_seq_readout = use_seq_readout
        self.chunk_size = chunk_size
        self.track_norm_state = track_norm_state

        self.k_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)

        if use_seq_readout:
            self.q_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
            self.g_out_proj = nn.Linear(d_model, num_heads * d_v, bias=True)
            self.o_proj = nn.Linear(num_heads * d_v, d_model, bias=False)

        if decay_mode == "fixed":
            if fixed_alphas is not None:
                assert len(fixed_alphas) == num_heads
                alphas = torch.tensor(fixed_alphas, dtype=torch.float32)
            else:
                hl = torch.logspace(
                    math.log10(min_halflife),
                    math.log10(max_halflife),
                    steps=num_heads,
                    dtype=torch.float32,
                )
                alphas = torch.pow(2.0, -1.0 / hl)
            g_fixed = torch.log(alphas.clamp(min=1e-6, max=1.0 - 1e-7))
            self.register_buffer("g_fixed", g_fixed)
        elif decay_mode == "band":
            assert alpha_bounds is not None
            a_min, a_max = alpha_bounds
            self.register_buffer("g_min", torch.tensor(math.log(a_min), dtype=torch.float32))
            self.register_buffer("g_max", torch.tensor(math.log(a_max), dtype=torch.float32))
            self.a_proj = nn.Linear(d_model, num_heads, bias=True)
            nn.init.zeros_(self.a_proj.weight)
            nn.init.constant_(self.a_proj.bias, 0.5)
        elif decay_mode == "learnable":
            self.a_proj = nn.Linear(d_model, num_heads, bias=False)
            hl = torch.logspace(
                math.log10(min_halflife),
                math.log10(max_halflife),
                steps=num_heads,
                dtype=torch.float32,
            )
            target_neg_g = math.log(2.0) / hl
            dt_bias = torch.log(torch.expm1(target_neg_g).clamp(min=1e-6))
            self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
            self.dt_bias = nn.Parameter(dt_bias)
        else:
            raise ValueError(f"Unknown decay_mode: {decay_mode}")

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 0.0)

    @property
    def state_numel(self) -> int:
        return self.num_heads * self.d_k * self.d_v

    def compute_g_and_beta(
        self,
        z: torch.Tensor,
        beta_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = z.shape
        if self.decay_mode == "fixed":
            g = self.g_fixed.view(1, 1, self.num_heads).expand(B, T, self.num_heads).contiguous()
        elif self.decay_mode == "band":
            gate = torch.sigmoid(self.a_proj(z).float())
            g = self.g_min + gate * (self.g_max - self.g_min)
        else:
            raw_a = self.a_proj(z).float() + self.dt_bias.float().view(1, 1, -1)
            g = -torch.exp(self.A_log.float()).view(1, 1, -1) * F.softplus(raw_a)
            g = g.clamp(min=-15.0, max=-1e-7)

        beta = self.beta_max * torch.sigmoid(self.beta_proj(z).float())
        if beta_gate is not None:
            beta = beta * beta_gate.float()
        return g, beta

    def step(
        self,
        z_t: torch.Tensor,
        S_prev: torch.Tensor,
        n_prev: Optional[torch.Tensor] = None,
        beta_gate_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        B = z_t.shape[0]
        z_seq = z_t.unsqueeze(1)
        bg_seq = beta_gate_t.unsqueeze(1) if beta_gate_t is not None else None
        k_t = l2_normalize(self.k_proj(z_t).view(B, self.num_heads, self.d_k), dim=-1)
        v_t = self.v_proj(z_t).view(B, self.num_heads, self.d_v)
        g_seq, beta_seq = self.compute_g_and_beta(z_seq, bg_seq)
        alpha_t = torch.exp(g_seq[:, 0])
        beta_t = beta_seq[:, 0]

        S_f = S_prev.float()
        k_f = k_t.float()
        v_f = v_t.float()
        Sk = torch.einsum("bhvk,bhk->bhv", S_f, k_f)
        S_next = alpha_t[:, :, None, None] * S_f + torch.einsum(
            "bhv,bhk->bhvk",
            beta_t.unsqueeze(-1) * (v_f - alpha_t.unsqueeze(-1) * Sk),
            k_f,
        )
        n_next = None
        if self.track_norm_state and n_prev is not None:
            nk = torch.sum(n_prev.float() * k_f, dim=-1)
            n_next = (
                alpha_t.unsqueeze(-1) * n_prev.float()
                + (beta_t * (1.0 - alpha_t * nk)).unsqueeze(-1) * k_f
            )

        return (
            S_next.to(z_t.dtype),
            n_next.to(z_t.dtype) if n_next is not None else None,
            {"k": k_t, "alpha": alpha_t, "beta": beta_t},
        )

    def forward(
        self,
        z: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        initial_norm_state: Optional[torch.Tensor] = None,
        external_q: Optional[torch.Tensor] = None,
        beta_gate: Optional[torch.Tensor] = None,
        read_from: Literal["S_prev", "S_curr"] = "S_prev",
        return_penultimate_state: bool = True,
        return_keys: bool = False,
        mode: Literal["chunk", "recurrent"] = "chunk",
    ) -> Dict[str, Optional[torch.Tensor]]:
        B, T, _ = z.shape
        k = l2_normalize(self.k_proj(z).view(B, T, self.num_heads, self.d_k), dim=-1)
        v = self.v_proj(z).view(B, T, self.num_heads, self.d_v)
        g, beta = self.compute_g_and_beta(z, beta_gate=beta_gate)

        q = external_q
        if q is None and self.use_seq_readout:
            q = l2_normalize(self.q_proj(z).view(B, T, self.num_heads, self.d_k), dim=-1)

        if mode == "chunk":
            raw_out, S_final, n_final, penult = chunk_gated_delta_rule(
                k=k,
                v=v,
                g=g,
                beta=beta,
                q=q,
                initial_state=initial_state,
                initial_norm_state=initial_norm_state,
                read_from=read_from,
                chunk_size=self.chunk_size,
                track_norm_state=self.track_norm_state,
                return_penultimate_state=return_penultimate_state,
            )
        else:
            raw_out, S_final, n_final, penult = recurrent_gated_delta_rule(
                k=k,
                v=v,
                g=g,
                beta=beta,
                q=q,
                initial_state=initial_state,
                initial_norm_state=initial_norm_state,
                read_from=read_from,
                track_norm_state=self.track_norm_state,
                return_penultimate_state=return_penultimate_state,
            )

        seq_out = raw_out
        if raw_out is not None and external_q is None and self.use_seq_readout:
            gate = torch.sigmoid(self.g_out_proj(z)).view(B, T, self.num_heads, self.d_v)
            seq_out = self.o_proj((gate * raw_out).reshape(B, T, self.num_heads * self.d_v))

        S_penult, n_penult = penult if penult is not None else (None, None)
        return {
            "seq_out": seq_out,
            "raw_out": raw_out,
            "S_final": S_final,
            "n_final": n_final,
            "S_penult": S_penult,
            "n_penult": n_penult,
            "keys": k if return_keys else None,
            "g": g,
            "beta": beta,
        }
