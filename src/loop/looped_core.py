"""Prelude, Weight-Tied Recurrent Core, Coda, and Matched Baseline Architectures."""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory.gated_deltanet import GatedDeltaNetLayer, chunk_gated_delta_rule, l2_normalize
from src.memory.multi_timescale_gdn import MultiTimescaleGDN, Variant1B
from src.memory.readout import MemoryReadout, ParallelMultiQueryReadout
from .halting import AdaptiveHaltingHead


class SwiGLU(nn.Module):
    """Compact SwiGLU MLP block with exact hidden dimension control."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class SinusoidalStepEmbedding(nn.Module):
    """Parameter-free sinusoidal position/time encoding + small-init learned projection."""

    def __init__(self, d_model: int, enabled: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.enabled = enabled
        half = d_model // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        if enabled:
            self.proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        else:
            self.proj = None

    def forward(self, T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.enabled or self.proj is None:
            return None
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(-1)
        angles = pos * self.freqs.unsqueeze(0)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(pe.to(dtype))


class Prelude(nn.Module):
    """Maps query-step representation z_t -> initial loop latent h_t^{(0)}."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return z_t + self.mlp(self.norm(z_t))


class RecurrentCore(nn.Module):
    """Depth-recurrent reasoning block F_theta(h^{(l)}, m^{(l)}, z_t, e_l)."""

    def __init__(self, d_model: int, d_ff: int, max_loops: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_loops = max_loops

        self.loop_emb = nn.Embedding(max_loops, d_model)
        nn.init.normal_(self.loop_emb.weight, mean=0.0, std=0.02)

        self.h_norm = nn.RMSNorm(d_model)
        self.m_norm = nn.RMSNorm(d_model)
        self.z_inj = nn.Linear(d_model, d_model, bias=False)
        self.fuse_proj = nn.Linear(3 * d_model, d_model, bias=True)
        self.update_gate = nn.Linear(3 * d_model, d_model, bias=True)
        nn.init.constant_(self.update_gate.bias, 0.5)

        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def get_loop_embedding(
        self, l_idx: int, B: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        clamped = min(l_idx, self.max_loops - 1)
        idx = torch.full((B,), clamped, device=device, dtype=torch.long)
        return self.loop_emb(idx).to(dtype)

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        z_t: torch.Tensor,
        e_l: torch.Tensor,
    ) -> torch.Tensor:
        h_in = self.h_norm(h + e_l)
        m_in = self.m_norm(m)
        z_in = self.z_inj(z_t)
        concat = torch.cat([h_in, m_in, z_in], dim=-1)
        cand = self.fuse_proj(concat)
        gate = torch.sigmoid(self.update_gate(concat))
        h_mid = h + gate * cand
        h_next = h_mid + self.ffn(self.ffn_norm(h_mid))
        return h_next


class Coda(nn.Module):
    """Maps final loop latent h_t^{(L)} -> output logits."""

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))


Arch1A = Literal[
    "single_gdn_b1",
    "single_gdn_b2",
    "two_layer_gdn_b2",
    "write_loop_gdn_b2",
    "static_query_loop_b2",
    "parallel_multiquery_b2",
    "untied_deep_readout_b2",
    "deltaloop_b1",
    "deltaloop_b2",
    "deltaloop_b2_scurr",
    "softmax_transformer",
]


class StructuredTokenEncoder(nn.Module):
    """Encodes discrete structured event tuples (op, arg1, arg2) into R^{d_model}."""

    def __init__(
        self,
        num_ops: int,
        vocab_size: int,
        d_model: int,
        use_time_emb: bool = True,
    ) -> None:
        super().__init__()
        self.op_emb = nn.Embedding(num_ops, d_model)
        self.sym_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.role1_proj = nn.Linear(d_model, d_model, bias=False)
        self.role2_proj = nn.Linear(d_model, d_model, bias=False)
        self.time_emb = SinusoidalStepEmbedding(d_model, enabled=use_time_emb)
        self.norm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, _ = tokens.shape
        e_op = self.op_emb(tokens[..., 0])
        e_a1 = self.role1_proj(self.sym_emb(tokens[..., 1]))
        e_a2 = self.role2_proj(self.sym_emb(tokens[..., 2]))
        base = e_op + e_a1 + e_a2
        pe = self.time_emb(T, device=tokens.device, dtype=e_op.dtype)
        if pe is not None:
            base = base + pe.unsqueeze(0)
        return self.norm(base + self.mlp(self.norm(base)))


class CausalTransformerBlock(nn.Module):
    """Standard Pre-Norm Causal Softmax Self-Attention Block (skyline baseline)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(attn_out.transpose(1, 2).reshape(B, T, D))
        x = x + self.ffn(self.norm2(x))
        return x


class WriteLoopGDNCell(nn.Module):
    """DeltaProduct + LT2-style Chunkwise-Parallel Write-Side Loop.

    Combines:
    - LT2 weight-tied depth looping of the temporal GDN mixer (read-before-write across K_loop=2 passes)
    - DeltaProduct multi-Householder expansion (n_h=2 rank-1 reflections per token with frame-wise
      decay g_{t,0} = log(alpha_t) and g_{t,1} = 0.0, totaling 4 Householder write steps/token)
    while remaining 100% chunkwise-parallel on GPU and keeping exact 4096 state elements!
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        beta_max: float = 2.0,
        num_write_loops: int = 4,
        d_ff: int = 96,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_householder = 2
        self.n_mixer_loops = max(1, num_write_loops // 2)  # 2 * 2 = 4 write micro-steps/token

        self.gdn = GatedDeltaNetLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            beta_max=beta_max,
            decay_mode="learnable",
            min_halflife=16.0,
            max_halflife=512.0,
            use_seq_readout=True,
        )
        self.loop_emb = nn.Embedding(self.n_mixer_loops, d_model)
        self.hh_emb = nn.Embedding(self.n_householder, d_model)
        nn.init.normal_(self.loop_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.hh_emb.weight, mean=0.0, std=0.02)

        self.m_norm = nn.RMSNorm(d_model)
        self.norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    @property
    def state_numel(self) -> int:
        return self.gdn.state_numel

    def _run_deltaproduct_pass(
        self,
        u: torch.Tensor,
        non_noop: torch.Tensor,
        need_seq_out: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Run n_h=2 Householder updates per token via exact chunkwise WY."""
        B, T, D = u.shape
        nh = self.n_householder
        hh_e = self.hh_emb.weight.to(u.dtype).view(1, 1, nh, D)
        u_exp = (u.unsqueeze(2) + hh_e).reshape(B, T * nh, D)  # (B, T*nh, D)
        gate_exp = non_noop.unsqueeze(2).expand(B, T, nh, 1).reshape(B, T * nh, 1)

        k = l2_normalize(
            self.gdn.k_proj(u_exp).view(B, T * nh, self.num_heads, self.d_k), dim=-1
        )
        v = self.gdn.v_proj(u_exp).view(B, T * nh, self.num_heads, self.d_v)
        g_raw, beta = self.gdn.compute_g_and_beta(u_exp, beta_gate=gate_exp)

        # Frame-wise decay (A1): decay active ONLY on slot s=0, zero on s > 0!
        g_slots = g_raw.view(B, T, nh, self.num_heads).clone()
        g_slots[:, :, 1:, :] = 0.0
        g_flat = g_slots.reshape(B, T * nh, self.num_heads)

        q = None
        if need_seq_out:
            q = l2_normalize(
                self.gdn.q_proj(u_exp).view(B, T * nh, self.num_heads, self.d_k), dim=-1
            )

        raw_out, S_final, n_final, _ = chunk_gated_delta_rule(
            k=k,
            v=v,
            g=g_flat,
            beta=beta,
            q=q,
            read_from="S_prev",
            chunk_size=44,
            track_norm_state=True,
            return_penultimate_state=False,
        )
        seq_out = None
        if need_seq_out and raw_out is not None:
            # Take readout at first sub-slot of each token (strictly S_{t-1} before token t's writes!)
            raw_tok = raw_out.view(B, T, nh, self.num_heads, self.d_v)[:, :, 0]
            gate_o = torch.sigmoid(self.gdn.g_out_proj(u)).view(B, T, self.num_heads, self.d_v)
            seq_out = self.gdn.o_proj((gate_o * raw_tok).reshape(B, T, self.num_heads * self.d_v))
        return seq_out, S_final, n_final

    def forward_prefix(
        self, z_prefix: torch.Tensor, non_noop_prefix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u = z_prefix
        S_mem: Optional[torch.Tensor] = None
        n_mem: Optional[torch.Tensor] = None
        for m_idx in range(self.n_mixer_loops):
            is_last = m_idx == self.n_mixer_loops - 1
            e_m = self.loop_emb.weight[m_idx].to(u.dtype).view(1, 1, -1)
            seq_out, S_f, n_f = self._run_deltaproduct_pass(
                u + e_m, non_noop_prefix, need_seq_out=(not is_last)
            )
            if not is_last and seq_out is not None:
                u_mid = u + self.m_norm(seq_out)
                u = u_mid + self.ffn(self.norm(u_mid))
            else:
                S_mem, n_mem = S_f, n_f
        assert S_mem is not None and n_mem is not None
        return S_mem, n_mem


class DeltaLoopSequenceModel(nn.Module):
    """Unified Model Suite for Tasks 1A-Online and 1A-Retro with Matched Budgets."""

    def __init__(
        self,
        arch: Arch1A,
        num_ops: int,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        max_loops: int = 6,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.max_loops = max_loops

        self.encoder = StructuredTokenEncoder(
            num_ops=num_ops,
            vocab_size=vocab_size,
            d_model=d_model,
            use_time_emb=True,
        )

        if arch == "softmax_transformer":
            # 4-layer causal Softmax Transformer skyline (~450K params)
            self.blocks = nn.ModuleList(
                [CausalTransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=96) for _ in range(4)]
            )
            self.coda = Coda(d_model, num_classes)
            self.halting_head = None

        elif arch == "two_layer_gdn_b2":
            # Strictly matched state = 2 * (4 * 32 * 16) = 4096 floats while keeping all 4 heads!
            self.gdn1 = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v // 2,
                beta_max=2.0,
                decay_mode="learnable",
                use_seq_readout=True,
            )
            self.mid_norm = nn.RMSNorm(d_model)
            self.mid_ffn = SwiGLU(d_model, d_ff=112)
            self.gdn2 = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v // 2,
                beta_max=2.0,
                decay_mode="learnable",
                use_seq_readout=False,
            )
            self.readout1 = MemoryReadout(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v // 2)
            self.readout2 = MemoryReadout(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v // 2)
            self.prelude = Prelude(d_model, d_ff=128)
            self.core = RecurrentCore(d_model, d_ff=128, max_loops=2)
            self.coda = Coda(d_model, num_classes)
            self.halting_head = None

        elif arch == "write_loop_gdn_b2":
            # Strictly matched state = 4 * 32 * 32 = 4096 floats!
            self.write_loop = WriteLoopGDNCell(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                beta_max=2.0,
                num_write_loops=4,
                d_ff=96,
            )
            self.readout = MemoryReadout(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
            self.prelude = Prelude(d_model, d_ff=112)
            self.core = RecurrentCore(d_model, d_ff=128, max_loops=2)
            self.coda = Coda(d_model, num_classes)
            self.halting_head = None

        elif arch == "parallel_multiquery_b2":
            # Matched total query count K_par = max_loops (6 * 4 = 24 head queries at l=0, L=1)!
            self.gdn = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                beta_max=2.0,
                decay_mode="learnable",
            )
            self.pmq_readout = ParallelMultiQueryReadout(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                num_parallel_queries=max_loops,
            )
            self.prelude = Prelude(d_model, d_ff=96)
            self.core = RecurrentCore(d_model, d_ff=96, max_loops=2)
            self.coda = Coda(d_model, num_classes)
            self.halting_head = None

        elif arch == "untied_deep_readout_b2":
            # Reviewer B1 baseline: L = max_loops UNTIED readout + core blocks (same depth L as DeltaLoop,
            # ~3x more parameters due to untied layers, testing weight-tying vs untied depth on ID & OOD!)
            self.gdn = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                beta_max=2.0,
                decay_mode="learnable",
            )
            self.prelude = Prelude(d_model, d_ff=192)
            self.untied_readouts = nn.ModuleList(
                [
                    MemoryReadout(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
                    for _ in range(max_loops)
                ]
            )
            self.untied_cores = nn.ModuleList(
                [RecurrentCore(d_model, d_ff=192, max_loops=max_loops) for _ in range(max_loops)]
            )
            self.coda = Coda(d_model, num_classes)
            self.halting_head = None

        else:
            beta_max = 1.0 if arch in ("single_gdn_b1", "deltaloop_b1") else 2.0
            self.gdn = GatedDeltaNetLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                beta_max=beta_max,
                decay_mode="learnable",
            )
            self.readout = MemoryReadout(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
            )
            self.prelude = Prelude(d_model, d_ff=192)
            self.core = RecurrentCore(d_model, d_ff=192, max_loops=max(max_loops, 8))
            self.coda = Coda(d_model, num_classes)

            if arch in ("deltaloop_b1", "deltaloop_b2", "deltaloop_b2_scurr"):
                self.halting_head = AdaptiveHaltingHead(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_loops=max(max_loops, 8),
                )
            else:
                self.halting_head = None

    @property
    def is_multi_depth(self) -> bool:
        return self.arch in (
            "static_query_loop_b2",
            "untied_deep_readout_b2",
            "deltaloop_b1",
            "deltaloop_b2",
            "deltaloop_b2_scurr",
        )

    @property
    def state_numel(self) -> int:
        if self.arch == "softmax_transformer":
            return -1
        if self.arch == "two_layer_gdn_b2":
            return self.gdn1.state_numel + self.gdn2.state_numel
        if self.arch == "write_loop_gdn_b2":
            return self.write_loop.state_numel
        return self.gdn.state_numel

    def core_parameter_count(self) -> int:
        halt_params = (
            sum(p.numel() for p in self.halting_head.parameters())
            if self.halting_head is not None
            else 0
        )
        return sum(p.numel() for p in self.parameters()) - halt_params

    def forward(
        self,
        tokens: torch.Tensor,
        num_loops: Optional[int] = None,
        return_probes: bool = False,
    ) -> Dict[str, object]:
        B, T, _ = tokens.shape
        non_noop = (tokens[..., 0] != 0).unsqueeze(-1).float()
        z = self.encoder(tokens)
        z_query = z[:, -1]

        if self.arch == "softmax_transformer":
            x = z
            for blk in self.blocks:
                x = blk(x)
            logits = self.coda(x[:, -1])
            return {"logits": logits, "logits_per_depth": logits.unsqueeze(1)}

        if self.arch == "two_layer_gdn_b2":
            res1 = self.gdn1(z, beta_gate=non_noop, read_from="S_prev", return_penultimate_state=True)
            assert res1["seq_out"] is not None
            z_mid = z + res1["seq_out"]
            z_mid = z_mid + self.mid_ffn(self.mid_norm(z_mid))
            res2 = self.gdn2(z_mid, beta_gate=non_noop, read_from="S_prev", return_penultimate_state=True)

            h0 = self.prelude(z_query)
            e0 = self.core.get_loop_embedding(0, B, z.device, z.dtype)
            ro1 = self.readout1(h0 + e0, res1["S_penult"], res1["n_penult"])  # type: ignore[arg-type]
            ro2 = self.readout2(h0 + e0, res2["S_penult"], res2["n_penult"])  # type: ignore[arg-type]
            m0 = 0.5 * (ro1["m"] + ro2["m"])
            h1 = self.core(h0, m0, z_query, e0)
            logits = self.coda(h1)
            return {"logits": logits, "logits_per_depth": logits.unsqueeze(1)}

        if self.arch == "write_loop_gdn_b2":
            S_mem, n_mem = self.write_loop.forward_prefix(z[:, :-1], non_noop[:, :-1])
            h0 = self.prelude(z_query)
            e0 = self.core.get_loop_embedding(0, B, z.device, z.dtype)
            ro = self.readout(h0 + e0, S_mem, n_mem)
            h1 = self.core(h0, ro["m"], z_query, e0)
            logits = self.coda(h1)
            return {"logits": logits, "logits_per_depth": logits.unsqueeze(1)}

        if self.arch == "parallel_multiquery_b2":
            res = self.gdn(z, beta_gate=non_noop, read_from="S_prev", return_penultimate_state=True)
            h0 = self.prelude(z_query)
            e0 = self.core.get_loop_embedding(0, B, z.device, z.dtype)
            ro = self.pmq_readout(h0 + e0, res["S_penult"], res["n_penult"])  # type: ignore[arg-type]
            h1 = self.core(h0, ro["m"], z_query, e0)
            logits = self.coda(h1)
            return {"logits": logits, "logits_per_depth": logits.unsqueeze(1)}

        if self.arch == "untied_deep_readout_b2":
            res = self.gdn(z, beta_gate=non_noop, read_from="S_prev", return_penultimate_state=True)
            S_mem = res["S_penult"]
            n_mem = res["n_penult"]
            assert S_mem is not None
            L_run = min(num_loops if num_loops is not None else self.max_loops, self.max_loops)
            h = self.prelude(z_query)
            logits_depths = []
            for l_idx in range(L_run):
                e_l = self.untied_cores[l_idx].get_loop_embedding(l_idx, B, z.device, z.dtype)
                ro = self.untied_readouts[l_idx](h + e_l, S_mem, n_mem)
                h = self.untied_cores[l_idx](h, ro["m"], z_query, e_l)
                logits_depths.append(self.coda(h))
            logits_stack = torch.stack(logits_depths, dim=1)
            return {"logits": logits_stack[:, -1], "logits_per_depth": logits_stack}

        read_mode: Literal["S_prev", "S_curr"] = (
            "S_curr" if self.arch == "deltaloop_b2_scurr" else "S_prev"
        )
        res = self.gdn(
            z,
            beta_gate=non_noop,
            read_from=read_mode,
            return_penultimate_state=(read_mode == "S_prev"),
            return_keys=return_probes,
        )
        S_mem = res["S_penult"] if read_mode == "S_prev" else res["S_final"]
        n_mem = res["n_penult"] if read_mode == "S_prev" else res["n_final"]
        assert S_mem is not None

        if self.arch in ("single_gdn_b1", "single_gdn_b2"):
            L_run = 1
        else:
            L_run = num_loops if num_loops is not None else self.max_loops

        h = self.prelude(z_query)
        logits_depths = []
        h_list: List[torch.Tensor] = []
        m_list: List[torch.Tensor] = []
        dh_norms: List[torch.Tensor] = []
        r_norms: List[torch.Tensor] = []
        coverages: List[torch.Tensor] = []
        queries: List[torch.Tensor] = []

        cached_q0: Optional[torch.Tensor] = None
        for l_idx in range(L_run):
            e_l = self.core.get_loop_embedding(l_idx, B, z.device, z.dtype)
            fixed_q = cached_q0 if (self.arch == "static_query_loop_b2" and l_idx > 0) else None
            ro = self.readout(h + e_l, S_mem, n_mem, fixed_q=fixed_q)
            if l_idx == 0:
                cached_q0 = ro["q"]

            h_next = self.core(h, ro["m"], z_query, e_l)
            dh = torch.linalg.vector_norm((h_next - h).float(), ord=2, dim=-1, keepdim=True)
            logits_l = self.coda(h_next)

            logits_depths.append(logits_l)
            h_list.append(h_next)
            m_list.append(ro["m"])
            dh_norms.append(dh)
            r_norms.append(ro["r_norm"])
            coverages.append(ro["coverage"])
            if return_probes:
                queries.append(ro["q"].detach())
            h = h_next

        logits_stack = torch.stack(logits_depths, dim=1)
        out_dict: Dict[str, object] = {
            "logits": logits_stack[:, -1],
            "logits_per_depth": logits_stack,
            "dh_norms": torch.cat(dh_norms, dim=1),
        }

        if self.halting_head is not None:
            halt_out = self.halting_head(
                h_list=h_list,
                m_list=m_list,
                dh_norms=dh_norms,
                r_norms=r_norms,
                coverages=coverages,
            )
            out_dict["halt"] = halt_out

        if return_probes and res["keys"] is not None:
            k_seq = res["keys"].detach()
            q_stack = torch.stack(queries, dim=1)
            qk_sim = torch.einsum("blhd,bthd->blht", q_stack.float(), k_seq.float())
            out_dict["qk_sim"] = qk_sim

        return out_dict


class MultiTimescaleSequenceModel(nn.Module):
    """Sequence model wrapper for Task 1B (Iso-State-Size Multi-Timescale Benchmark)."""

    def __init__(
        self,
        variant: Variant1B,
        num_ops: int,
        vocab_size: int,
        num_classes: int,
        d_model: int = 96,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.d_model = d_model

        # Disable absolute positional embeddings so Task 1B strictly tests recurrent dynamics!
        self.encoder = StructuredTokenEncoder(
            num_ops=num_ops,
            vocab_size=vocab_size,
            d_model=d_model,
            use_time_emb=False,
        )
        self.memory = MultiTimescaleGDN(
            d_model=d_model,
            variant=variant,
            d_k=16,
            d_v=16,
            beta_max=1.0,
            chunk_size=chunk_size,
        )

        # Match total parameter count (~174.5K, within <0.5%!) across all 8 variants!
        d_ff_map = {
            "1_single_mh": 196,
            "1b_stacked_3layer": 131,
            "2_3bank_same_fixed": 198,
            "3_3bank_diverse_fixed": 198,
            "4_3bank_learnable": 196,
            "5a_3bank_consolidation": 129,
            "5b_3bank_routing": 179,
            "5c_3bank_full": 112,
        }
        d_ff = d_ff_map[variant]
        self.prelude = Prelude(d_model, d_ff=d_ff)
        self.h_norm = nn.RMSNorm(d_model)
        self.m_norm = nn.RMSNorm(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.coda = Coda(d_model, num_classes)

    @property
    def state_numel(self) -> int:
        return self.memory.total_state_numel

    def forward(self, tokens: torch.Tensor) -> Dict[str, object]:
        non_noop = (tokens[..., 0] != 0).unsqueeze(-1).float()
        z = self.encoder(tokens)
        mem_state = self.memory.write_sequence(z, non_noop_gate=non_noop, read_from="S_prev")
        z_q = z[:, -1]
        h0 = self.prelude(z_q)
        ro = self.memory.read_step(h0, mem_state)
        h1 = h0 + self.fuse(torch.cat([self.h_norm(h0), self.m_norm(ro["m"])], dim=-1))
        logits = self.coda(h1)
        return {
            "logits": logits,
            "pi": ro.get("pi", None),
            "mem_diag": mem_state,
        }
