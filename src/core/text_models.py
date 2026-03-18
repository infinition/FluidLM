"""FluidLM V4.5.0 core modules -- Dual-ODE language model.

Architecture: Reaction-Diffusion PDE + Selective State Space Model (Mamba).
  - Spatial PDE (Laplacian diffusion): local coherence, smoothness
  - Selective SSM (Mamba): content-based temporal routing, long-range deps
  - SwiGLU reaction: semantic nonlinearity
  - RMSNorm: stable normalization
  - Persistent h-state: global memory across segments
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding applied once at the input."""

    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SwiGLU(nn.Module):
    """SwiGLU activation (Shazeer 2020). Used in LLaMA, PaLM.

    out = (Swish(W_gate @ x)) * (W_up @ x), then project down.
    Uses 8/3 ratio to match GELU 4x param count.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(2 * (4 * d_model) / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Selective SSM (Mamba) -- pure PyTorch, no custom CUDA kernels
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Mamba-style selective state space model.

    Replaces CausalLongConv with content-based temporal routing.
    The SSM is a discretized ODE: dx/dt = Ax + Bu, y = Cx + Du
    where A, B, C are input-dependent (selective scan).

    This is mathematically in the same family as FluidLM's reaction-diffusion
    PDE, making it a natural complement: spatial PDE + temporal SSM.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.dt_rank = max(1, d_model // 16)

        # Input projection: x -> (x_ssm, gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short causal conv (local context before SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )

        # SSM projections: x -> (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: log-spaced initialization (negative real eigenvalues for stability)
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection (like residual)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialize dt bias for numerical stability
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project and split into SSM input + gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Short causal conv + activation
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_ssm = F.silu(x_conv.transpose(1, 2))

        # SSM parameter projections
        x_dbl = self.x_proj(x_ssm)
        dt_raw, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # Discretize A (negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan
        y = self._selective_scan(x_ssm, dt, A, B, C)

        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,      # (B, L, d_inner)
        dt: torch.Tensor,     # (B, L, d_inner)
        A: torch.Tensor,      # (d_inner, d_state)
        B: torch.Tensor,      # (B, L, d_state)
        C: torch.Tensor,      # (B, L, d_state)
    ) -> torch.Tensor:
        """Sequential selective scan. Pure PyTorch, no custom kernels.

        Implements the recurrence:
            h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
            y_t = C_t @ h_t + D * x_t
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Precompute discretized A and B*x for all timesteps
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        # dB_x: (B, L, d_inner, d_state)
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        # Sequential scan over time
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB_x[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        # Skip connection
        y = y + x * self.D
        return y


# ---------------------------------------------------------------------------
# Core FluidLayer
# ---------------------------------------------------------------------------

class FluidLayer(nn.Module):
    """1D reaction-diffusion layer with Mamba SSM and SwiGLU.

    Dual-ODE architecture:
        du/dt = Laplacian_diffusion(u)   -- spatial coherence
              + SelectiveSSM(u)          -- content-based routing
              + SwiGLU(u)                -- semantic nonlinearity
              + alpha * h_state          -- global memory
              + alpha_local * local_mem  -- causal local summary
    Integration: Forward Euler with adaptive stopping.
    """

    def __init__(
        self,
        d_model: int,
        num_layers_total: int = 4,
        dropout: float = 0.1,
        init_dt: float = 0.1,
        min_steps: int = 3,
        stop_patience: int = 2,
        stop_probe_tokens: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.dilations = [1, 4, 16]
        self.min_steps = max(1, int(min_steps))
        self.stop_patience = max(1, int(stop_patience))
        self.stop_probe_tokens = max(1, int(stop_probe_tokens))
        self.local_memory_tokens = 32
        self.residual_scale = 1.0 / math.sqrt(num_layers_total)

        # Diffusion (3-scale Laplacian)
        self.register_buffer(
            "diffusion_kernel", torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3)
        )
        self.diff_coeffs = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, d_model) * 0.15),
            nn.Parameter(torch.ones(1, 1, d_model) * 0.10),
            nn.Parameter(torch.ones(1, 1, d_model) * 0.08),
        ])

        # Selective SSM (replaces CausalLongConv)
        self.ssm = SelectiveSSM(d_model, d_state=16, d_conv=4, expand=1)

        # SwiGLU reaction
        self.reaction = SwiGLU(d_model, dropout=dropout)

        # Local causal memory
        self.local_memory_proj = nn.Linear(d_model, d_model)
        self.alpha_local_param = nn.Parameter(torch.tensor(0.02))

        # Global memory pump (h-state)
        self.memory_gate_input_norm = RMSNorm(d_model)
        self.memory_gate_state_norm = RMSNorm(d_model)
        self.memory_gate_x = nn.Linear(d_model, d_model)
        self.memory_gate_h = nn.Linear(d_model, d_model, bias=False)
        self.memory_gate_temp = nn.Parameter(torch.tensor(3.5))
        self.decay_param = nn.Parameter(torch.full((d_model,), 3.0))

        # Integration
        self.norm = RMSNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        dt_clamped = max(0.002, min(0.199, init_dt))
        dt_init = math.log((dt_clamped - 0.001) / (0.2 - dt_clamped))
        self.dt_gate = nn.Parameter(torch.tensor(dt_init))
        self.alpha_param = nn.Parameter(torch.tensor(0.05))

    def _make_stop_probe(self, x: torch.Tensor) -> torch.Tensor:
        probe_tokens = min(self.stop_probe_tokens, x.shape[1])
        return x[:, -probe_tokens:, :]

    def _causal_local_memory(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        window = max(1, math.ceil(seq_len / self.local_memory_tokens))
        hidden = x.transpose(1, 2)
        padded = F.pad(hidden, (window - 1, 0), mode="constant", value=0.0)
        local_mem = F.avg_pool1d(padded, kernel_size=window, stride=1)
        return local_mem.transpose(1, 2)

    def _should_stop(self, history: List[float], step_idx: int, epsilon: float) -> bool:
        if epsilon <= 0.0:
            return False
        if (step_idx + 1) < self.min_steps:
            return False
        if len(history) < self.stop_patience:
            return False
        recent = history[-self.stop_patience:]
        return max(recent) < epsilon and recent[-1] <= recent[0] * 1.05

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int,
        dt,
        return_history: bool = False,
        epsilon: float = 1e-4,
        h_state_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict] | Tuple[torch.Tensor, List, Dict]:
        batch, seq_len, d_model = x.shape

        if h_state_in is not None:
            h_state = h_state_in
        else:
            h_state = torch.zeros(batch, d_model, device=x.device, dtype=x.dtype)

        hist = []
        stop_history: List[float] = []
        diff_turbulences: List[torch.Tensor] = []
        steps_needed = max_steps
        equilibrium_step = max_steps
        prev_probe = self._make_stop_probe(x).detach()

        kernel = self.diffusion_kernel.expand(self.d_model, 1, 3)
        dt_val = 0.001 + (0.2 - 0.001) * torch.sigmoid(self.dt_gate)
        alpha = F.softplus(self.alpha_param)
        alpha_local = F.softplus(self.alpha_local_param)
        diffusion_coeffs = [F.softplus(c) for c in self.diff_coeffs]

        for step_idx in range(max_steps):
            # Diffusion (3-scale Laplacian)
            out = x.transpose(1, 2)
            total_diffusion = torch.zeros_like(x)
            for idx, dilation in enumerate(self.dilations):
                pad_len = 2 * dilation
                padded = F.pad(out, (pad_len, 0), mode="constant", value=0.0)
                lap = F.conv1d(padded, kernel, dilation=dilation, groups=self.d_model)
                total_diffusion = total_diffusion + lap.transpose(1, 2) * diffusion_coeffs[idx]

            # Selective SSM (content-based temporal routing)
            ssm_out = self.ssm(x)

            # SwiGLU reaction
            react = self.reaction(x)

            # Causal local memory
            local_mem = self.local_memory_proj(self._causal_local_memory(x))
            local_mem = self.dropout_layer(local_mem)

            # Global memory pump
            summary_tokens = min(self.stop_probe_tokens, seq_len)
            react_summary = react[:, -summary_tokens:, :].mean(dim=1)
            x_summary = x[:, -summary_tokens:, :].mean(dim=1)
            gate_logits = (
                self.memory_gate_x(self.memory_gate_input_norm(x_summary))
                + self.memory_gate_h(self.memory_gate_state_norm(h_state))
            )
            memory_gate_temp = self.memory_gate_temp.abs() + 1.0
            gate = torch.sigmoid(torch.tanh(gate_logits / memory_gate_temp) * 2.0)
            decay = torch.sigmoid(self.decay_param)
            h_state = decay * h_state + gate * torch.tanh(react_summary)

            # PDE integration (forward Euler)
            du = total_diffusion + ssm_out + react + alpha * h_state.unsqueeze(1) + alpha_local * local_mem
            x_candidate = x + self.residual_scale * dt_val * du
            x = self.norm(x_candidate)

            if return_history:
                hist.append(x.abs().mean(dim=-1).detach().cpu().float().numpy())

            # Stopping criterion
            current_probe = self._make_stop_probe(x_candidate).detach()
            stop_turb = (current_probe - prev_probe).abs().mean() / (prev_probe.abs().mean() + 1e-8)
            step_energy = du.abs().mean()
            lap_energy = total_diffusion.abs().mean()
            diff_turb = stop_turb + 0.05 * step_energy + 0.01 * lap_energy
            diff_turbulences.append(diff_turb)

            stop_val = float(stop_turb.item())
            stop_history.append(stop_val)
            if stop_val < epsilon and equilibrium_step == max_steps:
                equilibrium_step = step_idx + 1
            steps_needed = step_idx + 1
            prev_probe = current_probe

            if self._should_stop(stop_history, step_idx, epsilon):
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else torch.tensor(0.0, device=x.device)
        )

        # Gate regularization
        gate_margin = torch.relu((gate - 0.5).abs() - 0.30)
        decay_target = torch.tensor(0.90, device=x.device, dtype=x.dtype)
        gate_reg_loss = gate_margin.pow(2).mean()
        decay_reg_loss = torch.relu(decay.mean() - decay_target).pow(2)
        total_gate_reg = gate_reg_loss + 0.5 * decay_reg_loss

        with torch.no_grad():
            gate_mean_val = gate.mean()
            gate_sat_val = ((gate < 0.05) | (gate > 0.95)).float().mean()
            decay_mean_val = torch.sigmoid(self.decay_param).mean()

        info = {
            "steps_needed": steps_needed,
            "stop_history": stop_history,
            "equilibrium_step": equilibrium_step,
            "final_turbulence": stop_history[-1] if stop_history else 0.0,
            "min_turbulence": min(stop_history) if stop_history else 0.0,
            "diff_turbulence": diff_turb_mean,
            "gate_mean": gate_mean_val.detach(),
            "gate_sat": gate_sat_val.detach(),
            "decay_mean": decay_mean_val.detach(),
            "gate_reg_loss": gate_reg_loss.detach(),
            "decay_reg_loss": decay_reg_loss.detach(),
            "total_gate_reg": total_gate_reg.detach(),
            "h_state_out": h_state.detach(),
        }

        return (x, hist, info) if return_history else (x, info)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FluidNet(nn.Module):
    """Dual-ODE language model: Reaction-Diffusion PDE + Selective SSM."""

    def __init__(
        self,
        v_size: int,
        d_model: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        init_dt: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = nn.Embedding(v_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            FluidLayer(d_model, num_layers_total=num_layers, dropout=dropout, init_dt=init_dt)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, v_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(
        self,
        x: torch.Tensor,
        steps: int,
        dt,
        return_history: bool = False,
        epsilon: float = 1e-4,
        h_states: Optional[List[torch.Tensor]] = None,
    ):
        x = self.embedding(x)
        x = self.pos_enc(x)

        hist = []
        total_steps = 0
        layer_turbulences = []
        layer_infos = []
        h_states_out = []

        for idx, layer in enumerate(self.layers):
            is_last = idx == len(self.layers) - 1
            h_in = h_states[idx] if h_states is not None else None

            if return_history and is_last:
                x, hist, info = layer(x, steps, dt, return_history=True, epsilon=epsilon, h_state_in=h_in)
            else:
                x, info = layer(x, steps, dt, epsilon=epsilon, h_state_in=h_in)

            total_steps += info["steps_needed"]
            layer_turbulences.append(info["diff_turbulence"])
            layer_infos.append(info)
            h_states_out.append(info["h_state_out"])

        avg_steps = total_steps / len(self.layers)
        self._last_hidden = x
        self._h_states_out = h_states_out
        logits = self.head(x)

        mean_diff_turb = (
            torch.stack(layer_turbulences).mean()
            if layer_turbulences
            else torch.tensor(0.0, device=x.device)
        )

        def _mean_stat(key: str):
            values = [info[key] for info in layer_infos if key in info and isinstance(info[key], torch.Tensor)]
            return torch.stack(values).mean() if values else torch.tensor(0.0, device=x.device)

        gate_stats = {
            "gate_mean": _mean_stat("gate_mean"),
            "gate_sat": _mean_stat("gate_sat"),
            "decay_mean": _mean_stat("decay_mean"),
            "gate_reg_loss": _mean_stat("gate_reg_loss"),
            "decay_reg_loss": _mean_stat("decay_reg_loss"),
            "total_gate_reg": _mean_stat("total_gate_reg"),
        }

        if return_history:
            return logits, hist, avg_steps, mean_diff_turb, gate_stats
        return logits, avg_steps, mean_diff_turb, gate_stats
