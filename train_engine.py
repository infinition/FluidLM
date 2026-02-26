#!/usr/bin/env python3
# =============================================================================
# FluidLM Training Engine — V4.2.2
# =============================================================================
#
# Core training loop for the FluidLM (Fluid Language Model) architecture.
#
# This file contains:
#   1. BPETokenizer     — GPT-2 BPE tokenizer wrapper (via tiktoken)
#   2. TextDataset      — Sliding-window dataset with configurable stride
#   3. FluidLayer       — The fundamental reaction-diffusion PDE layer
#   4. FluidNet         — Full model: embedding → N×FluidLayer → tied head
#   5. generate_text()  — Autoregressive sampling with repetition penalty
#   6. train()          — Main training loop with live config hot-reloading
#
# Key design decisions documented inline. The architecture replaces the
# Transformer's O(N²) self-attention with O(N) multi-scale dilated diffusion
# governed by a discretized PDE (forward Euler). See README.md §4 for the
# full mathematical treatment.
#
# Author: Fabien POLLY aka Infinition
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import json
import time
import numpy as np
from tqdm import tqdm
import tiktoken

# ── Device selection ─────────────────────────────────────────────────────────
# We default to CUDA if available. MPS (Apple Silicon) support could be added
# here but is untested with the dilated conv1d operations.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── File paths ───────────────────────────────────────────────────────────────
# These three JSON files form the IPC layer between the training engine and
# the Streamlit dashboard:
#   - CONFIG_FILE: bidirectional — dashboard writes params, engine reads them
#   - LOG_FILE:    engine → dashboard — per-step telemetry snapshot
#   - STATS_FILE:  engine → dashboard — rolling history (last 1000 points)
CONFIG_FILE = "config.json"
LOG_FILE = "live_logs.json"
STATS_FILE = "training_stats.json"

SAVE_PATH = "./training"
MODEL_FILE = os.path.join(SAVE_PATH, "fluidlm_model.pth")


# =============================================================================
# Utility Functions
# =============================================================================

def atomic_write(data, path, retries=3):
    """
    Atomic JSON write using tmp file + os.replace().

    Critical for concurrent access: the training loop and Streamlit dashboard
    both read/write config.json. Without atomic writes, a partial write could
    corrupt the JSON and crash the reader. os.replace() is atomic on POSIX
    systems and nearly atomic on Windows (NTFS).

    Retry logic handles rare filesystem lock contention (observed on Windows
    with antivirus software holding brief locks on newly created files).
    """
    for attempt in range(retries):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"⚠️ Failed to write {path} after {retries} attempts: {e}")
            time.sleep(0.1 * (attempt + 1))
    return False


def load_config():
    """
    Load the shared configuration file, falling back to sensible defaults
    if the file doesn't exist or is corrupted.

    Default hyperparameters rationale:
      - lr=1e-4: standard AdamW starting point for small models
      - batch_size=32, seq_len=128: conservative for 8-12GB VRAM
      - d_model=512: ~36M params — fast iteration for architecture research
      - t_steps=12: max integration steps per layer (adaptive may use fewer)
      - dt=0.1: Euler step size — larger = faster integration but less stable
      - epsilon=1e-4: convergence threshold for Turing equilibrium detection
      - grad_accum_steps=2: effective batch size = 32×2 = 64
      - warmup_steps=500: linear LR warmup to avoid early training instability
    """
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "lr": 1e-4,
            "batch_size": 32,
            "seq_len": 128,
            "d_model": 512,
            "t_steps": 12,
            "dt": 0.1,
            "repetition_penalty": 1.5,
            "temperature": 0.8,
            "grad_accum_steps": 2,
            "warmup_steps": 500,
            "epsilon": 0.05,
        }


# =============================================================================
# Tokenizer
# =============================================================================

class BPETokenizer:
    """
    Thin wrapper around tiktoken's GPT-2 BPE encoding.

    We use GPT-2's tokenizer (50,257 vocab) rather than training our own
    because: (a) the architecture research is about the model, not the
    tokenizer, and (b) BPE gives us a realistic vocabulary size that tests
    the embedding/head weight-tying at production scale.

    For a proper scaling study, you'd want to train a SentencePiece or
    BPE tokenizer on your specific corpus. For PoC purposes, GPT-2's
    tokenizer is a well-understood baseline.
    """

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        """Encode text → tensor of token IDs."""
        return torch.tensor(
            self.enc.encode(text, allowed_special="all"), dtype=torch.long
        )

    def decode(self, tokens):
        """Decode token IDs → text. Silently handles invalid sequences."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        try:
            return self.enc.decode(tokens)
        except Exception:
            return ""


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
    """
    Sliding-window text dataset with configurable stride.

    Design choices:
      - stride = seq_len // 2: 50% overlap between consecutive windows.
        This doubles the effective dataset size compared to non-overlapping
        windows, at zero additional memory cost (we index into a single
        contiguous token tensor). The overlap also means that most tokens
        appear in two different context positions, which acts as a mild
        form of data augmentation.

      - We load ALL .txt files from the data directory and concatenate them
        into a single stream. For a more principled approach, you'd want
        document boundaries and proper shuffling at the document level
        (as in GPT-2/3 training), but for a PoC this is sufficient.

      - Labels are simply the input shifted by one position (standard
        causal language modeling objective).
    """

    def __init__(self, path, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.stride = seq_len // 2  # 50% overlap between windows

        # ── Load and concatenate all .txt files ──────────────────────────
        files = glob.glob(os.path.join(path, "*.txt"))
        raw = ""
        for f in files:
            with open(f, "r", encoding="utf-8") as fl:
                raw += fl.read() + "\n"

        if not raw:
            raise ValueError("❌ No .txt files found in /data/")

        print("🧠 Tokenizing dataset (BPE)...")
        self.data = self.tokenizer.encode(raw)
        print(f"✅ Dataset loaded: {len(self.data):,} tokens.")

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        i = idx * self.stride
        return self.data[i : i + self.seq_len], self.data[i + 1 : i + self.seq_len + 1]


# =============================================================================
# FluidLayer — The Core PDE Unit
# =============================================================================

class FluidLayer(nn.Module):
    """
    A single Reaction-Diffusion layer implementing the FluidLM governing equation:

        du/dt = Σ_k [ D_k · Laplacian_{d_k}(u) ] + R(u, θ) + α · h_t

    where:
      - The first term is multi-scale diffusion (local information propagation)
      - R(u, θ) is the reaction function (nonlinear per-token MLP)
      - h_t is the intra-sequence memory pump (gated accumulation)

    This replaces the Transformer's self-attention + FFN block.

    Complexity: O(N × d_model × K) per time step, where K=3 (number of
    dilation scales). Compare to O(N² × d_model) for self-attention.

    The layer integrates the PDE for up to `max_steps` Euler steps, but can
    halt early when the system reaches Turing equilibrium (turbulence < ε).
    During training, it always runs all steps (for consistent gradients) but
    records when equilibrium would have been reached. During inference
    (model.eval()), it genuinely halts early, saving compute.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # ── Reaction function ────────────────────────────────────────────
        # 2-layer MLP with GELU activation and hidden dim = 2×d_model.
        # This is the "chemistry" — local nonlinear transformation that
        # creates new feature combinations. Without it, pure diffusion
        # would blur everything to a uniform average.
        self.reaction = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ── Memory gate ──────────────────────────────────────────────────
        # Sigmoid gate controlling how much reaction output enters the
        # accumulation reservoir (h-state). This lets the model learn
        # *which* reactions are worth remembering across integration steps.
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # ── Discrete Laplacian kernel ────────────────────────────────────
        # The 1D discrete Laplacian: [1, -2, 1]
        # Applied as a depthwise conv1d (groups=d_model), computing:
        #   Lap(u_i) = u_{i-d} - 2·u_i + u_{i+d}
        # for each dilation d. This measures how different a token's
        # representation is from its neighbors — where the gradient is
        # steep, diffusion is strong.
        self.register_buffer(
            "diffusion_kernel", torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3)
        )

        # ── Multi-scale dilation configuration ───────────────────────────
        # Three scales capture different linguistic granularities:
        #   d=1  → adjacent tokens (morphology, local syntax)
        #   d=4  → phrase-level (4-token span ≈ 1-2 words in BPE)
        #   d=16 → sentence/paragraph level dependencies
        # Over T=12 steps × 4 layers, the coarsest scale reaches ~768 positions.
        self.dilations = [1, 4, 16]

        # Learnable diffusion coefficients D_k — one per scale, per feature dim.
        #
        # [FIX v4.2.1] Staggered initialization by dilation:
        #   - dilation=1  → 0.15 (slightly above the original 0.1 init)
        #   - dilation=4  → 0.10 (forced to exist from the start)
        #   - dilation=16 → 0.08 (forced to exist from the start)
        # Without this, the gradient kills long-range dilations because
        # local dependencies (dilation=1) are sufficient to reduce next-token
        # loss on short sequences (seq_len=128).
        # The differential LR in the optimizer complements this initialization
        # by compensating for the naturally weak gradient on long-range D_k.
        self.diff_coeffs = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, d_model) * 0.15),  # dilation=1
            nn.Parameter(torch.ones(1, 1, d_model) * 0.10),  # dilation=4
            nn.Parameter(torch.ones(1, 1, d_model) * 0.08),  # dilation=16
        ])

        # ── Post-step normalization ──────────────────────────────────────
        # LayerNorm after each Euler step prevents state divergence.
        # Forward Euler is only conditionally stable; without normalization,
        # large dt or aggressive diffusion coefficients can cause the state
        # to explode. LayerNorm acts as a soft constraint keeping the state
        # on a bounded manifold.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, max_steps, dt, return_history=False, epsilon=1e-4):
        """
        Integrate the reaction-diffusion PDE for up to max_steps Euler steps.

        Args:
            x:              Input tensor [batch, seq_len, d_model]
            max_steps:      Maximum number of integration steps (T_max)
            dt:             Euler step size (Δt)
            return_history: If True, record activation heatmaps per step
            epsilon:        Convergence threshold for Turing equilibrium

        Returns:
            x:              Final latent state [batch, seq_len, d_model]
            hist:           (optional) List of activation snapshots
            steps_needed:   Number of steps before equilibrium was reached
        """
        # ── Initialize accumulators ──────────────────────────────────────
        # local_acc: the "memory pump" (h-state) — accumulates gated
        #   reaction outputs across time steps. Reset every forward pass
        #   (this is intra-sequence memory, not inter-sequence).
        local_acc = torch.zeros_like(x)
        hist = []
        steps_needed = max_steps
        equilibrium_reached = False

        # Detached copy for turbulence computation.
        # We detach to avoid including the convergence check in the
        # computational graph (it's a diagnostic, not a learned operation).
        x_prev = x.detach().clone()

        for step_idx in range(max_steps):
            # ── Multi-scale diffusion ────────────────────────────────────
            # Transpose to [batch, d_model, seq_len] for conv1d.
            out = x.transpose(1, 2)

            total_diffusion = torch.zeros_like(x)
            for i, d in enumerate(self.dilations):
                # Causal padding: pad_len = 2×dilation on the LEFT only.
                # This ensures token at position i can only receive info
                # from positions ≤ i (no future leakage during training).
                # For kernel size 3 with dilation d, we need padding = 2d
                # to maintain the output sequence length.
                pad_len = 2 * d
                padded_out = F.pad(out, (pad_len, 0), mode="constant", value=0.0)

                # Depthwise conv1d: each feature dimension is convolved
                # independently (groups=d_model), applying the Laplacian
                # kernel at the current dilation. This is O(N×d) per scale.
                lap = F.conv1d(
                    padded_out,
                    self.diffusion_kernel.expand(self.d_model, 1, 3),
                    dilation=d,
                    groups=self.d_model,
                )

                # Accumulate weighted diffusion from this scale.
                # diff_coeffs[i] has shape [1, 1, d_model] — each feature
                # dimension has its own learned diffusion rate at this scale.
                total_diffusion = total_diffusion + lap.transpose(1, 2) * self.diff_coeffs[i]

            # ── Reaction ─────────────────────────────────────────────────
            # Per-token MLP — the "chemistry" that creates new semantic
            # compounds. This is analogous to the FFN in a Transformer.
            react = self.reaction(x)

            # ── Memory pump update ───────────────────────────────────────
            # h_{t} = h_{t-1} + gate(u) · tanh(R(u))
            # The sigmoid gate controls what enters the reservoir;
            # tanh compresses the reaction output to [-1, 1].
            # α=0.05 keeps the memory influence moderate relative to
            # the primary diffusion and reaction forces.
            local_acc = local_acc + self.memory_gate(x) * torch.tanh(react)

            # ── Euler integration step ───────────────────────────────────
            # u_{t+1} = LayerNorm( u_t + Δt · [diffusion + reaction + α·h] )
            # This is the complete FluidLM update rule, discretized with
            # forward Euler and stabilized by LayerNorm.
            x = self.norm(x + dt * (total_diffusion + react + 0.05 * local_acc))

            # ── Record activation history (for Turing wave visualization) ─
            if return_history:
                hist.append(x.abs().mean(dim=-1).detach().cpu().float().numpy())

            # ── Turing equilibrium check ─────────────────────────────────
            # Every 3 steps (to reduce GPU overhead), we measure the
            # "turbulence" — how much the state has changed since the
            # last check. If turbulence < ε, the fluid has stabilized
            # into a Turing pattern and further integration is redundant.
            #
            # CRITICAL: We normalize per-token (dim=-1 gives the L2 norm
            # across d_model for each token, then .mean() averages across
            # batch and sequence positions). This makes the metric
            # INDEPENDENT of sequence length — a 10-token input and a
            # 100-token input are compared on the same scale.
            #
            # The old formula (torch.norm(x-x_prev) / x.numel()) divided
            # by batch×seq_len×d_model, making convergence artificially
            # easier for longer sequences (larger denominator).
            #
            # With this fix, epsilon values are on a different scale:
            #   Old ε=1e-4  ≈  New ε=0.05 (roughly)
            #   The new scale is the mean L2 displacement per token.
            #
            # During training: we record the step but DON'T break (we need
            # a consistent computational graph for gradient computation).
            # During inference (model.eval()): we genuinely halt early.
            if not equilibrium_reached and (step_idx % 3 == 2 or step_idx == max_steps - 1):
                with torch.no_grad():
                    # [FIX v4.2.2] Relative turbulence — divides the absolute delta
                    # by the current norm of x. Without this, LayerNorm re-inflates
                    # x at each step (~fixed norm √d_model ≈ 22.6 for d=512),
                    # making the absolute delta always large regardless of true
                    # convergence. Relative turbulence measures the RATIO
                    # change/amplitude, which is scale-invariant and sensitive to
                    # genuine fluid stabilization.
                    # Formula: mean(‖Δu_i‖ / (‖u_i‖ + 1e-8)) over all tokens.
                    delta_norm = (x - x_prev).norm(dim=-1)          # [batch, seq]
                    state_norm = x.norm(dim=-1).clamp(min=1e-8)      # [batch, seq]
                    turbulence = (delta_norm / state_norm).mean()

                if turbulence < epsilon:
                    steps_needed = step_idx + 1
                    equilibrium_reached = True
                    if not self.training:
                        break  # Genuine early exit during inference

                # Update reference point for next turbulence measurement
                x_prev = x.detach().clone()

        return (x, hist, steps_needed) if return_history else (x, steps_needed)


# =============================================================================
# FluidNet — Full Model
# =============================================================================

class FluidNet(nn.Module):
    """
    The complete FluidLM language model.

    Architecture:
        Input IDs → Embedding + PosEmb → [FluidLayer × num_layers] → Linear Head → Logits

    Key design choices:
      - Weight tying: the output head shares weights with the embedding.
        This is standard practice (Press & Wolf, 2017) and reduces params
        by ~25M for a 50K vocab with d_model=512.

      - Learned absolute positional embeddings (up to 4096 positions).
        This is a known limitation — the model cannot generalize to unseen
        sequence lengths without interpolation. RoPE integration is on the
        roadmap (see README §10).

      - num_layers=4: each layer runs its own independent PDE integration.
        The total "depth" is num_layers × avg_steps_per_layer, which is
        adaptive. With 4 layers × 12 max steps = 48 effective "layers"
        of computation in the worst case.

      - avg_steps is reported as the mean across all layers, giving a
        single scalar that tracks the model's computational effort.
    """

    def __init__(self, v_size, d_model, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(v_size, d_model)

        # Learned absolute positional encoding — pre-allocated for 4096 positions.
        # If seq_len > 4096, we fall back to linear interpolation (not ideal,
        # but sufficient for the PoC). RoPE would be strictly better here.
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, d_model))

        self.layers = nn.ModuleList([FluidLayer(d_model) for _ in range(num_layers)])

        # Output head — weight-tied with embedding (reduces param count
        # significantly and provides a useful inductive bias: tokens with
        # similar embeddings produce similar output distributions).
        self.head = nn.Linear(v_size, d_model, bias=False)  # Placeholder dims
        self.head = nn.Linear(d_model, v_size, bias=False)
        self.head.weight = self.embedding.weight  # Weight tying

    def forward(self, x, steps, dt, return_history=False, epsilon=1e-4):
        """
        Full forward pass: embed → integrate × N layers → project to vocab.

        Args:
            x:              Input token IDs [batch, seq_len]
            steps:          Max integration steps per layer
            dt:             Euler step size
            return_history: Record wave visualization data (last layer only)
            epsilon:        Convergence threshold

        Returns:
            logits:     Output logits [batch, seq_len, vocab_size]
            hist:       (optional) Activation history from last layer
            avg_steps:  Mean integration steps across all layers
        """
        seq_len = x.size(1)

        # ── Positional encoding with fallback interpolation ──────────────
        if seq_len > self.pos_emb.size(1):
            # Linear interpolation for sequences longer than pre-allocated size.
            # This is a stopgap — it works but degrades quality for very long
            # sequences. RoPE or ALiBi would handle this more gracefully.
            pos_emb = F.interpolate(
                self.pos_emb.transpose(1, 2), size=seq_len, mode="linear"
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_emb[:, :seq_len, :]

        x = self.embedding(x) + pos_emb

        hist = []
        total_steps = 0

        # ── Layer-by-layer PDE integration ───────────────────────────────
        for i, layer in enumerate(self.layers):
            is_last_layer = i == len(self.layers) - 1

            # Only record wave history for the last layer (to keep the
            # visualization meaningful and avoid excessive memory usage).
            if return_history and is_last_layer:
                x, hist, s = layer(x, steps, dt, return_history=True, epsilon=epsilon)
            else:
                x, s = layer(x, steps, dt, epsilon=epsilon)
            total_steps += s

        avg_steps = total_steps / len(self.layers)
        logits = self.head(x)

        return (logits, hist, avg_steps) if return_history else (logits, avg_steps)


# =============================================================================
# Text Generation
# =============================================================================

def generate_text(model, tokenizer, config, start_str="The ", max_tokens=60):
    """
    Autoregressive text generation with temperature and repetition penalty.

    The generation loop runs the full model in eval mode (which enables
    genuine early stopping in the PDE integration — see FluidLayer.forward).

    Repetition penalty implementation follows Keskar et al. (2019):
      - For tokens that have appeared in the recent window:
        - If logit < 0: multiply by penalty (makes it more negative → less likely)
        - If logit > 0: divide by penalty (makes it less positive → less likely)
      - This asymmetric treatment prevents the penalty from accidentally
        *boosting* tokens that the model actively wants to suppress.

    The history window (default 50 tokens) limits the penalty's reach,
    preventing it from suppressing common function words that naturally
    repeat across long outputs.

    Note: this is called inside the training loop every 50 steps for
    monitoring purposes. At ~30 tokens per sample, this adds ~0.5s of
    overhead per log step. For faster training, increase the log interval.
    """
    model.eval()  # Switches self.training=False → enables early stopping in PDE

    with torch.no_grad():
        if not start_str:
            start_str = " "
        enc = tokenizer.encode(start_str)
        if enc.numel() == 0:
            enc = torch.tensor([0], dtype=torch.long)
        chars = enc.unsqueeze(0).to(DEVICE)

        history_window = []
        window_size = 50

        for _ in range(max_tokens):
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, _ = model(
                    chars[:, -config["seq_len"] :],
                    config["t_steps"],
                    config["dt"],
                    epsilon=config.get("epsilon", 1e-4),
                )
                logits = logits[:, -1, :] / config["temperature"]

            # ── Repetition penalty ───────────────────────────────────────
            for idx in set(history_window):
                if logits[0, idx] < 0:
                    logits[0, idx] *= config["repetition_penalty"]
                else:
                    logits[0, idx] /= config["repetition_penalty"]

            next_c = torch.multinomial(F.softmax(logits, dim=-1), 1)
            chars = torch.cat([chars, next_c], dim=1)

            history_window.append(next_c.item())
            if len(history_window) > window_size:
                history_window.pop(0)

    model.train()  # Re-enable training mode for gradient computation
    return tokenizer.decode(chars[0].tolist())


# =============================================================================
# Data Loader Builder
# =============================================================================

def build_loader(cfg, tokenizer):
    """
    Construct a DataLoader from the current config.

    Called at startup and whenever batch_size or seq_len changes (detected
    by the hot-reload mechanism in the training loop). Rebuilding the loader
    is necessary because changing seq_len requires re-windowing the dataset,
    and changing batch_size requires a new DataLoader instance.

    num_workers=2 on Linux/Mac for async data loading; 0 on Windows because
    PyTorch's multiprocessing + Windows = pain.
    """
    ds = TextDataset("./data", cfg["seq_len"], tokenizer)
    if len(ds) == 0:
        raise ValueError(
            f"Dataset too small for seq_len={cfg['seq_len']}! Add more text to /data/."
        )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2 if os.name != "nt" else 0,
        persistent_workers=False,
    ), ds


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_lr(step, max_lr, warmup_steps):
    """
    Linear warmup schedule.

    During the first `warmup_steps` optimizer steps, the LR ramps linearly
    from 0 to max_lr. After warmup, it stays constant at max_lr.

    We don't use cosine decay here because the training runs indefinitely
    (no fixed total steps), and the auto-pilot system handles LR reduction
    on plateaus. For a fixed-length run, cosine decay would be better.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr


# =============================================================================
# Optimizer Builder — Differential LR per dilation
# =============================================================================

def build_optimizer(model, cfg):
    """
    Build the AdamW optimizer with differential learning rates per parameter group.

    [FIX v4.2.1] Diagnosed problem: long-range diffusion coefficients
    (dilation=4 and dilation=16) were converging toward ~0 because the
    next-token loss gradient on seq_len=128 does not reward long-range
    dependencies. The model was learning to ignore diffusion and effectively
    becoming a stacked MLP.

    Solution: multiplicative LR per dilation group.
      - other params   → lr × 1   (original behavior)
      - diff_coeffs[0] → lr × 10  (dilation=1,  moderate boost)
      - diff_coeffs[1] → lr × 50  (dilation=4,  strong boost)
      - diff_coeffs[2] → lr × 100 (dilation=16, very strong boost)

    The effective LR for each group is updated in the training loop
    via get_lr() × the group multiplier.
    """
    base_lr = cfg["lr"]

    # Split parameters into 4 groups by role
    diff_params_d1  = []  # dilation=1
    diff_params_d4  = []  # dilation=4
    diff_params_d16 = []  # dilation=16
    other_params    = []

    for name, p in model.named_parameters():
        if "diff_coeffs.0" in name:
            diff_params_d1.append(p)
        elif "diff_coeffs.1" in name:
            diff_params_d4.append(p)
        elif "diff_coeffs.2" in name:
            diff_params_d16.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params,   "lr": base_lr,         "lr_mult": 1},
        {"params": diff_params_d1, "lr": base_lr * 10,    "lr_mult": 10},
        {"params": diff_params_d4, "lr": base_lr * 50,    "lr_mult": 50},
        {"params": diff_params_d16,"lr": base_lr * 100,   "lr_mult": 100},
    ]

    # Filter out empty groups (in case the model is modified)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return torch.optim.AdamW(param_groups)


def update_lr(opt, current_lr):
    """
    Update the LR for each group while respecting its multiplier.
    Called at each step instead of the simple `g["lr"] = current_lr`.
    """
    for g in opt.param_groups:
        g["lr"] = current_lr * g.get("lr_mult", 1)


# =============================================================================
# Main Training Loop
# =============================================================================

def train():
    """
    The main training loop. Runs indefinitely (epoch after epoch) until
    manually stopped. Key features:

    1. Hot-reloading: every 10 optimizer steps, the config file is re-read.
       Changes to lr, dt, t_steps, epsilon, temperature, repetition_penalty
       take effect immediately. Changes to batch_size or seq_len trigger a
       DataLoader rebuild.

    2. Gradient accumulation: the effective batch size is
       batch_size × grad_accum_steps. We accumulate gradients over multiple
       micro-batches before calling optimizer.step(). This lets us simulate
       large batch training on limited VRAM.

    3. Mixed precision (AMP): FP16 forward pass with FP32 gradient scaling.
       Reduces VRAM usage by ~40% and speeds up training on Ampere+ GPUs.

    4. Checkpointing: best model saved every 500 steps (if loss improved),
       full checkpoint (model + optimizer + scaler) every 1000 steps.

    5. Telemetry: every 50 optimizer steps, we log a comprehensive snapshot
       (loss, VRAM, speed, sample text, Turing waves, weight histogram)
       to JSON files that the dashboard reads.

    6. Chat mode: when the dashboard sends a chat request, the engine
       pauses training, generates a response, writes it to the log file,
       then resumes training. This lets the researcher interactively
       probe the model's current capabilities without stopping training.

    [FIX v4.2.1] Three changes from V4.2:
      - build_optimizer() replaces direct AdamW → differential LR per dilation
      - update_lr() replaces simple loop over param_groups
      - Diffusion regularization: penalizes coefficients too close to 0
        (forces long-range dilations to remain active)
      - Reset diff_coeffs on checkpoint load if saved values are too small
        (< 0.02 on average)
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    cfg = load_config()
    tokenizer = BPETokenizer()

    # ── Model initialization ─────────────────────────────────────────────
    model = FluidNet(tokenizer.vocab_size, cfg["d_model"]).to(DEVICE)

    # [FIX v4.2.1] Optimizer with differential LR per dilation
    opt = build_optimizer(model, cfg)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    global_step = 0
    global_epoch = 0
    best_loss = float("inf")

    # Rolling statistics buffer — keeps last 1000 data points for dashboard plots
    stats = {
        "step": [], "loss": [], "vram": [], "it_s": [],
        "lr": [], "temp": [], "penalty": [], "avg_steps": [],
    }

    # ── Resume from checkpoint if available ──────────────────────────────
    if os.path.exists(MODEL_FILE):
        print("📂 Resuming from checkpoint...")
        ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
        model.load_state_dict(
            ckpt["model_state"] if isinstance(ckpt, dict) else ckpt,
            strict=False,
        )
        if isinstance(ckpt, dict):
            if "optimizer_state" in ckpt:
                # [FIX v4.2.1] The number of param_groups changed (1 → 4)
                # after introducing differential LR per dilation.
                # We attempt loading but silently ignore it if groups are
                # incompatible — the model weights are loaded correctly,
                # only the Adam moment state is lost (acceptable for resuming).
                try:
                    opt.load_state_dict(ckpt["optimizer_state"])
                except ValueError as e:
                    print(f"⚠️  Optimizer state ignored (incompatible): {e}")
                    print("   → Resuming with reset optimizer (weights OK).")
            if "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            global_step = ckpt.get("step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))

        # [FIX v4.2.1] Reset diffusion coefficients if saved values are too
        # small (diagnosed: dilation=16 was converging toward ~0.003 after
        # 10k steps, making long-range diffusion non-existent). Force a clean
        # restart if necessary.
        print("🔍 Checking diffusion coefficients...")
        needs_reinit = False
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.layers):
                for dil_idx, coeff in enumerate(layer.diff_coeffs):
                    mean_val = coeff.abs().mean().item()
                    dilation = layer.dilations[dil_idx]
                    if mean_val < 0.02:
                        needs_reinit = True
                        print(f"  ⚠️  Layer {layer_idx}, dilation={dilation}: "
                              f"mean={mean_val:.5f} < 0.02 → reinitialization required")

        if needs_reinit:
            print("🔧 Forcing diff_coeffs reinitialization...")
            with torch.no_grad():
                for layer in model.layers:
                    layer.diff_coeffs[0].fill_(0.15)  # dilation=1
                    layer.diff_coeffs[1].fill_(0.10)  # dilation=4
                    layer.diff_coeffs[2].fill_(0.08)  # dilation=16
            print("✅ Coefficients reinitialized: [0.15, 0.10, 0.08]")
        else:
            print("✅ Diffusion coefficients OK — no reinitialization needed.")

    loader, ds = build_loader(cfg, tokenizer)
    opt.zero_grad(set_to_none=True)  # set_to_none=True saves memory vs .zero_grad()

    # ── Infinite training loop ───────────────────────────────────────────
    while True:
        pbar = tqdm(loader, desc=f"Epoch {global_epoch}")

        for i, (x, y) in enumerate(pbar):
            batch_start = time.time()

            # ── Learning rate schedule ───────────────────────────────────
            # [FIX v4.2.1] update_lr() respects lr_mult of each group
            current_lr = get_lr(global_step, cfg["lr"], cfg.get("warmup_steps", 500))
            update_lr(opt, current_lr)

            # Determine if this is a logging step (every 50 optimizer steps)
            is_log_step = global_step > 0 and global_step % 50 == 0

            # ── Hot-reload configuration ─────────────────────────────────
            # Every 10 steps, re-read config.json to pick up any changes
            # made by the dashboard (LR slider, epsilon, temperature, etc.)
            if global_step > 0 and global_step % 10 == 0:
                new_cfg = load_config()
                # If batch_size or seq_len changed, we need to rebuild the
                # DataLoader (different window sizes, different batching).
                if (new_cfg["batch_size"] != cfg["batch_size"] or
                        new_cfg["seq_len"] != cfg["seq_len"]):
                    cfg = new_cfg
                    loader, ds = build_loader(cfg, tokenizer)
                    break  # Restart the epoch with the new DataLoader
                cfg = new_cfg

            # ── Pause / Chat handling ────────────────────────────────────
            if cfg.get("pause"):
                if cfg.get("request_chat"):
                    # Generate a response to the user's chat prompt
                    res = generate_text(
                        model, ds.tokenizer, cfg,
                        cfg["chat_prompt"], max_tokens=100,
                    )
                    # Append the generated text to the current log packet
                    try:
                        with open(LOG_FILE, "r") as f:
                            log_packet = json.load(f)
                    except Exception:
                        log_packet = {
                            "loss": 0, "step": global_step, "vram": 0,
                            "waves": [], "w_hist": [], "w_bins": [],
                            "avg_steps": 12,
                        }
                    log_packet["sample"] = res
                    atomic_write(log_packet, LOG_FILE)
                    # Clear the chat request and unpause
                    cfg["request_chat"] = False
                    cfg["pause"] = False
                    atomic_write(cfg, CONFIG_FILE)
                time.sleep(0.5)
                continue

            # ── Forward pass ─────────────────────────────────────────────
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out_model = model(
                    x, cfg["t_steps"], cfg["dt"],
                    return_history=is_log_step,
                    epsilon=cfg.get("epsilon", 1e-4),
                )

                if is_log_step:
                    logits_raw, waves, avg_steps = out_model
                else:
                    logits_raw, avg_steps = out_model
                    waves = []

                # Standard cross-entropy loss for causal LM
                loss = nn.CrossEntropyLoss()(
                    logits_raw.view(-1, tokenizer.vocab_size), y.view(-1)
                )

                # [FIX v4.2.2] Corrected diffusion regularization ─────────
                # Bug in v4.2.1: relu(0.05 - coeff.abs()) ignored negative
                # coefficients (anti-diffusion). A coeff at -0.08 had
                # abs()=0.08 > 0.05, so it passed without penalty even though
                # it was actively destroying diffusion.
                # Fix: relu(0.05 - coeff) penalizes any coeff < 0.05,
                # including negatives (anti-diffusion heavily penalized).
                diffusion_reg = torch.tensor(0.0, device=DEVICE)
                for layer in model.layers:
                    for coeff in layer.diff_coeffs:
                        diffusion_reg = diffusion_reg + torch.relu(0.05 - coeff).mean()
                loss = loss + 0.01 * diffusion_reg

                # Scale loss for gradient accumulation
                grad_accum = cfg.get("grad_accum_steps", 2)
                loss = loss / grad_accum

            # ── Backward pass ────────────────────────────────────────────
            scaler.scale(loss).backward()

            # ── Optimizer step (every grad_accum micro-batches) ──────────
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                # Gradient clipping at norm=1.0 prevents exploding gradients,
                # which are a real risk with forward Euler integration
                # (the PDE can amplify gradients across many time steps).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            # ── Telemetry logging ────────────────────────────────────────
            if is_log_step:
                t_delta = time.time() - batch_start
                it_s = 1.0 / t_delta if t_delta > 0 else 0.1
                eta_sec = (len(loader) - i) / it_s
                loss_val = float(loss.item() * grad_accum)  # Unscaled loss
                vram_val = (
                    torch.cuda.memory_reserved() / 1e6
                    if torch.cuda.is_available()
                    else 0
                )

                # ── Best model checkpoint ────────────────────────────────
                if global_step % 500 == 0 and loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(
                        {"model_state": model.state_dict(), "step": global_step,
                         "best_loss": best_loss},
                        MODEL_FILE.replace(".pth", "_best.pth"),
                    )

                # ── Regular checkpoint ───────────────────────────────────
                if cfg.get("save_now") or global_step % 1000 == 0:
                    torch.save(
                        {"model_state": model.state_dict(),
                         "optimizer_state": opt.state_dict(),
                         "scaler_state": scaler.state_dict(),
                         "step": global_step, "best_loss": best_loss},
                        MODEL_FILE,
                    )
                    if cfg.get("save_now"):
                        cfg["save_now"] = False
                        atomic_write(cfg, CONFIG_FILE)

                # ── Update rolling statistics ────────────────────────────
                stats["step"].append(global_step)
                stats["loss"].append(loss_val)
                stats["vram"].append(vram_val)
                stats["it_s"].append(it_s)
                stats["lr"].append(current_lr)
                stats["temp"].append(cfg["temperature"])
                stats["penalty"].append(cfg["repetition_penalty"])
                stats["avg_steps"].append(float(avg_steps))
                # Keep only last 1000 points to bound file size
                for k in stats.keys():
                    stats[k] = stats[k][-1000:]

                # ── Generate sample text for monitoring ──────────────────
                sample = generate_text(model, ds.tokenizer, cfg, max_tokens=30)

                # ── Embedding weight histogram ───────────────────────────
                # Tracking the distribution of embedding weights over time
                # reveals training health: a collapsing or exploding distribution
                # is a strong signal of instability.
                w_vals = model.embedding.weight.detach().flatten().cpu().float().numpy()
                counts, bins = np.histogram(w_vals, bins=30, range=(-0.2, 0.2))

                # ── Write telemetry packet ───────────────────────────────
                log_packet = {
                    "epoch": global_epoch,
                    "step": global_step,
                    "loss": loss_val,
                    "it_s": it_s,
                    "eta": eta_sec,
                    "vram": vram_val,
                    "sample": sample,
                    "waves": [w[0].tolist() for w in waves],
                    "w_hist": counts.tolist(),
                    "w_bins": bins.tolist(),
                    "timestamp": time.time(),
                    "avg_steps": float(avg_steps),
                }
                atomic_write(log_packet, LOG_FILE)
                atomic_write(stats, STATS_FILE)

        else:
            # The for-else construct: this block runs when the for loop
            # completes naturally (not via break). It means we finished
            # an entire epoch without a DataLoader rebuild.
            global_epoch += 1


if __name__ == "__main__":
    train()