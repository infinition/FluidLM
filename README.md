# FluidLM

<img width="1950" height="674" alt="fluidlm_banner" src="https://github.com/user-attachments/assets/d17ee5df-5c7d-423c-a276-1ff4528b9614" />




# FluidLM

**A Transformer-free language model replacing O(N^2) self-attention with reaction-diffusion PDEs -- achieving O(N) scaling, adaptive computation, and no KV-cache.**

> Early-stage proof of concept. The goal is to demonstrate that the mathematical framework is sound and the core mechanisms work -- not to compete with production LLMs.

---

## What is FluidLM?

FluidLM replaces the self-attention mechanism with a system of continuous partial differential equations (PDEs). Inspired by Alan Turing's 1952 work on morphogenesis, it treats tokens as chemical concentrations that diffuse and react in a latent space.

Instead of every token explicitly "looking at" every other token through an N x N matrix, information propagates through:

- **Local diffusion** -- a multi-scale Laplacian (dilations 1, 4, 16) that spreads information between neighbors, like heat through a medium.
- **Selective State Space (Mamba)** -- a content-based temporal routing mechanism that replaces the previous fixed CausalLongConv. Each token selectively retains or forgets information based on its content, providing the content-aware long-range mixing that pure diffusion lacks.
- **Reaction MLP (SwiGLU)** -- the nonlinear component where all semantic capacity is concentrated.
- **Global memory pump** -- a reservoir `h` of shape `(B, D)` with learned forget gate that accumulates a sequence summary.

This shift from "every token talks to every token" to "information flows like a fluid" eliminates quadratic complexity.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The FluidLM Approach](#2-the-fluidlm-approach)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Architecture V4.5.0](#4-architecture-v450)
5. [Implemented Features](#5-implemented-features)
6. [Architecture Comparison](#6-architecture-comparison)
7. [Research Dashboard](#7-research-dashboard)
8. [Experimental Status](#8-experimental-status)
9. [Research Roadmap](#9-research-roadmap)
10. [Getting Started](#10-getting-started)
11. [Version History](#11-version-history)
12. [References](#12-references)

---

## 1. Motivation

### The O(N^2) Attention Wall

Self-attention computes pairwise interactions between every token. For a sequence of N tokens, this produces an N x N matrix -- **O(N^2) computational and memory complexity**. Doubling context quadruples cost.

### Static Computation

A Transformer applies exactly L layers to every input, whether it is processing "Hello" or a complex mathematical proof.

### The KV-Cache Memory Wall

During inference, Transformers store Key and Value matrices for every past token in every layer. This cache grows with sequence length x layer count -- tens of gigabytes for long-context models.

### GPU Dependency

The combination of O(N^2) compute, large KV-caches, and irregular memory access makes Transformers fundamentally GPU-dependent. Their patterns are poorly suited for CPUs, NPUs, and embedded processors.

FluidLM asks: **what if we replaced the attention matrix entirely with a different mathematical object?**

---

## 2. The FluidLM Approach

Two complementary propagation mechanisms replace global attention:

**Local diffusion** -- the multi-scale Laplacian propagates information from neighbor to neighbor at three spatial scales (dilations 1, 4, 16). Each token influences its immediate neighborhood; patterns emerge globally through repeated application.

**Selective State Space (Mamba)** -- V4.5.0 replaces the fixed CausalLongConv filter with a Mamba-style selective SSM. This is the key architectural upgrade: while the Laplacian provides position-based spatial mixing, the SSM provides content-based temporal routing. The SSM's input-dependent matrices (A, B, C) allow the model to selectively retain or discard information based on what the token contains -- the missing capability that caused the previous loss plateau at ~6.0.

**Memory pump** -- a global reservoir `h` of shape `(B, D)` that accumulates a sequence summary at each integration step, then broadcasts it uniformly to all positions. The **Forget Gate** (`decay = sigmoid(decay_param)`) introduces learned viscosity: the model learns what persists in the reservoir and what dissipates. The gate is h-aware (`sigmoid(Wx + Uh)`) to avoid accumulating what is already present.

**Laplacian smoothness regularizer** (V4.4.8, ported from FluidWorld) -- a grad_loss term computed on the final hidden representation penalizes high-frequency noise along the sequence dimension. This acts as an implicit spatial regularizer that improves autoregressive generation stability.

---

## 3. Mathematical Foundations

### The Standard Transformer (what we replace)

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V$$

The product $Q \cdot K^\top$ produces the $N \times N$ attention matrix -- the source of $O(N^2)$ complexity.

### The FluidLM Governing Equation

$$\frac{\partial u}{\partial t} = \underbrace{\sum_{k} D_k \cdot \nabla^2_{d_k}(u)}_{\text{local diffusion}} + \underbrace{\text{SelectiveSSM}(u)}_{\text{content-based routing}} + \underbrace{R(u,\, \theta)}_{\text{reaction (SwiGLU)}} + \underbrace{\alpha \cdot h}_{\text{global memory}}$$

#### Term 1: Multi-Scale Local Diffusion

The discrete Laplacian $[1, -2, 1]$ applied at three dilation levels:

| Dilation | Reach per step | Role |
|----------|----------------|------|
| 1 | ~1 token | Local syntax, morphology |
| 4 | ~4 tokens | Phrase-level structure |
| 16 | ~16 tokens | Sentence / paragraph |

$O(N)$ per step, sequential memory access -- ideal for CPU SIMD.

#### Term 2: Selective State Space (Mamba)

$$h_t = \bar{A}_t \cdot h_{t-1} + \bar{B}_t \cdot x_t$$

$$y_t = C_t \cdot h_t + D \cdot x_t$$

Where $A$, $B$, $C$ are **input-dependent** (computed from the current token). This is mathematically a discretized ODE -- the same family as the PDE diffusion. The SSM selectively chooses what to remember and what to forget based on content, providing the content-based routing that pure diffusion lacks.

Pure PyTorch implementation (no custom CUDA kernels). $O(N \cdot d \cdot s)$ training, $O(d \cdot s)$ constant per-token inference. No KV-cache.

#### Term 3: Reaction Function (SwiGLU)

$$R(u,\, \theta) = \left(W_1 \cdot u \odot \sigma(W_g \cdot u)\right) \cdot W_2$$

SwiGLU (V4.5.0) replaces the previous GELU MLP. Proven by LLaMA/PaLM to improve language modeling quality. $\frac{8}{3}$ expansion ratio.

#### Term 4: Global Memory Pump + Forget Gate

$$\text{react\_summary} = \text{mean}_{L}(R(u)) \quad \in \mathbb{R}^{B \times D}$$

$$\text{gate} = \sigma(W_x \cdot \bar{u} + W_h \cdot h) \quad \in \mathbb{R}^{B \times D}$$

$$\text{decay} = \sigma(\text{decay\_param}) \quad \in (0,1)^{D} \quad \text{(learned viscosity)}$$

$$h \leftarrow \text{decay} \odot h + \text{gate} \cdot \tanh(\text{react\_summary})$$

$h$ is of shape $(B, D)$ -- global reservoir, $O(1)$ memory. Initialized at $\text{decay} \approx 0.97$.

#### Term 5: Laplacian Grad Loss (V4.4.8)

$$\mathcal{L}_{\text{grad}} = w_g \cdot \text{mean}\!\left(|\nabla^2_{1D}(z_{\text{final}})|\right)$$

Applied to the final hidden representation $z_{\text{final}}$ before the LM head. Penalizes second-order discontinuity along the sequence, encouraging smooth latent representations that degrade gracefully during autoregressive generation.

### Positional Encoding: Sinusoidal

Applied once at the FluidNet input (not at each integration step):

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Sinusoidal encoding gives a clean additive signal that propagates naturally through the PDE.

### Time Integration (Forward Euler)

$$u_{t+1} = \text{RMSNorm}\!\left( u_t + \Delta t \cdot \left[ \sum_k D_k \nabla^2_{d_k}(u_t) + \text{SSM}(u_t) + R(u_t,\theta) + \alpha \cdot h_t \right] \right)$$

$\Delta t$ is a per-layer learned parameter (`dt_gate`), initialized from the config `dt` value.

### Turing Equilibrium and Adaptive Computation

$$\text{turbulence} = \text{mean}\left(\frac{|u_{t+1} - u_t|}{|u_t| + \varepsilon}\right) \xrightarrow{< \varepsilon} \text{HALT}$$

During inference, the model halts early when the fluid stabilizes. During training, turbulence contributes to a differentiable regularization loss.

---

## 4. Architecture V4.5.0

```
Input tokens
    |
    v
[Embedding]  (d_model = 512)
    |
    v
[Sinusoidal PE]  (0 params, applied once)
    |
    v
[FluidLayer 0]  -- Diffusion (Laplacian x3 scales)
                 -- Selective SSM (Mamba, content-based)
                 -- Reaction SwiGLU (8/3 expansion)
                 -- Local Memory (causal avg pool + projection)
                 -- Memory Pump h (B,D) + Forget Gate (x T steps, T<=8)
    |
[FluidLayer 1..3]  (same structure, independent params)
    |
    v
[RMSNorm]
    |
    v
[Linear Head]  (weight-tied with Embedding)
    |
    v
Logits (vocab: 50,257)
```

### Parameter Count (d_model=512, 4 layers)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Embedding (weight-tied with head) | ~25.7M | Counts once |
| Sinusoidal PE | 0 | Buffer |
| Reaction SwiGLU 8/3 (x4 layers) | ~10.7M | Replaces GELU MLP |
| Selective SSM / Mamba (x4 layers) | ~3.4M | Replaces CausalLongConv |
| Memory Gate x + h (x4 layers) | ~2.1M | h-aware, (B,D) |
| Multi-Head LongConv (x4 layers) | ~1.2M | K=33/65/129/257 |
| Local Memory proj (x4 layers) | ~1.05M | Causal avg pool |
| Diffusion coefficients (x4 layers) | ~6K | |
| RMSNorm + dt_gate + alpha (x4 layers) | ~8K | |
| **Total V4.5.0** | **~44.2M** | |

---

## 5. Implemented Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-Scale Dilated Diffusion | Done | Dilations [1, 4, 16], learnable coefficients |
| Selective SSM (Mamba) | Done | Content-based routing, pure PyTorch, V4.5.0 |
| Multi-Head LongConv (K=33/65/129/257) | Done | 4 heads, multi-scale, V4.5.0 |
| Reaction SwiGLU | Done | 8/3 expansion ratio, V4.5.0 |
| RMSNorm | Done | Replaces LayerNorm, V4.5.0 |
| Sinusoidal PE | Done | Applied once at input |
| Memory Pump (B, D), h-aware gate | Done | O(1), gate = sigmoid(Wx + Uh) |
| Local Memory (causal avg pool) | Done | 1D low-res reinjection |
| Forget Gate (learned viscosity) | Done | decay = sigmoid(decay_param), init ~0.97 |
| Layer-wise gradient clipping | Done | Embedding / each FluidLayer / Head |
| Causal Zero-Padding | Done | No future leakage |
| Adaptive Compute (inference) | Done | Early stop at Turing equilibrium |
| Differentiable turbulence (eq_loss) | Done | eq_weight configurable |
| Laplacian grad_loss | Done | Spatial smoothness regularizer, V4.4.8 |
| Curriculum scheduler | Done | Linear ramp of regularizer weights, V4.4.8 |
| Cosine LR Schedule | Done | Warmup -> cosine decay |
| Weight Tying | Done | Embedding <-> head |
| Mixed Precision | Done | FP16 forward, FP32 grad scaling |
| BPE Tokenization | Done | GPT-2 via tiktoken (50,257 vocab) |
| Real-Time Dashboard | Done | Streamlit with health monitoring |

---

## 6. Architecture Comparison

| Property | Transformer (GPT-class) | FluidLM V4.5.0 |
|----------|-------------------------|-----------------|
| Core mechanism | Self-Attention (Q*K^T*V) | Reaction-Diffusion PDE + Selective SSM |
| Complexity per layer | O(N^2 * d) | O(N * d * s) |
| Long-range mixing | Attention -- O(N^2), content-based | SSM -- O(N), content-based |
| Positional encoding | Learned / RoPE | Sinusoidal (input only) |
| Computation | Fixed (L layers always) | Adaptive (T_min to T_max) |
| Inference memory | O(L * N * d) KV-cache | O(N * d) -- no cache |
| Memory pump | -- | h global (B,D), O(1) |
| Spatial regularizer | None | Laplacian grad_loss |
| Minimum deployment | GPU + Python + framework | Python + PyTorch |

### What FluidLM Trades Away

- **Proven scaling laws.** Not empirically characterized beyond 44M params.
- **Ecosystem maturity.** No FlashAttention, speculative decoding, or equivalent.
- **Production validation.** Research prototype, not battle-tested.

---

## 7. Research Dashboard

The Streamlit dashboard provides real-time visibility into training.

```bash
python launch_lab.py
# -> http://localhost:8501
```

**Header bar:** step, loss (with delta), adaptive effort (avg_steps/max), speed, VRAM, ETA, main CE, eq loss, turbulence, actual cosine-decayed LR, tokens seen, epoch.

**Loss curve:** total loss, main CE, eq loss, grad_loss, actual LR on secondary axis.

**Turing Waves heatmap:** activation magnitude across positions and integration steps. Toggle "Norm rows" for per-row normalization.

**Effort trend:** avg_steps over time with turbulence on secondary axis and epsilon threshold line.

**Sidebar dt:** shows the actual learned dt per layer. The slider sets the initialization point; the model adapts from there.

**Auto-Pilot:** automatic loop detection and plateau-based LR decay.

---

## 8. Experimental Status

### What Has Been Demonstrated

1. **The architecture trains and converges.** Loss ~10.8 -> 5.85 on TinyStories.
2. **Linear memory scaling.** VRAM scales linearly with sequence length.
3. **Adaptive computation works mechanically.** Convergence criterion fires correctly at inference.
4. **O(1) memory pump confirmed.** 128x less VRAM than the (B,L,D) variant.

### What Has NOT Been Demonstrated

1. Competitive perplexity against equivalent Transformers.
2. Adaptive compute correlating with input difficulty (formal proof pending).
3. Scaling behavior at 100M+ parameters.
4. Whether V4.5.0 (SSM + SwiGLU) breaks through the ~6.0 loss plateau.

### Known Limitations

- Forward Euler is first-order and conditionally stable (RK4 planned).
- h-state resets every forward pass -- no inter-sequence memory.
- Generation quality at ~44M params is not representative of the architecture's ceiling.
---

## 9. Research Roadmap

```
PHASE 1 -- Core Validation
[done] V4.2    PoC architecture + TinyStories (loss ~10.8 -> 5.85)
[done] V4.3    RoPE + MLP 4x
[done] V4.4.0  CausalLongConv K=65, sinusoidal PE, cosine LR
[done] V4.4.1  Bug fixes (diff_turbulences, dt slider, LongConv init, h-aware gate)
[done] V4.4.2  h-state (B,D) O(1) restored
[done] V4.4.4  Forget Gate (learned viscosity), layer-wise grad clip
[done] V4.4.7  Desaturated gates, local memory
[done] V4.4.8  FluidWorld cross-pollination (grad_loss, curriculum scheduler)
[done] V4.5.0  Selective SSM (Mamba), SwiGLU, Multi-Head LongConv, RMSNorm
[wip]  -----   Training V4.5.0 -- evaluate loss plateau breakthrough

PHASE 2 -- Scaling and Validation
[next] Step 1  Formal adaptive compute demonstration
[next] Step 2  Scaling 100M / 300M vs Transformers
[next] Step 3  Needle-in-a-Haystack context validation (4K / 16K / 64K)

PHASE 3 -- Architecture Modernization
[next] Step 4  Persistent memory (pass-through h-state between segments)
[next] Step 5  Extended dilations [1, 4, 16, 64, 256, 1024]
[next] Step 6  Adjoint Methods (backprop without VRAM explosion)
[next] Step 7  RK4 integrator

```

---

## 10. Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CUDA for training)
- Streamlit, Tiktoken

### Installation

```bash
git clone https://github.com/infinition/FluidLM.git
cd FluidLM
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Training

```bash
# Fresh start (recommended when changing major version)
rm -f config.json
rm -rf ./training/

# Launch training + dashboard
python launch_lab.py
# -> http://localhost:8501
```

Drop `.txt` files into the `/data/` folder. The model trains on all `.txt` files found there using BPE tokenization (GPT-2 vocabulary, 50,257 tokens).

### Configuration

Key parameters in `config.json` (auto-created on first launch):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Base learning rate (cosine-decayed) |
| `batch_size` | 32 | Training batch size |
| `seq_len` | 256 | Sequence length |
| `d_model` | 512 | Model dimension |
| `t_steps` | 8 | Max integration steps per layer |
| `dt` | 0.1 | Initial dt (learned via dt_gate) |
| `eq_weight` | 0.01 | Turbulence regularization weight |
| `grad_loss_weight` | 0.005 | Laplacian smoothness weight |
| `curriculum_steps` | 5000 | Steps to ramp regularizers (0 = disabled) |
| `gate_reg_weight` | 0.08 | Gate saturation penalty |
| `epsilon` | 0.05 | Adaptive stopping threshold |
| `warmup_steps` | 500 | LR warmup steps |
| `total_steps` | 50000 | Total training steps |
| `grad_accum_steps` | 4 | Gradient accumulation steps |

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| V4.2.0 | 2025 | Initial architecture, forward Euler, learned pos_emb |
| V4.3.0 | 2025 | RoPE, MLP 4x |
| V4.4.0 | 2026 | CausalLongConv K=65, sinusoidal PE, cosine LR, memory pump (B,L,D) |
| V4.4.1-3 | 2026 | Bug fixes: diff_turbulences, dt slider, LongConv init, h-state O(1) restored |
| V4.4.4-7 | 2026 | Forget Gate, layer-wise grad clip, desaturated gates, local memory |
| V4.4.8 | 2026 | Laplacian grad_loss + curriculum scheduler (ported from FluidWorld) |
| **V4.5.0** | **2026** | **Selective SSM (Mamba), SwiGLU, Multi-Head LongConv, RMSNorm** |

---

## 12. References

1. Turing, A. M. (1952). *The Chemical Basis of Morphogenesis.* Phil. Trans. R. Soc. London B.
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08514.
4. Chen, R. T., et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
5. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
6. Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.

---

*Research prototype by Fabien POLLY (Infinition). Not a production system.*
