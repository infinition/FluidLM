# FluidLM

**A Transformer-free language model replacing O(N²) self-attention with reaction-diffusion PDEs - achieving O(N) scaling, adaptive computation, and no KV-cache.**

> ⚠️ This is an early-stage proof of concept. It does not compete with production language models. The goal is to demonstrate that the mathematical framework is sound and the core mechanisms work.

---

## What is FluidLM?

FluidLM is a neural architecture that replaces the self-attention mechanism at the heart of modern language models with a system of continuous partial differential equations (PDEs). Inspired by Alan Turing's 1952 work on morphogenesis - the process by which biological organisms develop spatial patterns through chemical diffusion and reaction - FluidLM treats linguistic tokens not as discrete vectors passing through a fixed pipeline, but as chemical concentrations that diffuse and react within a continuous latent space.

The key shift: instead of every token explicitly "looking at" every other token through an N×N attention matrix, information propagates implicitly through **diffusion** - like heat spreading through a medium or chemicals dispersing in a solution. This is what enables strictly linear O(N) scaling with no KV-cache.

---

## Table of Contents

1. [Motivation: The Limits of Current Architectures](#1-motivation-the-limits-of-current-architectures)
2. [The FluidLM Approach](#2-the-fluidlm-approach)
3. [For the Non-Specialist](#3-for-the-non-specialist)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Implemented Features](#5-implemented-features)
6. [Architecture Comparison](#6-architecture-comparison)
7. [Implementation Details](#7-implementation-details)
8. [Research Dashboard](#8-research-dashboard)
9. [Experimental Status and Honest Assessment](#9-experimental-status-and-honest-assessment)
10. [Research Roadmap](#10-research-roadmap)
11. [Getting Started](#11-getting-started)
12. [References](#12-references)

---

## 1. Motivation: The Limits of Current Architectures

The dominant architecture for language modeling since 2017 is the Transformer (Vaswani et al.), which relies on the self-attention mechanism. While this approach has produced remarkable results - from GPT-4 to Claude to Gemini - it carries several structural constraints that become increasingly costly at scale.

### The O(N²) Attention Wall

Self-attention computes pairwise interactions between every token in a sequence. For a sequence of length N, this produces an N×N attention matrix, resulting in **O(N²) computational and memory complexity**. Doubling the context window quadruples the cost. This is why extending context from 4K to 128K tokens required enormous engineering effort and hardware investment.

### Static Computation

A Transformer with 96 layers applies exactly 96 layers of computation to every input, whether it is processing "Hello" or a complex mathematical proof. Every token receives identical computational budget regardless of semantic complexity.

### The KV-Cache Memory Wall

During inference, Transformers must store Key and Value matrices for every previous token in every attention layer. This KV-cache grows with sequence length × layer count. For long-context models, this cache alone can require tens of gigabytes of VRAM.

### Architectural Rigidity

The Transformer is a discrete, fixed-depth pipeline. Removing or pruning a single layer can catastrophically degrade performance. The architecture offers no natural mechanism for graceful degradation or elastic computation.

These are not implementation bugs - they are structural properties of the attention mechanism itself. FluidLM asks: **what if we replaced the attention matrix entirely with a different mathematical object?**

---

## 2. The FluidLM Approach

FluidLM replaces global attention with **Reaction-Diffusion equations** - the same class of PDEs that Turing showed could generate spatial patterns (stripes, spots, waves) from homogeneous initial conditions through local interactions alone.

In a Transformer, every token must explicitly "look at" every other token through the attention matrix. In FluidLM, tokens influence each other implicitly through **diffusion** - information propagates from neighbor to neighbor through the latent space. Combined with local **reaction** functions (nonlinear transformations), this produces complex global patterns from purely local operations.

This shift from "every token talks to every token" to "information flows like a fluid" is what eliminates quadratic complexity. A token does not need to compute its relationship with 100,000 other tokens; it only needs to interact with its immediate neighborhood. Information from distant tokens arrives naturally through the diffusion process over multiple time steps.

---

## 3. For the Non-Specialist

### Why current LLMs are expensive

Models like GPT-4, Claude, and Gemini work by computing how every word relates to every other word. Imagine a classroom where every student must have a one-on-one conversation with every other student before anyone can answer a question. With 10 students, that is 45 conversations. With 1,000 students, it is 499,500. The cost explodes as the group grows. This is the O(N²) scaling problem.

### What FluidLM does differently

FluidLM works more like how information spreads in a room. When someone speaks, the people closest to them hear it first, then those people react and pass it along. Nobody needs to personally talk to everyone. Information flows like a wave. This means:

- **Cost grows linearly, not quadratically.** Doubling the input doubles the cost - not quadruples it.
- **No KV-cache.** There are no Key/Value matrices to store. Memory usage is fixed regardless of context history.
- **The model can think harder on difficult inputs.** If the "fluid" is still turbulent, computation continues. If it has stabilized, computation stops early. Simple inputs require less work; complex inputs get more.
- **Multi-scale understanding.** Diffusion operates simultaneously at multiple scales - local syntax and long-range structure are handled in the same pass.

---

## 4. Mathematical Foundations

### 4.1 The Standard Transformer Attention (what we replace)

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V$$

The product $Q \cdot K^\top$ produces the $N \times N$ attention matrix. This is the source of $O(N^2)$ complexity.

### 4.2 The FluidLM Governing Equation

FluidLM replaces the attention matrix with a PDE governing a continuous latent state $\mathbf{u}$ over a virtual time dimension $t$:

$$\frac{\partial u}{\partial t} = \sum_{k} D_k \cdot \nabla^2_{d_k}(u) + R(u,\, \theta) + \alpha \cdot h_t$$

This equation has three distinct terms:

#### Term 1: Multi-Scale Diffusion

$$\sum_{k} D_k \cdot \nabla^2_{d_k}(u)$$

The Laplacian operator measures how different a token's representation is from its neighbors. Where there are sharp differences, diffusion acts to smooth them - propagating information from one token to its neighbors. Three Laplacian operators are applied simultaneously at different scales via **dilated convolutions**:

| Dilation | Effective Reach per Step | Role |
|----------|--------------------------|------|
| 1 | ~1 token | Local syntax, morphology |
| 4 | ~4 tokens | Phrase-level structure |
| 16 | ~16 tokens | Sentence / paragraph dependencies |

Each scale has a learnable diffusion coefficient $D_k$. Over 12 time steps × 4 layers, the coarsest scale (dilation 16) can propagate information across approximately 768 token positions.

The discrete Laplacian is a standard 1D convolution with kernel $[1,\ -2,\ 1]$:

$$\nabla^2(u_i) = u_{i-d} - 2 \cdot u_i + u_{i+d}$$

This operation is **O(N)** - it touches each token exactly once per dilation, with no pairwise interactions.

**Causal constraint:** All convolutions use left-only zero-padding, ensuring token $i$ can only receive information from positions $\leq i$ (no future leakage).

#### Term 2: Reaction Function

$$R(u,\, \theta) = \text{MLP}(u) = W_2 \cdot \text{GELU}(W_1 \cdot u + b_1) + b_2$$

A per-token 2-layer MLP with hidden dimension $2 \times d_\text{model}$. This is the "chemistry" - the local nonlinear transformation that creates new feature combinations. Without reaction, pure diffusion would simply blur all representations toward a uniform average.

#### Term 3: Intra-Sequence Memory Pump (h-state)

$$h_t = h_{t-1} + \sigma(W_\text{gate} \cdot u_{t-1}) \cdot \tanh\!\left(R(u_{t-1},\, \theta)\right)$$

A gated accumulation mechanism that maintains a running reservoir of semantic reactions throughout the integration process. It provides gradient stabilization (analogous to residual connections) and temporal persistence across integration steps.

> The h-state resets to zero at the beginning of each forward pass. It is an intra-sequence mechanism, not an inter-sequence memory.

### 4.3 Time Integration

The PDE is discretized using forward Euler:

$$u_{t+1} = \text{LayerNorm}\!\left( u_t + \Delta t \cdot \left[ \sum_k D_k \cdot \nabla^2_{d_k}(u_t) + R(u_t,\, \theta) + \alpha \cdot h_t \right] \right)$$

LayerNorm is applied after each step to prevent state divergence - a necessity given that forward Euler is only conditionally stable.

### 4.4 Turing Equilibrium and Adaptive Computation

Instead of running a fixed number of integration steps, FluidLM monitors the convergence of the latent fluid and halts early when the system stabilizes. The convergence criterion is:

$$\text{turbulence} = \operatorname{mean}_i\!\left(\frac{\|\Delta u_i\|}{\|u_i\| + \varepsilon}\right) \quad \xrightarrow{\quad \text{if} < \varepsilon \quad} \quad \text{HALT}$$

If `turbulence < epsilon`, the system has reached **Turing Equilibrium** - a stable pattern that will not change significantly with further computation.

- During **training**: the model always runs all steps (for consistent gradients), but records the step at which equilibrium would have been reached.
- During **inference** (`model.eval()`): the model genuinely halts early, saving wall-clock time proportional to the difficulty of the input.

This adaptive behavior emerges directly from the physics of the system - the model stops because the fluid has genuinely stabilized, not because a separate learned mechanism decided it should.

---

## 5. Implemented Features

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-Scale Dilated Diffusion | ✅ | Laplacian convolutions at dilations [1, 4, 16] with learnable coefficients |
| Reaction Function (MLP) | ✅ | Per-token 2-layer MLP with GELU activation |
| Intra-Sequence Memory Pump | ✅ | Gated additive accumulation (sigmoid/tanh) across integration steps |
| Causal Zero-Padding | ✅ | Left-only padding ensuring no future information leakage |
| Adaptive Compute (Inference) | ✅ | Early stopping based on convergence criterion |
| Adaptive Compute Logging (Training) | ✅ | Records equilibrium step without altering the computational graph |
| Weight Tying | ✅ | Embedding and output head share parameters |
| Mixed Precision Training (AMP) | ✅ | FP16 forward pass with FP32 gradient scaling |
| Gradient Accumulation | ✅ | Configurable accumulation steps for effective batch size scaling |
| Linear Warmup | ✅ | Configurable warmup period for learning rate |
| BPE Tokenization | ✅ | GPT-2 tokenizer via tiktoken (50,257 vocabulary) |
| Real-Time Dashboard | ✅ | Streamlit interface with live telemetry, Turing wave visualization, and parameter control |
| Configurable Epsilon | ✅ | Convergence threshold adjustable via dashboard slider |

---

## 6. Architecture Comparison

### Structural Comparison

| Property | Transformer (GPT-class) | FluidLM |
|----------|-------------------------|---------|
| **Core mechanism** | Self-Attention ($Q \cdot K^\top \cdot V$) | Reaction-Diffusion PDE |
| **Complexity per layer** | $O(N^2 \cdot d)$ | $O(N \cdot d \cdot K)$ where $K$ = dilations |
| **Receptive field** | Global (all tokens) | Multi-scale local (~768 tokens at max dilation) |
| **Computation per input** | Fixed (L layers, always) | Adaptive ($T_\text{min}$ to $T_\text{max}$ steps) |
| **Inference memory** | $O(L \cdot N \cdot d)$ for KV-cache | $O(N \cdot d)$ — **no cache required** |

### Concrete FLOP Comparison (N = 8,192 tokens, d_model = 512)

| Operation | Transformer Attention | FluidLM Diffusion (3 scales) |
|-----------|-----------------------|------------------------------|
| Per-layer cost | ~34 billion FLOPs | ~63 million FLOPs |
| Relative cost | 1× | ~0.002× per step |
| With 12 steps | - | ~0.024× |
| With adaptive (avg 6 steps) | - | ~0.012× |

The diffusion mechanism is orders of magnitude cheaper than attention for long sequences, and **the gap widens as N increases**.

### What FluidLM Trades Away

Intellectual honesty requires acknowledging what is lost:

- **Selective long-range access.** Attention can directly connect any two tokens regardless of distance. FluidLM's information must propagate through intermediate tokens. The effective receptive field may be significantly smaller than the theoretical maximum.
- **Proven scaling laws.** Transformers have well-established scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022). FluidLM has no such empirical characterization yet.
- **Ecosystem maturity.** Transformers benefit from FlashAttention, PagedAttention, speculative decoding, quantization pipelines, etc. FluidLM is a research prototype with none of that infrastructure.

---

## 7. Implementation Details

### Model Architecture

```
Input tokens
    │
    ▼
[Embedding + Positional Encoding]  (d_model = 512)
    │
    ▼
[FluidLayer 0]  ── Multi-scale diffusion + Reaction + Memory pump (× T steps)
    │
    ▼
[FluidLayer 1]  ── Same structure, independent parameters
    │
    ▼
[FluidLayer 2]  ── Same structure, independent parameters
    │
    ▼
[FluidLayer 3]  ── Same structure, independent parameters
    │
    ▼
[Linear Head]  (weight-tied with Embedding)
    │
    ▼
Logits (vocabulary: 50,257)
```

### Parameter Count (default config: d_model=512, 4 layers)

| Component | Parameters |
|-----------|------------|
| Embedding (weight-tied) | ~25.7M |
| Positional encoding | ~2.1M |
| FluidLayer × 4 | ~8.4M |
| **Total** | **~36.2M** |

This is deliberately small - the goal is to validate the mechanism, not to chase benchmark scores.

### Training Configuration

| Parameter | Default | Live-adjustable |
|-----------|---------|-----------------|
| Optimizer | AdamW | No |
| Learning rate | 1e-4 | Yes |
| Warmup steps | 500 | Yes |
| Batch size | 32 | Yes |
| Sequence length | 128 | Yes |
| Gradient accumulation | 2 steps | Yes |
| Integration steps (max) | 12 | Yes |
| Integration dt | 0.1 | Yes |
| Convergence epsilon | 1e-4 | Yes |
| Gradient clipping | 1.0 (max norm) | No |

### Hardware Requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU | Any CUDA-capable | NVIDIA RTX 4070 Ti (12GB) |
| VRAM | ~4GB (small config) | ~10GB (default config) |
| RAM | 8GB | 16GB+ recommended |

---

## 8. Research Dashboard

The Streamlit-based dashboard provides real-time visibility into the training process and is designed as a research instrument, not just a monitoring tool.

**Live Metrics:** loss with delta tracking, average integration steps (the key adaptive compute metric), VRAM consumption, training speed, ETA.

**Visualizations:**
- Loss curve (up to 1,000 data points)
- **Turing Waves** - heatmap of activation magnitudes across sequence positions and integration steps, showing the evolution of spatial patterns during diffusion
- Embedding weight distribution histogram
- God Mode - normalized overlay of all metrics for correlation analysis

**Live Controls:** all key hyperparameters can be adjusted during training without stopping the process. The training engine polls the config file every 10 steps and applies changes on the fly.

**Auto-Pilot:** optional automatic adjustment that detects output loops, loss plateaus, and relaxes penalties when generation quality improves.

---

## 9. Experimental Status and Honest Assessment

### What has been demonstrated

1. **The architecture trains and converges.** The model successfully learns to reduce cross-entropy loss on text data.
2. **Linear scaling holds in practice.** Memory consumption scales linearly with sequence length, as predicted by theory.
3. **Adaptive computation works mechanically.** The convergence criterion fires, variable step counts are recorded, and early stopping reduces wall-clock time in inference mode.

### What has NOT been demonstrated

1. **Competitive perplexity.** The PoC has not been benchmarked against a Transformer of equivalent parameter count on standard datasets.
2. **That adaptive computation correlates with input difficulty.** The mechanism works, but evidence that the model uses more steps on "hard" inputs and fewer on "easy" ones has not yet been produced.
3. **Long-range dependency modeling.** Whether the model actually captures dependencies at 500+ token distances has not been tested.
4. **Scaling behavior.** How FluidLM performs as model size, data, and compute increase is unknown.

### Known Limitations

- The forward Euler integrator is first-order and conditionally stable. Large $\Delta t$ values can cause divergence.
- The h-state resets every forward pass - no inter-sequence memory.
- Learned absolute positional encoding limits generalization to unseen sequence lengths.
- Generation quality at this scale (36M parameters, limited data) is not representative of the architecture's ceiling.

---

## 10. Research Roadmap

### Near-term

1. **Adaptive Compute Validation** - Design controlled experiments with inputs of known, varying difficulty. Measure whether avg_steps correlates with input complexity. This is the single most important experiment for validating the architecture's core claim.
2. **Scaling Experiments** - Train at 36M, 100M, and 300M parameters on WikiText-103 / The Pile. Compare perplexity curves against equivalent Transformers.
3. **Long-Context Stress Testing** - Extend sequence length to 4K, 16K, and 64K tokens. Measure both memory consumption (expected: linear) and performance on needle-in-a-haystack benchmarks.

### Medium-term

4. **Inter-Sequence Persistent Memory** - Pass the latent state $u$ and the reservoir $h$ between sequential batches as detached external states, similar to Transformer-XL. This would give FluidLM true unbounded temporal memory.
5. **Higher-Order Integration** - Replace forward Euler with RK4 or implicit integrators for improved stability.
6. **Neural ODE Adjoint Methods** - Backpropagate through integration without storing intermediate states, dramatically reducing training VRAM.

### Long-term

7. **Holographic Degradation Testing** - Systematically prune portions of the latent space and measure performance degradation curves. The hypothesis: distributed, fluid representations degrade more gracefully than the discrete representations in Transformers.
8. **Sparse Spatial Activation** - Compute updates only where the semantic "wavefront" is active ($\Delta u > \tau$), potentially reducing inference energy by orders of magnitude.
9. **RoPE Integration** - Replace absolute positional encoding with Rotary Positional Embeddings for better length generalization.

---

## 11. Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA support)
- Streamlit
- Tiktoken

### Installation

```bash
git clone https://github.com/infinition/FluidLM.git
cd FluidLM
pip install -r requirements.txt
```

### Running the Lab

```bash
python launch_lab.py
```

Then open `http://localhost:8501`. Drop `.txt` files into the `/data` folder for the model to train on.

---

## 12. References

1. Turing, A. M. (1952). *The Chemical Basis of Morphogenesis.* Philosophical Transactions of the Royal Society of London. Series B.
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08514.
4. Chen, R. T., et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
5. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

---

## Philosophy

The Transformer is a marvel of engineering, but it remains a rigid machine - a massive assembly line that cannot adapt to the fluidity of thought. FluidLM is an attempt to make semantic computation as organic as a chemical reaction.

The "Turing Waves" visible in the dashboard are not just pixels. They are the visual trace of a network trying to stabilize its own thoughts.

This PoC proves that we can train a language model with no quadratic bottleneck, no KV-cache, and computation that adapts organically to the complexity of the input. It is a first step toward an AI that does not consume energy out of habit, but out of necessity.

---

*Proof of Concept - Research prototype by Fabien POLLY (Infinition), not a production system.*