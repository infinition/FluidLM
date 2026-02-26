#!/usr/bin/env python3
# =============================================================================
# FluidLM Research Dashboard 
# =============================================================================
#
# Streamlit-based real-time research interface for the FluidLM training engine.
#
# This dashboard serves dual purposes:
#   1. MONITORING - live telemetry, loss curves, Turing wave visualization,
#      embedding health tracking, adaptive compute metrics
#   2. CONTROL - live hyperparameter tuning without restarting training
#
# Communication with the training engine is via three JSON files:
#   - config.json:        bidirectional config (dashboard ↔ engine)
#   - live_logs.json:     per-step telemetry snapshot (engine → dashboard)
#   - training_stats.json: rolling history for plots (engine → dashboard)
#
# Design philosophy: this is a research instrument, not a production UI.
# Every control has a tooltip explaining what it does, when to change it,
# and what the tradeoffs are. The goal is to make the dashboard self-
# documenting so a collaborator can sit down and understand the system
# without reading the codebase.
#
# Author: Fabien POLLY (Infinition)
# =============================================================================

import streamlit as st
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
from datetime import datetime

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FluidLM Research Lab",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── File paths (must match train_engine.py) ──────────────────────────────────
CFG_F = "config.json"
LOG_F = "live_logs.json"
STAT_F = "training_stats.json"


# =============================================================================
# Utility Functions
# =============================================================================

def atomic_write(data, path, retries=3):
    """
    Atomic JSON write - see train_engine.py for full documentation.
    Duplicated here because the dashboard runs in a separate process
    and we avoid importing from the engine (which would trigger CUDA init).
    """
    for attempt in range(retries):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
            return True
        except Exception:
            time.sleep(0.1 * (attempt + 1))
    return False


def get_cfg():
    """Load current config, returning empty dict on failure."""
    if not os.path.exists(CFG_F):
        return {}
    try:
        with open(CFG_F, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def set_cfg(c):
    """Write config atomically."""
    atomic_write(c, CFG_F)


def detect_loop(text, min_len=10):
    """
    Heuristic loop detector for generated text samples.

    Checks two patterns:
      1. Single-character repetition: "aaaaaaa" (6+ identical chars in a row)
      2. N-gram over-representation: if any 2/3/4-gram appears more than
         len(text) / n / 2 times, the text is likely stuck in a loop.

    This is used by the auto-pilot to detect degenerate outputs and
    automatically increase repetition penalty and temperature.

    Known limitation: this can false-positive on naturally repetitive text
    (e.g., poetry with refrains). For a PoC, this is acceptable.
    """
    if not text or len(text) < min_len:
        return False
    # Check for single-character runs
    if any(len(set(text[i:i + 6])) == 1 for i in range(len(text) - 5)):
        return True
    # Check for n-gram over-representation
    for n in range(2, 5):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        if ngrams:
            most_common = Counter(ngrams).most_common(1)[0]
            if most_common[1] > len(text) / n / 2:
                return True
    return False


def format_eta(seconds):
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m"
    return f"{int(m)}m {int(s)}s"


def norm_data(arr):
    """
    Min-max normalize an array to [0, 100] for God Mode overlay.
    When all values are identical, returns 50.0 (midpoint) to avoid div/0.
    """
    if not arr:
        return []
    mi, ma = min(arr), max(arr)
    if ma == mi:
        return [50.0] * len(arr)
    return [((x - mi) / (ma - mi)) * 100 for x in arr]


# =============================================================================
# Custom CSS - Professional dark theme styling
# =============================================================================

st.markdown("""
<style>
    /* ── Status badges ─────────────────────────────────────────────────── */
    .badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        color: white;
        margin-bottom: 12px;
        display: inline-block;
    }
    .bg-run { background: linear-gradient(135deg, #28a745, #20c997); }
    .bg-pause { background: linear-gradient(135deg, #ffc107, #fd7e14); color: #1a1a2e; }
    .bg-chat { background: linear-gradient(135deg, #17a2b8, #6f42c1); }

    /* ── Equation blocks ───────────────────────────────────────────────── */
    .equation-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        font-family: 'Computer Modern', 'Latin Modern Math', 'STIX Two Math', serif;
    }
    .equation-box .eq-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255, 255, 255, 0.4);
        margin-bottom: 6px;
    }

    /* ── Section headers ───────────────────────────────────────────────── */
    .section-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 20px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Sidebar refinements ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.85rem;
    }

    /* ── Tooltip-style help text ───────────────────────────────────────── */
    .param-help {
        font-size: 0.72rem;
        color: rgba(255, 255, 255, 0.35);
        line-height: 1.4;
        margin-top: -4px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Page Title & Status
# =============================================================================

st.title("🌊 FluidLM Research Lab")

c = get_cfg()

# ── Training status badge ────────────────────────────────────────────────────
if c.get("request_chat"):
    st.markdown(
        '<div class="badge bg-chat">🔵 Generation in progress...</div>',
        unsafe_allow_html=True,
    )
elif c.get("pause"):
    st.markdown(
        '<div class="badge bg-pause">🟡 Training Paused</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="badge bg-run">🟢 Training Active</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar - Architecture & Hyperparameter Controls
# =============================================================================

with st.sidebar:
    # ── Architecture section ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True)

    new_batch = st.slider(
        "Batch Size",
        min_value=8, max_value=256,
        value=c.get("batch_size", 32),
        help=(
            "Number of sequences per micro-batch. "
            "**Increase** if VRAM allows - larger batches give smoother gradients. "
            "**Decrease** if you hit OOM errors. "
            "Effective batch = this × Grad Accum Steps. "
            "Changes trigger a DataLoader rebuild (brief pause)."
        ),
    )
    new_seq = st.slider(
        "Sequence Length",
        min_value=32, max_value=512,
        value=c.get("seq_len", 128),
        help=(
            "Number of tokens per training window. "
            "**Increase** to learn longer-range dependencies (at higher VRAM cost). "
            "**Decrease** if training is slow or OOM. "
            "FluidLM scales linearly with seq_len (unlike Transformers which scale quadratically). "
            "Changes trigger a DataLoader rebuild."
        ),
    )
    d_mod = st.number_input(
        "D_MODEL (restart required)",
        min_value=128, max_value=1024,
        value=c.get("d_model", 512),
        step=128,
        help=(
            "Embedding dimension - the width of the latent fluid. "
            "Determines model capacity: 512→~36M params, 768→~80M params, 1024→~140M params. "
            "Changing this requires a full restart (model architecture changes). "
            "Cannot be hot-reloaded."
        ),
    )

    # ── PDE Physics Controls ─────────────────────────────────────────────
    st.markdown('<div class="section-header">⚛️ PDE Physics</div>', unsafe_allow_html=True)

    ui_live = st.checkbox(
        "Live Refresh",
        value=c.get("ui_live", True),
        help=(
            "When enabled, the dashboard auto-refreshes every 2 seconds. "
            "Disable to freeze the display (useful for taking screenshots or "
            "analyzing a specific snapshot without it changing)."
        ),
    )
    if ui_live != c.get("ui_live"):
        c["ui_live"] = ui_live
        set_cfg(c)

    auto_pilot = st.checkbox(
        "🤖 Intelligent Auto-Pilot",
        value=c.get("auto_pilot", False),
        help=(
            "Automatic hyperparameter adjustment: "
            "• Detects output loops → increases repetition penalty & temperature. "
            "• Detects loss plateaus → reduces learning rate by 20%. "
            "• Gradually relaxes penalties when generation quality improves. "
            "Recommended for overnight runs where you can't manually tune."
        ),
    )
    if auto_pilot != c.get("auto_pilot"):
        c["auto_pilot"] = auto_pilot
        set_cfg(c)

    lr = st.slider(
        "Learning Rate",
        min_value=1e-6, max_value=1e-3,
        value=float(c.get("lr", 1e-4)),
        format="%.6f",
        help=(
            "AdamW learning rate (after warmup). "
            "**Increase** if loss is barely moving (stuck at a plateau). "
            "**Decrease** if loss is noisy/oscillating or if you see NaN. "
            "Typical range: 1e-5 to 5e-4. "
            "The auto-pilot will reduce this automatically on plateaus."
        ),
    )

    t_st = st.slider(
        "T_STEPS (Max Integration Steps)",
        min_value=4, max_value=32,
        value=c.get("t_steps", 12),
        help=(
            "Maximum number of forward Euler steps per FluidLayer. "
            "More steps = deeper effective computation = better expressiveness, "
            "but slower training (linear cost increase). "
            "The adaptive equilibrium system may halt early if the fluid stabilizes. "
            "**Increase** if 'Effort' consistently hits the max (model needs more thinking time). "
            "**Decrease** to speed up training if 'Effort' is well below the max."
        ),
    )

    dt = st.slider(
        "Δt (Euler Step Size)",
        min_value=0.01, max_value=0.5,
        value=c.get("dt", 0.1),
        help=(
            "Integration step size for the forward Euler discretization. "
            "**Increase** for faster information propagation per step (but risk instability). "
            "**Decrease** for more stable integration (but need more steps to reach equilibrium). "
            "Rule of thumb: dt × t_steps ≈ 1.0-2.0 is a good effective integration time. "
            "If you see NaN loss, reduce dt first."
        ),
    )

    epsilon_val = st.select_slider(
        "Turing Threshold (ε)",
        options=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
        value=c.get("epsilon", 0.05),
        help=(
            "Convergence threshold for Turing Equilibrium detection. "
            "Turbulence is measured as the mean per-token L2 displacement: "
            "mean(‖u_t − u_{t-3}‖₂) across all sequence positions. "
            "This metric is INDEPENDENT of sequence length. "
            "**Increase** (e.g., 0.2) for faster inference at slight quality cost. "
            "**Decrease** (e.g., 0.01) for more precise convergence at higher compute cost. "
            "During training, this only affects the reported 'Effort' metric - "
            "all steps are always computed for gradient consistency."
        ),
    )

    # ── Generation Controls ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🎲 Generation</div>', unsafe_allow_html=True)

    temp = st.slider(
        "Temperature",
        min_value=0.1, max_value=1.5,
        value=c.get("temperature", 0.8),
        help=(
            "Softmax temperature for text generation. "
            "**Lower** (0.3-0.6): more deterministic, repetitive, 'safe' outputs. "
            "**Higher** (0.9-1.2): more creative, diverse, but potentially incoherent. "
            "Only affects the Chat and sample text - does NOT affect training loss."
        ),
    )

    pen = st.slider(
        "Repetition Penalty",
        min_value=1.0, max_value=2.5,
        value=c.get("repetition_penalty", 1.5),
        help=(
            "Penalizes tokens that appeared in the recent generation window (50 tokens). "
            "1.0 = no penalty. Higher values suppress repetition more aggressively. "
            "**Increase** if the model is stuck in loops ('the the the...'). "
            "**Decrease** if outputs seem unnaturally varied or miss common words. "
            "The auto-pilot adjusts this automatically when loops are detected."
        ),
    )

    # ── Training Control ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">⏯️ Control</div>', unsafe_allow_html=True)

    pause = st.checkbox(
        "Pause Training",
        value=c.get("pause", False),
        help=(
            "Pause the training loop. The engine enters a sleep loop "
            "and stops processing batches until unpaused. "
            "VRAM remains allocated (the model stays on GPU). "
            "Use this when you need to free GPU compute for other tasks "
            "without losing the current training state."
        ),
    )
    if pause != c.get("pause"):
        c["pause"] = pause
        set_cfg(c)

    if st.button("🚀 Apply Changes", width="stretch"):
        c.update({
            "batch_size": new_batch,
            "seq_len": new_seq,
            "d_model": d_mod,
            "lr": lr,
            "t_steps": t_st,
            "dt": dt,
            "epsilon": epsilon_val,
            "temperature": temp,
            "repetition_penalty": pen,
        })
        set_cfg(c)
        st.success("✅ Config synchronized with engine.")


# =============================================================================
# Main Content - Telemetry & Visualization
# =============================================================================

if os.path.exists(LOG_F):
    try:
        with open(LOG_F, "r") as f:
            d = json.load(f)

        step = d.get("step", 0)
        cfg_updated = False

        # ── Auto-Pilot Logic ─────────────────────────────────────────────
        # This runs in the dashboard process (not the engine) because it
        # needs access to the generated text sample for loop detection.
        # It writes adjustments back to config.json, which the engine
        # picks up on its next config poll (every 10 steps).
        if auto_pilot and step != c.get("last_autopilot_step", -1):
            pen_val = c.get("repetition_penalty", 1.5)
            temp_val = c.get("temperature", 0.8)
            curr_lr = c.get("lr", 1e-4)
            c["last_autopilot_step"] = step

            if detect_loop(d.get("sample", "")):
                # Loop detected → increase penalty and temperature to break out
                new_pen = min(2.5, pen_val + 0.15)
                new_temp = min(1.2, temp_val + 0.05)
                c.update({"repetition_penalty": new_pen, "temperature": new_temp})
                st.toast(f"🚨 Loop detected! Pen→{new_pen:.2f} | Temp→{new_temp:.2f}", icon="🤖")
                cfg_updated = True
            else:
                # No loop → gently relax penalty and temperature toward defaults
                new_pen = max(1.2, pen_val - 0.02)
                new_temp = max(0.8, temp_val - 0.01)
                if new_pen != pen_val or new_temp != temp_val:
                    c.update({"repetition_penalty": new_pen, "temperature": new_temp})
                    cfg_updated = True

            # ── Plateau detection for LR decay ───────────────────────────
            if os.path.exists(STAT_F):
                try:
                    with open(STAT_F, "r") as sf:
                        s_data = json.load(sf)
                    loss_h = s_data.get("loss", [])
                    last_decay = c.get("last_decay_step", 0)
                    # Only trigger if we have enough history and haven't
                    # decayed recently (cooldown of 500 steps).
                    if len(loss_h) > 50 and (step - last_decay) > 500:
                        recent_loss = np.mean(loss_h[-20:])
                        old_loss = np.mean(loss_h[-50:-30])
                        # If recent loss hasn't improved by at least 1%
                        if recent_loss > (old_loss * 0.99) and curr_lr > 1e-6:
                            new_lr = max(1e-6, curr_lr * 0.8)
                            c.update({"lr": new_lr, "last_decay_step": step})
                            st.toast(
                                f"📉 Plateau detected! LR reduced to {new_lr:.2e}",
                                icon="🧠",
                            )
                            cfg_updated = True
                except Exception:
                    pass

            if cfg_updated:
                set_cfg(c)

        # ── Extract current metrics ──────────────────────────────────────
        loss_val = d.get("loss", 0)
        it_val = d.get("it_s", 0)
        avg_steps = d.get("avg_steps", c.get("t_steps", 12))

        # ── Compute deltas from previous logged step ─────────────────────
        delta_loss, delta_it, delta_steps = None, None, None
        if os.path.exists(STAT_F):
            try:
                with open(STAT_F, "r") as sf:
                    s_data = json.load(sf)
                loss_h = s_data.get("loss", [])
                it_h = s_data.get("it_s", [])
                steps_h = s_data.get("avg_steps", [])
                if len(loss_h) > 1:
                    delta_loss = loss_val - loss_h[-2]
                if len(it_h) > 1:
                    delta_it = it_val - it_h[-2]
                if len(steps_h) > 1:
                    delta_steps = avg_steps - steps_h[-2]
            except Exception:
                pass

        # ── Top-level metrics row ────────────────────────────────────────
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Step", f"{step:,}")
        c2.metric(
            "Loss",
            f"{loss_val:.4f}",
            delta=f"{delta_loss:.4f}" if delta_loss is not None else None,
            delta_color="inverse",
            help="Cross-entropy loss (lower is better). Delta shows change from previous logged step.",
        )
        c3.metric(
            "🌊 Effort (Steps)",
            f"{avg_steps:.1f} / {c.get('t_steps', 12)}",
            delta=f"{delta_steps:.1f}" if delta_steps is not None else None,
            delta_color="inverse",
            help=(
                "Average PDE integration steps across layers before Turing equilibrium. "
                "If consistently at max → the model needs more thinking time (increase T_STEPS). "
                "If well below max → the model converges fast (could decrease T_STEPS to save compute)."
            ),
        )
        c4.metric("VRAM", f"{d.get('vram', 0):.0f} MB", help="GPU memory reserved by PyTorch (includes fragmentation overhead).")
        c5.metric("Epoch", d.get("epoch", 0))
        c6.metric(
            "Speed",
            f"{it_val:.2f} it/s",
            delta=f"{delta_it:.2f}" if delta_it is not None else None,
            help="Training iterations per second. Drops during logging steps (sample generation overhead).",
        )
        c7.metric("ETA", format_eta(d.get("eta", 0)), help="Estimated time remaining for current epoch.")

        # ── Timestamp ────────────────────────────────────────────────────
        ts = d.get("timestamp", 0)
        if ts > 0:
            st.caption(f"⏱️ Last telemetry: {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")

        # ── Live Equations Panel ─────────────────────────────────────────
        # Display the governing equations with current parameter values
        # injected in real-time. This serves as both documentation and
        # a quick reference for what the current configuration means
        # mathematically.
        with st.expander("📐 Live Equations - Current Configuration", expanded=False):
            eq_col1, eq_col2 = st.columns(2)

            with eq_col1:
                st.markdown("""<div class="equation-box"><div class="eq-label">Governing PDE (FluidLM Update Rule)</div></div>""", unsafe_allow_html=True)
                st.latex(
                    r"\frac{\partial u}{\partial t} = \sum_{k} D_k \cdot \nabla^2_{d_k}(u) "
                    r"\;+\; R(u, \theta) \;+\; \alpha \cdot h_t"
                )
                st.markdown("**Current discretization (Forward Euler):**")
                _dt = c.get('dt', 0.1)
                _ts = c.get('t_steps', 12)
                st.latex(
                    rf"u_{{t+1}} = \text{{LayerNorm}}\!\left( u_t + "
                    rf"\underbrace{{{_dt}}}_{{\\Delta t}}"
                    rf"\cdot \left[ \text{{Diffusion}} + \text{{Reaction}} + 0.05 \cdot h_t \right] \right)"
                )
                st.markdown(f"""
- **Δt** = `{_dt}` · **T_max** = `{_ts}` → Effective integration time: **{_dt * _ts:.2f}**
- **Dilations**: [1, 4, 16] → Max single-step reach: 16 tokens
""")

            with eq_col2:
                _eps = c.get('epsilon', 0.05)
                _ts2 = c.get('t_steps', 12)
                st.markdown("""<div class="equation-box"><div class="eq-label">Turing Equilibrium (Adaptive Compute)</div></div>""", unsafe_allow_html=True)
                st.latex(
                    r"\text{turbulence} = \text{mean}_{i}\!\left(\| u_t^{(i)} - u_{t-3}^{(i)} \|_2\right)"
                )
                st.latex(
                    rf"\text{{if }} \text{{turbulence}} < "
                    rf"\underbrace{{{_eps}}}_{{\\varepsilon}} "
                    rf"\implies \text{{HALT}}"
                )
                st.markdown(f"""
- **ε** = `{_eps}` - Per-token L2 threshold (sequence-length independent)
- **Avg effort**: `{avg_steps:.1f}` / `{_ts2}` steps
- **Theoretical max receptive field**: ~{int(_ts2 * 16 * 4)} tokens (T × d_max × L)
""")

            _t = c.get('temperature', 0.8)
            _p = c.get('repetition_penalty', 1.5)
            _lr = c.get('lr', 1e-4)
            st.markdown(f"""
<div class="equation-box">
<div class="eq-label">Generation Parameters</div>
</div>
""", unsafe_allow_html=True)
            st.latex(
                rf"\text{{logits}}_{{\text{{scaled}}}} = "
                rf"\frac{{\text{{logits}}}}{{\underbrace{{{_t}}}_{{T}}}}"
            )
            st.latex(
                rf"\text{{penalty}}:\;"
                rf"\ell_i \times {_p:.2f} \;\text{{ if }} \ell_i < 0"
                rf"\quad|\quad"
                rf"\ell_i \;/\; {_p:.2f} \;\text{{ if }} \ell_i \geq 0"
            )
            st.markdown(f"""
- **Temperature** = `{_t}` - Lower → deterministic, Higher → creative
- **Repetition penalty** = `{_p:.2f}` - Applied to tokens seen in last 50 tokens
- **Learning rate** = `{_lr:.2e}` (after warmup)
""")


        # ── Two-column layout: Telemetry (left) | Waves + Sample (right) ─
        col_l, col_r = st.columns(2)

        # ────────────────────────────────────────────────────────────────
        # LEFT COLUMN - Charts & Controls
        # ────────────────────────────────────────────────────────────────
        with col_l:
            st.subheader("📉 Telemetry")
            god_mode = st.toggle(
                "👁️ Superposition (God Mode)",
                help=(
                    "Overlay all metrics (loss, effort, VRAM, LR, temp, penalty, speed) "
                    "on a single normalized plot. Useful for spotting correlations: "
                    "e.g., does LR decay coincide with loss drops? "
                    "Does effort increase when loss is high?"
                ),
            )

            if os.path.exists(STAT_F):
                try:
                    with open(STAT_F, "r") as sf:
                        stats = json.load(sf)

                    if god_mode and len(stats.get("loss", [])) > 0:
                        # ── God Mode: normalized overlay of all metrics ──
                        fig_god = go.Figure()
                        metrics_to_plot = {
                            "Loss": ("loss", "#ff4b4b"),
                            "Effort (Steps)": ("avg_steps", "#ff69b4"),
                            "VRAM": ("vram", "#9b59b6"),
                            "LR": ("lr", "#f1c40f"),
                            "Temp": ("temp", "#e67e22"),
                            "Penalty": ("penalty", "#2ecc71"),
                            "Speed (it/s)": ("it_s", "#00d4ff"),
                        }
                        for name, (key, color) in metrics_to_plot.items():
                            if key in stats and len(stats[key]) > 0:
                                raw_vals = stats[key]
                                norm_vals = norm_data(stats[key])
                                fig_god.add_trace(go.Scatter(
                                    x=stats["step"],
                                    y=norm_vals,
                                    mode="lines",
                                    name=name,
                                    line=dict(color=color, width=1.5),
                                    hovertemplate=f"{name}: %{{customdata:.4g}}<extra></extra>",
                                    customdata=raw_vals,
                                ))
                        fig_god.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=30, b=0),
                            template="plotly_dark",
                            yaxis=dict(showticklabels=False, title="Normalized [0-100]"),
                            xaxis=dict(title="Optimizer Step"),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=10),
                            ),
                        )
                        st.plotly_chart(fig_god, width="stretch")
                    else:
                        # ── Standard loss curve ──────────────────────────
                        if len(stats.get("loss", [])) > 0:
                            fig_loss = go.Figure()
                            loss_data = stats["loss"][-500:]
                            step_data = stats["step"][-500:]
                            fig_loss.add_trace(go.Scatter(
                                x=step_data,
                                y=loss_data,
                                mode="lines",
                                line=dict(color="#00d4ff", width=1.5),
                                name="Loss",
                                hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
                            ))
                            fig_loss.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=10, b=0),
                                template="plotly_dark",
                                xaxis=dict(title="Optimizer Step"),
                                yaxis=dict(title="Cross-Entropy Loss"),
                            )
                            st.plotly_chart(fig_loss, width="stretch")
                        else:
                            st.info("⏳ Waiting for training data...")
                except Exception:
                    pass

            # ── Embedding weight distribution ────────────────────────────
            st.subheader("📊 Embedding Weight Distribution")
            st.caption(
                "Tracks the health of the embedding matrix. A healthy distribution is "
                "roughly Gaussian and stable. Watch for: collapse (all weights near 0), "
                "explosion (weights spreading to extremes), or bimodal splits."
            )
            if "w_hist" in d:
                fig_w = go.Figure(
                    data=[go.Bar(
                        x=d["w_bins"][:-1],
                        y=d["w_hist"],
                        marker_color="#00ffcc",
                        hovertemplate="Weight: %{x:.3f}<br>Count: %{y:,}<extra></extra>",
                    )]
                )
                fig_w.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=0, b=0),
                    template="plotly_dark",
                    xaxis=dict(title="Weight Value"),
                    yaxis=dict(title="Count"),
                )
                st.plotly_chart(fig_w, width="stretch")

            # ── Chat interface ───────────────────────────────────────────
            st.subheader("💬 Interactive Chat")
            st.caption(
                "Send a prompt to the model during training. The engine will pause, "
                "generate a response (100 tokens max), and resume automatically."
            )
            u_in = st.text_input("Prompt:", placeholder="Type a prompt to test the model...")
            if st.button("⚡ Generate", width="stretch"):
                c = get_cfg()
                c.update({
                    "chat_prompt": u_in,
                    "request_chat": True,
                    "pause": True,
                })
                set_cfg(c)
                st.info("📡 Request sent to engine. Waiting for response...")

        # ────────────────────────────────────────────────────────────────
        # RIGHT COLUMN - Turing Waves & Sample Output
        # ────────────────────────────────────────────────────────────────
        with col_r:
            st.subheader("🌊 Turing Waves")
            st.caption(
                "Heatmap of activation magnitudes across sequence positions (x-axis) "
                "and integration steps (y-axis) in the last FluidLayer. "
                "Structured patterns = the model is forming meaningful representations. "
                "Uniform color = the model hasn't differentiated the input yet."
            )
            if d.get("waves"):
                fig = px.imshow(
                    np.array(d["waves"]),
                    color_continuous_scale="Magma",
                    aspect="auto",
                    labels=dict(
                        x="Sequence Position",
                        y="Integration Step",
                        color="Activation |u|",
                    ),
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("⏳ Wave data available at next log step (every 50 steps).")

            # ── Sample output ────────────────────────────────────────────
            st.subheader("📝 Latest Sample Output")
            st.caption(
                "Auto-generated text sample (30 tokens) from the current model state. "
                "Quality improves as training progresses. Early outputs will be gibberish - "
                "that's normal. Watch for: recognizable words → phrases → coherent sentences."
            )
            sample_text = d.get("sample", "")
            if sample_text:
                st.code(sample_text, language="text")
            else:
                st.info("No sample generated yet.")

    except json.JSONDecodeError:
        st.warning("⚠️ Telemetry file is being written - will refresh automatically.")
    except Exception as e:
        st.sidebar.error(f"Dashboard error: {e}")

else:
    # ── No telemetry file yet - training hasn't started ──────────────────
    st.info(
        "⏳ Waiting for training engine to start writing telemetry... "
        "Make sure `train_engine.py` is running."
    )

# =============================================================================
# Auto-refresh Loop
# =============================================================================
if ui_live:
    time.sleep(2)
    st.rerun()
else:
    st.info("⏸️ Live refresh is paused. Enable 'Live Refresh' in the sidebar to resume.")