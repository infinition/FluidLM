#!/usr/bin/env python3
# =============================================================================
# FluidLM Research Dashboard — V4.4.5
# =============================================================================
#
# V4.4.5 Changes -- Intelligent Monitoring:
#
#   [DASH-1] Alert system with semantic thresholds
#     - "ACTION REQUIRED" banner when a signal exceeds a critical threshold
#     - Each metric is color-coded: green (OK) -> orange (watch) -> red (act)
#     - Thresholds: loss plateau (>200 steps without descent), turb > 0.15,
#       gate_sat > 0.25, effort stuck at max, LR too high/low
#
#   [DASH-2] Trends on all key metrics
#     - Arrows computed over the last 20 data points
#     - Semantic coloring: green if trending well, red if trending poorly
#
#   [DASH-3] Enhanced main chart
#     - Turbulence overlay (secondary Y axis, orange line)
#     - gate_sat + long_gate_sat overlay (red dashed line)
#     - decay_mean overlay (purple line)
#     - God Mode: all metrics normalized with interactive sliders
#
#   [DASH-4] "Health Monitor" panel replaces the Embeddings panel
#     - Gate sat, long gate sat, decay, dt_learned per layer
#     - Color-coded visual gauges
#     - Embeddings histogram moved to secondary tab
#
# Author: Fabien POLLY (Infinition)
# =============================================================================

import streamlit as st
import json
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter, deque
from datetime import datetime

st.set_page_config(
    page_title="FluidLM Lab",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_chat" not in st.session_state:
    st.session_state.pending_chat = False
if "last_chat_response_key" not in st.session_state:
    st.session_state.last_chat_response_key = None

CFG_F  = "config.json"
LOG_F  = "live_logs.json"
STAT_F = "training_stats.json"
SAMPLE_HISTORY_F = "sample_history.jsonl"
LIVE_REFRESH_SECONDS = 3.0

# =============================================================================
# Thresholds — the intelligence layer
# All decisions about "is this OK?" live here, not scattered in the UI
# =============================================================================

THRESHOLDS = {
    "loss_plateau_steps": 200,   # steps without >0.5% improvement = plateau
    "turb_warn":   0.12,         # turbulence: watch
    "turb_crit":   0.18,         # turbulence: act (raise eq_weight or lower LR)
    "gate_sat_warn": 0.20,       # gate saturation: watch
    "gate_sat_crit": 0.35,       # gate saturation: act (gates dying)
    "effort_stuck": 0.95,        # avg_steps / max_steps ratio: model never converges
    "decay_low":  0.92,          # decay_mean: forgetting too fast
    "decay_high": 0.999,         # decay_mean: not forgetting at all (reservoir saturating)
    "lr_low":     5e-6,          # LR dangerously low (cosine floor hit)
}

AUTOPILOT = {
    "start_step": 600,
    "cadence_steps": 150,
    "lr_cooldown": 600,
    "physics_cooldown": 300,
    "sample_cooldown": 150,
    "compute_cooldown": 450,
    "relax_cooldown": 450,
    "min_lr": 3e-5,
    "eq_weight_max": 0.08,
    "gate_reg_max": 0.18,
    "dt_min": 0.06,
    "dt_max": 0.14,
    "t_steps_min": 8,
    "t_steps_max": 20,
    "temperature_min": 0.45,
    "temperature_max": 1.05,
    "penalty_min": 1.05,
    "penalty_max": 2.30,
}


# =============================================================================
# Utilities
# =============================================================================

def atomic_write(data, path, retries=3):
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
    if not os.path.exists(CFG_F):
        return {}
    try:
        with open(CFG_F, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def set_cfg(c):
    atomic_write(c, CFG_F)

def load_jsonl_tail(path, limit=24):
    if not os.path.exists(path):
        return []
    items = deque(maxlen=limit)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return list(items)

def detect_loop(text, min_len=10):
    if not text or len(text) < min_len:
        return False
    if any(len(set(text[i:i + 6])) == 1 for i in range(len(text) - 5)):
        return True
    for n in range(2, 5):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        if ngrams:
            mc = Counter(ngrams).most_common(1)[0]
            if mc[1] > len(text) / n / 2:
                return True
    return False

def fmt_eta(s):
    if s < 60: return f"{int(s)}s"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m" if h > 0 else f"{int(m)}m{int(s)}s"

def norm_data(arr):
    if not arr: return []
    mi, ma = min(arr), max(arr)
    if ma == mi: return [50.0] * len(arr)
    return [((x - mi) / (ma - mi)) * 100 for x in arr]

def fmt_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return str(int(n))

def epsilon_options(current):
    base = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    try:
        current_val = float(current)
    except Exception:
        current_val = 0.05
    if current_val not in base:
        base.append(current_val)
    return sorted(set(base), reverse=True)

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def next_epsilon(current, direction="up"):
    options = sorted(epsilon_options(current))
    try:
        current_val = float(current)
    except Exception:
        current_val = 0.05
    if current_val not in options:
        options.append(current_val)
        options = sorted(set(options))
    index = options.index(current_val)
    if direction == "up":
        return options[min(len(options) - 1, index + 1)]
    return options[max(0, index - 1)]

def recent_mean(arr, n):
    if not arr:
        return None
    window = arr[-min(len(arr), n):]
    return float(np.mean(window)) if window else None

def recent_delta(arr, recent_n=6, previous_n=6):
    if len(arr) < (recent_n + previous_n):
        return 0.0
    recent = arr[-recent_n:]
    previous = arr[-(recent_n + previous_n):-recent_n]
    prev_mean = float(np.mean(previous)) if previous else 0.0
    if abs(prev_mean) < 1e-9:
        return 0.0
    return (float(np.mean(recent)) - prev_mean) / abs(prev_mean)

def update_counter(state, key, condition):
    state[key] = int(state.get(key, 0)) + 1 if condition else 0
    return state[key]

def compute_sample_quality(text):
    if not text:
        return {
            "score": 0.0,
            "loop": False,
            "weird_markers": 0,
            "token_repeat": 1.0,
            "unique_ratio": 0.0,
            "alpha_ratio": 0.0,
        }

    cleaned = text.replace("\n", " ").strip()
    tokens = [tok for tok in cleaned.split(" ") if tok]
    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    weird_markers = cleaned.lower().count("<|endoftext|>") + cleaned.count("�") + cleaned.count("ï¿½")
    loop = detect_loop(cleaned)

    if tokens:
        counts = Counter(tokens)
        token_repeat = counts.most_common(1)[0][1] / max(1, len(tokens))
        unique_ratio = len(counts) / max(1, len(tokens))
    else:
        token_repeat = 1.0
        unique_ratio = 0.0

    alpha_ratio = alpha_chars / max(1, len(cleaned))
    punctuation_hits = sum(ch in ".,!?:;\"'" for ch in cleaned)
    punctuation_ratio = punctuation_hits / max(1, len(cleaned))

    score = 100.0
    score -= 42.0 if loop else 0.0
    score -= min(45.0, weird_markers * 22.0)
    score -= max(0.0, (0.62 - unique_ratio) * 70.0)
    score -= max(0.0, (token_repeat - 0.11) * 180.0)
    score -= max(0.0, (0.70 - alpha_ratio) * 80.0)
    score -= max(0.0, (0.015 - punctuation_ratio) * 400.0)
    score = clamp(score, 0.0, 100.0)

    return {
        "score": score,
        "loop": loop,
        "weird_markers": weird_markers,
        "token_repeat": token_repeat,
        "unique_ratio": unique_ratio,
        "alpha_ratio": alpha_ratio,
    }

def analyze_autopilot_state(d, stats, c, sample_history):
    loss_h = stats.get("loss", [])
    turb_h = stats.get("diff_turb", [])
    steps_h = stats.get("avg_steps", [])

    sample_text = d.get("sample", "")
    quality = compute_sample_quality(sample_text)
    history_samples = [entry.get("sample", "") for entry in sample_history[-8:]]
    history_quality_scores = [compute_sample_quality(text)["score"] for text in history_samples if text]
    quality_series = history_quality_scores + [quality["score"]]

    recent_loss = recent_mean(loss_h, 6)
    previous_loss = recent_mean(loss_h[:-6], 6) if len(loss_h) > 12 else None
    recent_turb = recent_mean(turb_h, 6)
    previous_turb = recent_mean(turb_h[:-6], 6) if len(turb_h) > 12 else None
    recent_effort = recent_mean(steps_h, 6)
    recent_quality = recent_mean(quality_series, 4)
    previous_quality = recent_mean(quality_series[:-4], 4) if len(quality_series) > 8 else None
    long_gate_series = [float(entry.get("long_gate_sat", 0.0) or 0.0) for entry in sample_history[-8:]]
    long_gate_series.append(float(d.get("long_gate_sat", 0.0) or 0.0))
    recent_long_gate = recent_mean(long_gate_series, 4)
    long_gate_trend = recent_delta(long_gate_series, 4, 4)

    plateau = False
    if len(loss_h) >= 10:
        window = loss_h[-10:]
        best = min(window)
        worst = max(window)
        plateau = worst > 0 and ((worst - best) / worst) < 0.018

    loss_trend = 0.0
    if recent_loss is not None and previous_loss is not None and previous_loss > 0:
        loss_trend = (recent_loss - previous_loss) / previous_loss

    turb_trend = 0.0
    if recent_turb is not None and previous_turb is not None and previous_turb > 0:
        turb_trend = (recent_turb - previous_turb) / previous_turb

    max_steps = max(1, c.get("t_steps", 12))
    effort_ratio = (recent_effort or d.get("avg_steps", max_steps)) / max_steps
    long_gate_sat = float(d.get("long_gate_sat", 0.0) or 0.0)
    gate_sat = float(d.get("gate_sat", 0.0) or 0.0)
    decay_mean = float(d.get("decay_mean", 0.0) or 0.0)
    turb_now = float(d.get("diff_turb", 0.0) or 0.0)
    gate_mean = float(d.get("gate_mean", 0.0) or 0.0)
    gate_reg = float(d.get("gate_reg", 0.0) or 0.0)

    health = 100.0
    health -= min(40.0, max(0.0, turb_now - 0.18) * 78.0)
    health -= 15.0 if plateau else 0.0
    health -= max(0.0, effort_ratio - 0.90) * 28.0
    health -= max(0.0, long_gate_sat - 0.16) * 65.0
    health -= max(0.0, gate_sat - 0.08) * 120.0
    health -= max(0.0, decay_mean - 0.97) * 120.0
    health -= max(0.0, gate_mean - 0.70) * 30.0
    health -= max(0.0, 70.0 - quality["score"]) * 0.55
    health = clamp(health, 0.0, 100.0)

    regime = []
    if quality["loop"] or quality["weird_markers"] > 0 or quality["score"] < 65.0:
        regime.append("sample_degenerate")
    if plateau:
        regime.append("plateau")
    if turb_now > 0.40:
        regime.append("turbulence_high")
    if turb_now > 0.55:
        regime.append("turbulence_extreme")
    if turb_trend < -0.03:
        regime.append("turbulence_improving")
    elif turb_now > 0.40 and turb_trend > -0.01:
        regime.append("turbulence_stalled")
    if effort_ratio >= 0.99:
        regime.append("effort_stuck")
    if long_gate_sat > 0.24 or (recent_long_gate and recent_long_gate > 0.18 and long_gate_trend > 0.08):
        regime.append("long_gate_rising")
    if decay_mean > 0.956:
        regime.append("memory_sticky")
    if quality["score"] > 72.0 and loss_trend < -0.01:
        regime.append("quality_improving")
    if turb_now < 0.32 and effort_ratio < 0.90 and loss_trend < -0.01:
        regime.append("stable_learning")

    return {
        "health": round(health, 1),
        "plateau": plateau,
        "loss_trend": loss_trend,
        "turb_trend": turb_trend,
        "turb_now": turb_now,
        "effort_ratio": effort_ratio,
        "long_gate_sat": long_gate_sat,
        "gate_sat": gate_sat,
        "gate_mean": gate_mean,
        "gate_reg": gate_reg,
        "decay_mean": decay_mean,
        "quality": quality,
        "recent_quality": recent_quality,
        "previous_quality": previous_quality,
        "long_gate_trend": long_gate_trend,
        "regime": regime,
        "recent_loss": recent_loss,
        "recent_turb": recent_turb,
        "history_depth": len(sample_history),
    }

def run_autopilot(c, d, stats, sample_history):
    step = int(d.get("step", 0) or 0)
    state = dict(c.get("autopilot_state", {}))
    cadence = AUTOPILOT["cadence_steps"]
    last_step = int(c.get("last_autopilot_step", -cadence))

    analysis = analyze_autopilot_state(d, stats, c, sample_history)
    update_counter(state, "sample_bad_count", "sample_degenerate" in analysis["regime"])
    update_counter(state, "turb_high_count", "turbulence_high" in analysis["regime"])
    update_counter(state, "turb_extreme_count", "turbulence_extreme" in analysis["regime"])
    update_counter(state, "turb_stalled_count", "turbulence_stalled" in analysis["regime"])
    update_counter(state, "effort_stuck_count", "effort_stuck" in analysis["regime"])
    update_counter(state, "long_gate_rising_count", "long_gate_rising" in analysis["regime"])
    update_counter(state, "stable_learning_count", "stable_learning" in analysis["regime"])
    update_counter(state, "memory_sticky_count", "memory_sticky" in analysis["regime"])
    state.update({
        "health": analysis["health"],
        "regime": analysis["regime"],
        "quality": round(analysis["quality"]["score"], 1),
        "updated_step": step,
    })

    if step <= 0 or step < AUTOPILOT["start_step"] or (step - last_step) < cadence:
        return False, state, None

    updates = {}
    reasons = []
    toasts = []
    curr_lr = float(c.get("lr", 3e-4))
    eq_weight = float(c.get("eq_weight", 0.01))
    epsilon = float(c.get("epsilon", 0.05))
    gate_reg_weight = float(c.get("gate_reg_weight", 0.08))
    temp_val = float(c.get("temperature", 0.8))
    pen_val = float(c.get("repetition_penalty", 1.5))
    dt_val = float(c.get("dt", 0.1))
    t_steps = int(c.get("t_steps", 12))

    last_lr_step = int(state.get("last_lr_step", -AUTOPILOT["lr_cooldown"]))
    last_physics_step = int(state.get("last_physics_step", -AUTOPILOT["physics_cooldown"]))
    last_sample_step = int(state.get("last_sample_step", -AUTOPILOT["sample_cooldown"]))
    last_compute_step = int(state.get("last_compute_step", -AUTOPILOT["compute_cooldown"]))
    last_relax_step = int(state.get("last_relax_step", -AUTOPILOT["relax_cooldown"]))

    quality = analysis["quality"]
    turb_high_count = int(state.get("turb_high_count", 0))
    turb_stalled_count = int(state.get("turb_stalled_count", 0))
    turb_extreme_count = int(state.get("turb_extreme_count", 0))
    sample_bad_count = int(state.get("sample_bad_count", 0))
    effort_stuck_count = int(state.get("effort_stuck_count", 0))
    long_gate_rising_count = int(state.get("long_gate_rising_count", 0))
    stable_learning_count = int(state.get("stable_learning_count", 0))
    memory_sticky_count = int(state.get("memory_sticky_count", 0))

    if sample_bad_count >= 1 and (step - last_sample_step) >= AUTOPILOT["sample_cooldown"]:
        sample_penalty_bump = 0.08
        sample_temp_shift = -0.04
        if quality["loop"] and quality["token_repeat"] > 0.16:
            sample_penalty_bump = 0.14
            sample_temp_shift = -0.02
            reasons.append("sample loops")
        elif quality["weird_markers"] > 0:
            sample_penalty_bump = 0.12
            sample_temp_shift = -0.05
            reasons.append("control tokens leak")
        else:
            reasons.append("sample degenerates")

        updates["repetition_penalty"] = round(clamp(pen_val + sample_penalty_bump, AUTOPILOT["penalty_min"], AUTOPILOT["penalty_max"]), 3)
        updates["temperature"] = round(clamp(temp_val + sample_temp_shift, AUTOPILOT["temperature_min"], AUTOPILOT["temperature_max"]), 3)
        state["last_sample_step"] = step
        toasts.append("Auto-Pilot: sample hygiene adjustment")

    if (turb_stalled_count >= 1 or turb_extreme_count >= 1 or (analysis["plateau"] and turb_high_count >= 1)) and (step - last_physics_step) >= AUTOPILOT["physics_cooldown"]:
        eq_bump = 0.005
        if turb_extreme_count >= 1:
            eq_bump = 0.01
        updates["eq_weight"] = round(clamp(eq_weight + eq_bump, 0.0, AUTOPILOT["eq_weight_max"]), 3)
        updates["dt"] = round(clamp(dt_val - 0.01, AUTOPILOT["dt_min"], AUTOPILOT["dt_max"]), 3)
        if long_gate_rising_count >= 1 or analysis["gate_mean"] > 0.68 or memory_sticky_count >= 1:
            updates["gate_reg_weight"] = round(clamp(gate_reg_weight + 0.01, 0.0, AUTOPILOT["gate_reg_max"]), 3)
        reasons.append("physics stabilization")
        state["last_physics_step"] = step
        toasts.append("Auto-Pilot: stronger physics stabilization")

    if (analysis["plateau"] or turb_stalled_count >= 2 or long_gate_rising_count >= 1 or memory_sticky_count >= 2) and curr_lr > AUTOPILOT["min_lr"] and (step - last_lr_step) >= AUTOPILOT["lr_cooldown"]:
        lr_decay = 0.90 if turb_extreme_count >= 1 else (0.93 if turb_stalled_count >= 2 else 0.95)
        updates["lr"] = float(max(1e-6, curr_lr * lr_decay))
        reasons.append("lr decay")
        state["last_lr_step"] = step
        c["last_decay_step"] = step
        toasts.append(f"Auto-Pilot: LR -> {updates['lr']:.2e}")

    if (
        effort_stuck_count >= 2
        and analysis["turb_now"] > 0.38
        and (step - last_compute_step) >= AUTOPILOT["compute_cooldown"]
    ):
        updates["t_steps"] = int(clamp(t_steps + 2, AUTOPILOT["t_steps_min"], AUTOPILOT["t_steps_max"]))
        updates["epsilon"] = next_epsilon(epsilon, direction="down")
        reasons.append("more compute budget")
        state["last_compute_step"] = step
        toasts.append(f"Auto-Pilot: t_steps -> {updates['t_steps']}")

    if (
        effort_stuck_count >= 2
        and analysis["turb_now"] < 0.28
        and stable_learning_count >= 1
        and (step - last_compute_step) >= AUTOPILOT["compute_cooldown"]
    ):
        updates["epsilon"] = next_epsilon(epsilon, direction="up")
        updates["t_steps"] = int(clamp(t_steps - 1, AUTOPILOT["t_steps_min"], AUTOPILOT["t_steps_max"]))
        reasons.append("reduce wasted compute")
        state["last_compute_step"] = step
        toasts.append("Auto-Pilot: compute budget trimmed")

    if not updates and stable_learning_count >= 2 and (step - last_relax_step) >= AUTOPILOT["relax_cooldown"]:
        relax_pen = round(clamp(pen_val - 0.03, 1.10, AUTOPILOT["penalty_max"]), 3)
        relax_temp = round(clamp(temp_val + 0.02, 0.55, AUTOPILOT["temperature_max"]), 3)
        relax_eq = round(clamp(eq_weight - 0.002, 0.01, AUTOPILOT["eq_weight_max"]), 3)
        relax_dt = round(clamp(dt_val + 0.005, AUTOPILOT["dt_min"], 0.10), 3)
        if relax_pen != pen_val or relax_temp != temp_val or relax_eq != eq_weight or relax_dt != dt_val:
            updates["repetition_penalty"] = relax_pen
            updates["temperature"] = relax_temp
            updates["eq_weight"] = relax_eq
            updates["dt"] = relax_dt
            reasons.append("relax sample constraints")
            state["last_sample_step"] = step
            state["last_relax_step"] = step

    if not updates and analysis["quality"]["score"] > 78.0 and analysis["turb_now"] < 0.45 and temp_val < 0.62:
        updates["temperature"] = round(clamp(temp_val + 0.01, AUTOPILOT["temperature_min"], 0.68), 3)
        reasons.append("gentle generation relax")

    c["last_autopilot_step"] = step
    changed = False
    if updates:
        c.update(updates)
        state["last_action"] = ", ".join(reasons)
        state["last_action_step"] = step
        state["last_updates"] = updates
        changed = True
    else:
        state["last_action"] = "observe"
        state["last_action_step"] = step
        state["last_updates"] = {}

    c["autopilot_state"] = state
    return changed, state, toasts

def trend(arr, n=20):
    """Returns (arrow, css_class) based on last n points slope."""
    if len(arr) < n:
        return "→", "delta-neutral"
    slope = arr[-1] - arr[-n]
    rel = slope / (abs(arr[-n]) + 1e-9)
    if abs(rel) < 0.005:
        return "→", "delta-neutral"
    return ("↑", "delta-up") if slope > 0 else ("↓", "delta-down")

def trend_for(arr, n=20, good_direction="down"):
    """Returns (arrow, css_class) with semantic coloring."""
    arrow, _ = trend(arr, n)
    if arrow == "→":
        return "→", "delta-neutral"
    if good_direction == "down":
        cls = "delta-good" if arrow == "↓" else "delta-bad"
    else:
        cls = "delta-good" if arrow == "↑" else "delta-bad"
    return arrow, cls

def compute_alerts(d, stats, c):
    """
    Returns list of (level, message, suggestion) tuples.
    level: 'warn' | 'crit'
    """
    alerts = []
    loss_h = stats.get("loss", [])
    T = THRESHOLDS

    # 1. Loss plateau detection
    if len(loss_h) > T["loss_plateau_steps"] // 50:
        window = loss_h[-(T["loss_plateau_steps"] // 50):]
        best = min(window)
        worst = max(window)
        if worst > 0 and (worst - best) / worst < 0.005:
            alerts.append(("crit", "Loss plateau detected",
                           f"Lower LR x 0.3 or increase eq_weight"))

    # 2. Turbulence
    turb = d.get("diff_turb", 0)
    if turb > T["turb_crit"]:
        alerts.append(("crit", f"Critical turbulence ({turb:.4f})",
                       "Increase eq_weight or lower LR"))
    elif turb > T["turb_warn"]:
        alerts.append(("warn", f"High turbulence ({turb:.4f})",
                       "Monitor -- may stabilize naturally"))

    # 3. Gate saturation
    gate_sat = d.get("gate_sat", None)
    if gate_sat is not None:
        if gate_sat > T["gate_sat_crit"]:
            alerts.append(("crit", f"Critical gate saturation ({gate_sat:.3f})",
                           "Gates stuck -- dead gradient. Lower LR."))
        elif gate_sat > T["gate_sat_warn"]:
            alerts.append(("warn", f"High gate saturation ({gate_sat:.3f})",
                           "Monitor -- may indicate instability"))

    # 4. Effort stuck at max
    max_steps = c.get("t_steps", 12)
    avg_steps_val = d.get("avg_steps", max_steps)
    if avg_steps_val / max_steps > T["effort_stuck"]:
        if len(loss_h) > 100:  # only after warmup
            alerts.append(("warn", "Effort stuck at maximum (no early stop)",
                           "Turing Equilibrium never reached -- lower epsilon or increase T steps"))

    # 5. Decay
    decay = d.get("decay_mean", None)
    if decay is not None:
        if decay < T["decay_low"]:
            alerts.append(("warn", f"Decay too low ({decay:.4f}) -- fast forgetting",
                           "Reservoir h unstable -- monitor loss"))
        elif decay > T["decay_high"]:
            alerts.append(("warn", f"Decay near 1 ({decay:.4f}) -- reservoir saturating",
                           "Forget gate is not learning to forget"))

    # 6. LR floor
    current_lr = d.get("current_lr", c.get("lr", 3e-4))
    if current_lr < T["lr_low"]:
        alerts.append(("warn", f"LR very low ({current_lr:.2e})",
                       "Cosine schedule nearly exhausted -- consider restart"))

    return alerts


# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    * { font-family: 'IBM Plex Sans', sans-serif; }
    code, .mono { font-family: 'IBM Plex Mono', monospace; }

    .block-container {
        padding: 0.4rem 0.8rem 0rem 0.8rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] { height: 0px !important; min-height: 0 !important; }
    #MainMenu, footer, .stDeployButton { display: none !important; }
    .element-container { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 6px !important;
        background: rgba(12, 16, 22, 0.92) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        box-shadow: 0 1px 8px rgba(0,0,0,0.5);
        padding: 6px 10px !important;
        margin-bottom: 3px !important;
    }

    /* -- Header -- */
    .hdr-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 5px 0 3px 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 5px;
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .hdr-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e8f4f8;
        letter-spacing: -0.02em;
        white-space: nowrap;
        font-family: 'IBM Plex Mono', monospace;
    }
    .hdr-badge {
        padding: 2px 9px;
        border-radius: 3px;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #fff;
        white-space: nowrap;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
    }
    .hdr-live   { background: #0d7a4e; border: 1px solid #1aff8a22; }
    .hdr-pause  { background: #7a5a00; border: 1px solid #ffb30022; color: #ffd; }
    .hdr-chat   { background: #3a2a7a; border: 1px solid #8855ff22; }
    .hdr-sep { width: 1px; height: 18px; background: rgba(255,255,255,0.08); flex-shrink: 0; }

    /* -- Stat blocks in header -- */
    .hdr-stat {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 52px;
        flex-shrink: 0;
    }
    .hdr-stat-val {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ddeef8;
        line-height: 1.15;
        font-family: 'IBM Plex Mono', monospace;
    }
    .hdr-stat-lbl {
        font-size: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.3);
        font-weight: 500;
    }
    .hdr-trend {
        font-size: 0.62rem;
        font-weight: 700;
        line-height: 1;
    }

    /* -- Semantic colors -- */
    .delta-good    { color: #1aff8a; }
    .delta-bad     { color: #ff4d4d; }
    .delta-warn    { color: #ffb300; }
    .delta-neutral { color: rgba(255,255,255,0.25); }
    .delta-up      { color: #ff6b6b; }
    .delta-down    { color: #1aff8a; }

    /* -- Alert banners -- */
    .alert-banner {
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    .alert-crit {
        background: rgba(255, 60, 60, 0.12);
        border: 1px solid rgba(255, 60, 60, 0.4);
        color: #ffaaaa;
    }
    .alert-warn {
        background: rgba(255, 179, 0, 0.10);
        border: 1px solid rgba(255, 179, 0, 0.35);
        color: #ffd580;
    }
    .alert-ok {
        background: rgba(26, 255, 138, 0.07);
        border: 1px solid rgba(26, 255, 138, 0.25);
        color: #88ffbb;
    }
    .alert-msg   { font-weight: 600; }
    .alert-hint  { font-size: 0.65rem; opacity: 0.75; font-style: italic; }

    /* -- Health gauge bars -- */
    .gauge-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 3px 0;
    }
    .gauge-label {
        font-size: 0.62rem;
        color: rgba(255,255,255,0.45);
        font-family: 'IBM Plex Mono', monospace;
        min-width: 88px;
        text-align: right;
    }
    .gauge-track {
        flex: 1;
        height: 5px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .gauge-val {
        font-size: 0.62rem;
        font-family: 'IBM Plex Mono', monospace;
        min-width: 48px;
        color: rgba(255,255,255,0.6);
    }
    .gauge-ok   { background: #1aff8a; }
    .gauge-warn { background: #ffb300; }
    .gauge-crit { background: #ff4d4d; }

    /* -- Chat -- */
    .chat-msg {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        line-height: 1.45;
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 3px;
        color: rgba(255,255,255,0.82);
    }
    .chat-bg        { background: rgba(0,200,255,0.04); border-left: 2px solid rgba(0,200,255,0.25); }
    .chat-user      { background: rgba(255,180,50,0.05); border-left: 2px solid rgba(255,180,50,0.35); }
    .chat-assistant { background: rgba(80,220,120,0.05); border-left: 2px solid rgba(80,220,120,0.25); }
    .chat-step      { color: rgba(255,255,255,0.2); font-size: 0.58rem; margin-right: 4px; }

    /* -- Tabs -- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important; background: transparent !important;
        padding: 0 !important; min-height: 28px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 3px 12px !important; font-size: 0.7rem !important;
        border-radius: 3px 3px 0 0 !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-bottom: none !important; min-height: 24px !important;
        color: rgba(255,255,255,0.4) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,200,255,0.07) !important;
        border-top: 2px solid #00c8ff !important;
        color: #fff !important; font-weight: 600 !important;
    }

    /* -- Sidebar -- */
    section[data-testid="stSidebar"] .block-container { padding: 0.4rem 0.7rem !important; }
    section[data-testid="stSidebar"] .stSlider label { font-size: 0.75rem !important; }
    .section-header {
        font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: rgba(255,255,255,0.3);
        margin: 9px 0 3px 0; padding-bottom: 3px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* -- Plotly -- */
    .stPlotlyChart { margin-bottom: 0 !important; }

    div[data-testid="stMetricValue"]  { font-size: 1.05rem !important; font-weight: 700 !important; line-height: 1.2 !important; }
    div[data-testid="stMetricLabel"]  { font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; color: rgba(255,255,255,0.35) !important; }
    div[data-testid="metric-container"] { padding: 0 !important; }
    .streamlit-expanderHeader { font-size: 0.72rem !important; padding: 4px 8px !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Load state
# =============================================================================

c     = get_cfg()
d     = {}
stats = {}

if os.path.exists(LOG_F):
    try:
        with open(LOG_F, "r") as f: d = json.load(f)
    except Exception: d = {}

# -- Normalize log packet -- old/new format compatibility ----------------------
# V4.4.4 and earlier: gate_sat / long_conv_sat = list per layer
# V4.4.5+:            gate_sat / gate_mean / long_gate_sat / long_gate_mean = scalar float
def _scalar(val):
    """Coerce list to mean float, or return float/None as-is."""
    if val is None:
        return None
    if isinstance(val, list):
        return float(np.mean(val)) if val else None
    try:
        return float(val)
    except Exception:
        return None

# Normalize scalar fields that could be lists in old checkpoints
for _key in ("gate_sat", "long_gate_sat", "gate_mean", "long_gate_mean",
             "decay_mean", "long_conv_sat"):
    d[_key] = _scalar(d.get(_key))

if os.path.exists(STAT_F):
    try:
        with open(STAT_F, "r") as f: stats = json.load(f)
    except Exception: stats = {}

sample_history = load_jsonl_tail(SAMPLE_HISTORY_F, limit=40)

step       = d.get("step", 0)
loss_val   = d.get("loss", 0)
avg_steps  = d.get("avg_steps", c.get("t_steps", 12))
max_steps  = c.get("t_steps", 12)
loss_h     = stats.get("loss", [])
turb_h     = stats.get("diff_turb", [])
current_lr = d.get("current_lr", c.get("lr", 3e-4))

# -- Delta loss ----------------------------------------------------------------
delta_loss = (loss_val - loss_h[-2]) if len(loss_h) > 1 else None

# -- Chat capture --------------------------------------------------------------
if st.session_state.pending_chat and not c.get("request_chat"):
    st.session_state.pending_chat = False
    chat_resp = d.get("chat_response", "")
    if chat_resp:
        chat_prompt = d.get("chat_prompt", "")
        chat_step = d.get("chat_step", d.get("step", "?"))
        chat_key = f"{chat_step}|{chat_prompt}|{chat_resp}"
        if st.session_state.last_chat_response_key != chat_key:
            st.session_state.last_chat_response_key = chat_key
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": chat_resp,
                "step": chat_step,
            })

# -- Compute alerts ------------------------------------------------------------
alerts = compute_alerts(d, stats, c)
gate_reg_val = d.get("gate_reg", 0.0)

# =============================================================================
# Auto-Pilot
# =============================================================================

auto_pilot  = c.get("auto_pilot", False)
cfg_updated = False
autopilot_state = dict(c.get("autopilot_state", {}))

if auto_pilot:
    cfg_updated, autopilot_state, autopilot_toasts = run_autopilot(c, d, stats, sample_history)
    if autopilot_toasts:
        for msg in autopilot_toasts:
            st.toast(msg)
    if cfg_updated:
        set_cfg(c)
else:
    autopilot_state = dict(c.get("autopilot_state", {}))


# =============================================================================
# HEADER
# =============================================================================

if c.get("request_chat"):
    badge_cls, badge_txt = "hdr-chat", "GEN"
elif c.get("pause"):
    badge_cls, badge_txt = "hdr-pause", "PAUSE"
else:
    badge_cls, badge_txt = "hdr-live",  "LIVE"

tokens_seen = d.get("tokens_seen", 0)

# Compute trends for header metrics
loss_arr, loss_tr = trend_for(loss_h, good_direction="down") if len(loss_h) > 5 else ("→", "delta-neutral")
turb_arr, turb_tr = trend_for(turb_h, good_direction="down") if len(turb_h) > 5 else ("→", "delta-neutral")

# LR color
lr_cls = "delta-warn" if current_lr < THRESHOLDS["lr_low"] * 10 else "delta-neutral"

# Loss color
loss_color = "#ff4d4d" if (delta_loss and delta_loss > 0.01) else ("#1aff8a" if (delta_loss and delta_loss < -0.001) else "#ddeef8")

# Turb color
turb_val = d.get("diff_turb", 0)
turb_color = "#ff4d4d" if turb_val > THRESHOLDS["turb_crit"] else ("#ffb300" if turb_val > THRESHOLDS["turb_warn"] else "#ddeef8")

gate_sat_val = d.get("gate_sat", None)
gate_color = "#ff4d4d" if (gate_sat_val and gate_sat_val > THRESHOLDS["gate_sat_crit"]) else \
             ("#ffb300" if (gate_sat_val and gate_sat_val > THRESHOLDS["gate_sat_warn"]) else "#ddeef8")

gate_sat_display = f"{gate_sat_val:.3f}" if gate_sat_val is not None else "—"
decay_display    = f"{d.get('decay_mean', 0):.4f}" if d.get("decay_mean") else "—"

header_html = "".join([
        '<div class="hdr-bar">',
        '<span class="hdr-title">FluidLM</span>',
        f'<span class="hdr-badge {badge_cls}">{badge_txt}</span>',
        '<div class="hdr-sep"></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{step:,}</span><span class="hdr-stat-lbl">Step</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val" style="color:{loss_color}">{loss_val:.4f}</span><span class="hdr-trend {loss_tr}">{loss_arr}</span><span class="hdr-stat-lbl">Loss</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{avg_steps:.1f}<span style="color:rgba(255,255,255,0.2)">/{max_steps}</span></span><span class="hdr-stat-lbl">Effort</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val" style="color:{turb_color}">{turb_val:.4f}</span><span class="hdr-trend {turb_tr}">{turb_arr}</span><span class="hdr-stat-lbl">Turb</span></div>',
        '<div class="hdr-sep"></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val" style="color:#00c8ff">{d.get("main_loss", 0):.4f}</span><span class="hdr-stat-lbl">Main CE</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{d.get("eq_loss", 0):.1e}</span><span class="hdr-stat-lbl">Eq Loss</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{gate_reg_val:.2e}</span><span class="hdr-stat-lbl">Gate Reg</span></div>',
        '<div class="hdr-sep"></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val" style="color:{gate_color}">{gate_sat_display}</span><span class="hdr-stat-lbl">Gate Sat</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{decay_display}</span><span class="hdr-stat-lbl">Decay</span></div>',
        '<div class="hdr-sep"></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val {lr_cls}">{current_lr:.2e}</span><span class="hdr-stat-lbl">LR</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{d.get("it_s", 0):.1f}</span><span class="hdr-stat-lbl">it/s</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{d.get("vram", 0):.0f}</span><span class="hdr-stat-lbl">VRAM MB</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{fmt_eta(d.get("eta", 0))}</span><span class="hdr-stat-lbl">ETA</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{fmt_num(tokens_seen)}</span><span class="hdr-stat-lbl">Tokens</span></div>',
        f'<div class="hdr-stat"><span class="hdr-stat-val">{d.get("epoch", 0)}</span><span class="hdr-stat-lbl">Epoch</span></div>',
        '</div>',
])
st.markdown(header_html, unsafe_allow_html=True)


# =============================================================================
# ALERT BANNERS  [DASH-1]
# =============================================================================

if alerts:
    for level, msg, hint in alerts:
        css = "alert-crit" if level == "crit" else "alert-warn"
        icon = "CRIT:" if level == "crit" else "WARN:"
        st.markdown(
            f'<div class="alert-banner {css}">'
            f'<span class="alert-msg">{icon} {msg}</span>'
            f'<span class="alert-hint">-> {hint}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
elif step > 200:
    st.markdown(
        '<div class="alert-banner alert-ok">'
        '<span class="alert-msg">OK: All signals within normal range</span>'
        '<span class="alert-hint">Training healthy</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# Equations expander
with st.expander("Equations & PDE", expanded=False):
    _dt  = c.get("dt", 0.1)
    _ts  = c.get("t_steps", 12)
    _eps = c.get("epsilon", 0.05)
    st.markdown(rf"""
**Master PDE:**  $u_{{t+1}} = \text{{LN}}(u_t + \Delta t \cdot [\nabla^2 + \text{{LongConv}} + R + \alpha h])$
**Turb < epsilon:** halt early -- Turing Equilibrium  ·  **dt x T:** {_dt} x {_ts} = {_dt*_ts:.2f}  ·  **Effective receptive field:** ~{int(_ts*65*4):,} tokens
""")


# =============================================================================
# CHART CONSTANTS
# =============================================================================

CHART_H        = 248
CHART_MARGIN   = dict(l=42, r=12, t=8, b=28)
CHART_TEMPLATE = "plotly_dark"
CHART_BG       = "rgba(0,0,0,0)"
PAPER_BG       = "rgba(0,0,0,0)"
AXIS_STYLE     = dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)")


# =============================================================================
# ROW 1 — Loss Curve + Waves
# =============================================================================

col_loss, col_waves = st.columns([6, 4], gap="small")
with col_loss:
    with st.container(border=True):
        _ca, _cb = st.columns([2, 5])
        with _ca:
            god_mode = st.toggle("God", key="god", label_visibility="collapsed")
        with _cb:
            WIN_OPTS = {"200": 4, "500": 10, "1k": 20, "2k": 40, "5k": 100, "All": None}
            win_lbl = st.select_slider(
                "win", options=list(WIN_OPTS.keys()),
                value="2k", key="win_sel", label_visibility="collapsed",
            )
        win_n = WIN_OPTS[win_lbl]

        def _w(arr):
            if not arr: return arr
            if win_n is None or len(arr) <= win_n: return arr
            return arr[-win_n:]

        if god_mode and len(loss_h) > 0:
            fig = go.Figure()
            sx_g = _w(stats.get("step", []))
            palette = {
                "Loss":   ("#00d4ff", "loss"),
                "Turb":   ("#ff8c00", "diff_turb"),
                "Effort": ("#ff69b4", "avg_steps"),
                "VRAM":   ("#9b59b6", "vram"),
                "LR":     ("#f1c40f", "lr"),
                "Speed":  ("#2ecc71", "it_s"),
            }
            for name, (color, key) in palette.items():
                raw = _w(stats.get(key, []))
                if raw and len(raw) == len(sx_g):
                    fig.add_trace(go.Scatter(
                        x=sx_g, y=norm_data(raw), mode="lines", name=name,
                        line=dict(color=color, width=1.2),
                        hovertemplate=f"{name}: %{{customdata:.4g}}<extra></extra>",
                        customdata=raw,
                    ))
            fig.update_layout(
                height=CHART_H, margin=CHART_MARGIN, template=CHART_TEMPLATE,
                plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                yaxis=dict(showticklabels=False, title="normalized", **AXIS_STYLE),
                xaxis=dict(title="", **AXIS_STYLE),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=9)),
                showlegend=True,
            )
            st.plotly_chart(fig, width="stretch")
        elif len(loss_h) > 0:
            steps_x = _w(stats.get("step", []))
            loss_y  = _w(stats.get("loss", []))
            main_y  = _w(stats.get("main_loss", []))
            eq_y    = _w(stats.get("eq_loss", []))
            turb_y  = _w(stats.get("diff_turb", []))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps_x, y=loss_y, mode="lines", name="Total Loss",
                line=dict(color="#00d4ff", width=1.5),
                hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ))
            if main_y and len(main_y) == len(steps_x):
                fig.add_trace(go.Scatter(
                    x=steps_x, y=main_y, mode="lines",
                    line=dict(color="#ff9f43", width=1, dash="dot"), name="Main CE",
                    hovertemplate="CE: %{y:.4f}<extra></extra>",
                ))
            if eq_y and len(eq_y) == len(steps_x) and any(v > 1e-6 for v in eq_y):
                fig.add_trace(go.Scatter(
                    x=steps_x, y=eq_y, mode="lines",
                    line=dict(color="#ff6b81", width=1, dash="dot"), name="Eq Loss",
                    hovertemplate="Eq: %{y:.2e}<extra></extra>",
                ))
            if turb_y and len(turb_y) == len(steps_x):
                fig.add_trace(go.Scatter(
                    x=steps_x, y=turb_y, mode="lines", name="Turbulence",
                    line=dict(color="#ff8c00", width=1.1),
                    yaxis="y2", fill="tozeroy", fillcolor="rgba(255,140,0,0.05)",
                    hovertemplate="Turb: %{y:.4f}<extra></extra>",
                ))
                turb_max = max(0.25, max(turb_y) * 1.2)
                for thr, col_ in [
                    (THRESHOLDS["turb_warn"], "rgba(255,179,0,0.35)"),
                    (THRESHOLDS["turb_crit"], "rgba(255,77,77,0.35)"),
                ]:
                    fig.add_shape(type="line", xref="paper", yref="y2",
                                  x0=0, x1=1, y0=thr, y1=thr,
                                  line=dict(color=col_, width=1, dash="dot"))
            else:
                turb_max = 0.25

            fig.update_layout(
                height=CHART_H, margin=CHART_MARGIN, template=CHART_TEMPLATE,
                plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                xaxis=dict(title="", **AXIS_STYLE),
                yaxis=dict(title="Loss", **AXIS_STYLE, side="left"),
                yaxis2=dict(
                    title=dict(text="Turb", font=dict(size=9, color="#ff8c00")),
                    overlaying="y", side="right",
                    showgrid=False, zeroline=False,
                    tickfont=dict(size=8, color="#ff8c00"),
                    range=[0, turb_max],
                ),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=9)),
                showlegend=True,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Waiting for training data...")

with col_waves:
    with st.container(border=True):
        if d.get("waves"):
            fig_w = px.imshow(
                np.array(d["waves"]), color_continuous_scale="Magma",
                aspect="auto", labels=dict(x="Position", y="Step", color="|u|"),
            )
            fig_w.update_layout(
                height=CHART_H, margin=dict(l=28, r=8, t=8, b=28),
                coloraxis_showscale=False,
                plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
            )
            fig_w.update_xaxes(title="", **AXIS_STYLE)
            fig_w.update_yaxes(title="", **AXIS_STYLE)
            st.plotly_chart(fig_w, width="stretch")
        else:
            st.caption("Waves will appear at the next log step (every 50 steps)")


# =============================================================================
# ROW 2 — Health Monitor | Effort Trend | Chat
# =============================================================================

col_health, col_effort, col_chat = st.columns([3, 3, 4], gap="small")

# -- Health Monitor  [DASH-4] --------------------------------------------------
with col_health:
    with st.container(border=True):
        tabs = st.tabs(["Health", "Embed"])

        with tabs[0]:
            # Build gauges for key health signals
            def gauge_html(label, val, lo_warn, lo_crit, hi_warn, hi_crit,
                           fmt=".3f", pct_of=1.0):
                """
                Renders a labeled gauge bar.
                For gate_sat: 0 is perfect, 1 is dead.
                For decay:    0.92–0.999 is target range.
                """
                if val is None:
                    return f'<div class="gauge-row"><span class="gauge-label">{label}</span><span style="font-size:0.62rem;color:rgba(255,255,255,0.2)">no data</span></div>'
                fill_pct = min(100, max(0, (val / pct_of) * 100))
                if hi_crit is not None and val > hi_crit:
                    cls = "gauge-crit"
                elif hi_warn is not None and val > hi_warn:
                    cls = "gauge-warn"
                elif lo_crit is not None and val < lo_crit:
                    cls = "gauge-crit"
                elif lo_warn is not None and val < lo_warn:
                    cls = "gauge-warn"
                else:
                    cls = "gauge-ok"
                return (
                    f'<div class="gauge-row">'
                    f'<span class="gauge-label">{label}</span>'
                    f'<div class="gauge-track"><div class="gauge-fill {cls}" style="width:{fill_pct:.0f}%"></div></div>'
                    f'<span class="gauge-val">{val:{fmt}}</span>'
                    f'</div>'
                )

            gs  = d.get("gate_sat", None)
            lgs = d.get("long_gate_sat", None)
            gm  = d.get("gate_mean", None)
            lgm = d.get("long_gate_mean", None)
            dm  = d.get("decay_mean", None)
            turb_now = d.get("diff_turb", 0)
            gate_reg_now = d.get("gate_reg", 0.0)

            # dt per layer
            dt_vals   = d.get("dt_learned", [])
            decay_vals = d.get("decay_vals", [])

            health_html = ""
            health_html += gauge_html("gate_sat",       gs,   None, None, THRESHOLDS["gate_sat_warn"], THRESHOLDS["gate_sat_crit"], ".3f", 1.0)
            health_html += gauge_html("long_gate_sat",  lgs,  None, None, THRESHOLDS["gate_sat_warn"], THRESHOLDS["gate_sat_crit"], ".3f", 1.0)
            health_html += gauge_html("gate_mean",      gm,   0.05, None, 0.95, None,                  ".3f", 1.0)
            health_html += gauge_html("long_gate_mean", lgm,  0.05, None, 0.95, None,                  ".3f", 1.0)
            health_html += gauge_html("decay_mean",     dm,   THRESHOLDS["decay_low"], None, THRESHOLDS["decay_high"], None, ".4f", 1.0)
            health_html += gauge_html("turbulence",     turb_now, None, None, THRESHOLDS["turb_warn"], THRESHOLDS["turb_crit"], ".4f", 0.3)
            health_html += gauge_html("gate_reg",       gate_reg_now, None, None, 0.02, 0.05, ".2e", 0.05)

            st.markdown(health_html, unsafe_allow_html=True)

            # Per-layer dt and decay
            if dt_vals:
                dt_str    = " · ".join([f"L{i}:{v}" for i, v in enumerate(dt_vals)])
                decay_str = " · ".join([f"L{i}:{v}" for i, v in enumerate(decay_vals)]) if decay_vals else ""
                st.markdown(
                    f'<div style="font-size:0.58rem;color:rgba(255,255,255,0.2);margin-top:4px;font-family:IBM Plex Mono,monospace">'
                    f'dt: {dt_str}<br>'
                    f'{"decay: " + decay_str if decay_str else ""}</div>',
                    unsafe_allow_html=True,
                )

        with tabs[1]:
            if "w_hist" in d:
                fig_e = go.Figure(data=[go.Bar(
                    x=d["w_bins"][:-1], y=d["w_hist"],
                    marker_color="#00ffcc", marker_line_width=0,
                    hovertemplate="W: %{x:.3f}<br>N: %{y:,}<extra></extra>",
                )])
                fig_e.update_layout(
                    height=160, margin=dict(l=28, r=6, t=4, b=20),
                    template=CHART_TEMPLATE, plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                    xaxis=dict(title="", **AXIS_STYLE, tickfont=dict(size=8)),
                    yaxis=dict(title="", **AXIS_STYLE, tickfont=dict(size=8)),
                    bargap=0.04,
                )
                st.plotly_chart(fig_e, width="stretch")
                em = d.get("emb_mean", 0)
                es = d.get("emb_std",  0)
                st.markdown(
                    f'<div style="font-size:0.58rem;color:rgba(255,255,255,0.25);text-align:center;font-family:IBM Plex Mono,monospace">'
                    f'μ={em:.4f} · σ={es:.4f} · ↓{d.get("emb_min",0):.3f} · ↑{d.get("emb_max",0):.3f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Embedding data at next log step")


# -- Effort trend --------------------------------------------------------------
with col_effort:
    with st.container(border=True):
        steps_data = stats.get("avg_steps", [])
        if len(steps_data) > 2:
            fig_eff = go.Figure()
            sx = stats.get("step", [])[-400:]
            sy = steps_data[-400:]
            turb_s = stats.get("diff_turb", [])[-400:]

            # Effort area
            fig_eff.add_trace(go.Scatter(
                x=sx, y=sy, mode="lines", name="Effort",
                line=dict(color="#ff69b4", width=1.4),
                fill="tozeroy", fillcolor="rgba(255,105,180,0.07)",
                hovertemplate="Step %{x}<br>Effort: %{y:.1f}<extra></extra>",
            ))

            # Turbulence on secondary axis
            if len(turb_s) == len(sx):
                fig_eff.add_trace(go.Scatter(
                    x=sx, y=turb_s, mode="lines", name="Turb",
                    line=dict(color="#ff8c00", width=1.2),
                    yaxis="y2",
                    hovertemplate="Turb: %{y:.4f}<extra></extra>",
                ))
                # Warn/crit horizontal lines on y2
                for thresh, color, label in [
                    (THRESHOLDS["turb_warn"], "rgba(255,179,0,0.35)", f"warn"),
                    (THRESHOLDS["turb_crit"], "rgba(255,77,77,0.35)",  f"crit"),
                ]:
                    fig_eff.add_shape(
                        type="line", xref="paper", yref="y2",
                        x0=0, x1=1, y0=thresh, y1=thresh,
                        line=dict(color=color, width=1, dash="dot"),
                    )

            fig_eff.add_hline(
                y=max_steps, line_dash="dot",
                line_color="rgba(255,255,255,0.12)",
                annotation_text=f"max={max_steps}",
                annotation_font_size=8,
                annotation_font_color="rgba(255,255,255,0.25)",
            )

            fig_eff.update_layout(
                height=185, margin=dict(l=36, r=36, t=6, b=22),
                template=CHART_TEMPLATE, plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                xaxis=dict(title="", **AXIS_STYLE, tickfont=dict(size=8)),
                yaxis=dict(title="", range=[0, max_steps + 1.5], **AXIS_STYLE, tickfont=dict(size=8)),
                yaxis2=dict(
                    overlaying="y", side="right", showgrid=False, zeroline=False,
                    tickfont=dict(size=7, color="#ff8c00"),
                    range=[0, max(THRESHOLDS["turb_crit"] * 2, max(turb_s or [0.3]))],
                ),
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center",
                            font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
                showlegend=True,
            )
            st.plotly_chart(fig_eff, width="stretch")

            turb_now2 = d.get("diff_turb", 0)
            dr        = d.get("diff_reg", 0)
            eps       = c.get("epsilon", 0.05)
            effort_tr, effort_cls = trend_for(steps_data, good_direction="down")
            turb_tr2,  turb_cls2  = trend_for(stats.get("diff_turb", []), good_direction="down")
            st.markdown(
                f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.28);text-align:center;font-family:IBM Plex Mono,monospace">'
                f'turb=<span class="{turb_cls2}">{turb_now2:.4f} {turb_tr2}</span> · '
                f'effort=<span class="{effort_cls}">{avg_steps:.1f} {effort_tr}</span> · '
                f'ε={eps}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Effort trend at next log step")


# -- Chat + Samples ------------------------------------------------------------
with col_chat:
    with st.container(border=True):
        sample_text = d.get("sample", "")

        chat_tabs = st.tabs(["Prompt", "Samples"])

        with chat_tabs[0]:
            chat_html = ""
            for msg in st.session_state.chat_history[-8:]:
                if msg["role"] == "user":
                    txt = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                    chat_html += f'<div class="chat-msg chat-user">▸ {txt}</div>'
                elif msg["role"] == "assistant":
                    s = msg.get("step", "?")
                    txt = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                    chat_html += f'<div class="chat-msg chat-assistant"><span class="chat-step">step {s}</span>◂ {txt}</div>'

            if st.session_state.pending_chat:
                chat_html += '<div class="chat-msg chat-assistant" style="opacity:0.4;">Generating...</div>'
            if not chat_html:
                chat_html = '<div class="chat-msg" style="color:rgba(255,255,255,0.15);">Prompts and responses will appear here...</div>'

            st.markdown(
                f'<div style="height:192px;overflow-y:auto;overflow-x:hidden;'
                f'scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.08) transparent;">'
                f'{chat_html}</div>',
                unsafe_allow_html=True,
            )

        with chat_tabs[1]:
            if sample_text:
                latest_sample = sample_text.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    '<div class="chat-msg chat-bg">'
                    f'<span class="chat-step">latest training sample · step {step}</span>{latest_sample}'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if sample_history:
                hist_steps = [entry.get("step", 0) for entry in sample_history]
                hist_loss = [entry.get("loss", 0.0) for entry in sample_history]
                hist_turb = [entry.get("diff_turb", 0.0) for entry in sample_history]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist_steps, y=hist_loss, mode="lines+markers", name="Loss",
                    line=dict(color="#00d4ff", width=1.3), marker=dict(size=4),
                    hovertemplate="step %{x}<br>loss %{y:.4f}<extra></extra>",
                ))
                fig_hist.add_trace(go.Scatter(
                    x=hist_steps, y=hist_turb, mode="lines", name="Turb", yaxis="y2",
                    line=dict(color="#ff8c00", width=1.1, dash="dot"),
                    hovertemplate="turb %{y:.4f}<extra></extra>",
                ))
                fig_hist.update_layout(
                    height=110, margin=dict(l=26, r=26, t=6, b=20),
                    template=CHART_TEMPLATE, plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                    xaxis=dict(title="", **AXIS_STYLE, tickfont=dict(size=8)),
                    yaxis=dict(title="", **AXIS_STYLE, tickfont=dict(size=8)),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False,
                                tickfont=dict(size=7, color="#ff8c00")),
                    legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center",
                                font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_hist, width="stretch")

                history_html = ""
                for entry in reversed(sample_history[-8:]):
                    ts = entry.get("timestamp")
                    ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "--:--:--"
                    s = entry.get("step", "?")
                    loss_i = entry.get("loss", 0.0)
                    turb_i = entry.get("diff_turb", 0.0)
                    effort_i = entry.get("avg_steps", 0.0)
                    txt = entry.get("sample", "").replace("<", "&lt;").replace(">", "&gt;")
                    history_html += (
                        f'<div class="chat-msg chat-bg">'
                        f'<span class="chat-step">{ts_str} · step {s} · loss {loss_i:.4f} · turb {turb_i:.4f} · effort {effort_i:.1f}</span>'
                        f'{txt}</div>'
                    )

                st.markdown(
                    f'<div style="height:120px;overflow-y:auto;overflow-x:hidden;'
                    f'scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.08) transparent;">'
                    f'{history_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("sample_history.jsonl will be visible after the next log steps")

        if u_in := st.chat_input("Prompt…", disabled=st.session_state.pending_chat, key="chat_main"):
            st.session_state.chat_history.append({"role": "user", "content": u_in})
            st.session_state.pending_chat = True
            _c = get_cfg()
            _c.update({"chat_prompt": u_in, "request_chat": True, "pause": True})
            set_cfg(_c)
            st.rerun()


# =============================================================================
# SIDEBAR — Controls
# =============================================================================

with st.sidebar:
    st.markdown("### Controls")

    st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
    new_batch = st.slider("Batch",   8, 256, c.get("batch_size", 32),  key="sb_batch")
    new_seq   = st.slider("Seq Len", 32, 512, c.get("seq_len", 128),   key="sb_seq")
    d_mod     = st.number_input("d_model", 128, 1024, c.get("d_model", 512), 128, key="sb_dmodel")

    st.markdown('<div class="section-header">PDE Physics</div>', unsafe_allow_html=True)

    # LR: number_input instead of slider (1e-6 to 1e-3 range is impractical on a linear slider)
    lr_val = st.number_input(
        "LR", min_value=1e-6, max_value=1e-3,
        value=float(c.get("lr", 3e-4)),
    format="%.6f", step=1e-5,
    key="sb_lr"
)
    t_st    = st.slider("T Steps",   4, 32, c.get("t_steps", 12),            key="sb_tsteps")
    dt_val  = st.slider("Δt",        0.01, 0.5, c.get("dt", 0.1),            key="sb_dt")
    eps_choices = epsilon_options(c.get("epsilon", 0.05))
    eps_val = st.select_slider("ε", options=eps_choices, value=float(c.get("epsilon", 0.05)), key="sb_eps")
    eq_w    = st.slider("eq_weight", 0.0, 2.0, float(c.get("eq_weight", 0.01)), 0.01, key="sb_eqw")

    st.markdown('<div class="section-header">Generation</div>', unsafe_allow_html=True)
    temp = st.slider("Temperature",  0.1, 1.5, c.get("temperature", 0.8),          key="sb_temp")
    pen  = st.slider("Rep Penalty",  1.0, 2.5, c.get("repetition_penalty", 1.5),   key="sb_pen")

    st.markdown('<div class="section-header">Control</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: ui_live = st.checkbox("Live",       c.get("ui_live",    True),  key="sb_live")
    with c2: ap      = st.checkbox("Auto-Pilot", c.get("auto_pilot", False), key="sb_ap")
    pause = st.checkbox("Pause", c.get("pause", False), key="sb_pause")

    if st.button("Apply", width="stretch", key="sb_apply"):
        c.update({
            "batch_size": new_batch, "seq_len": new_seq, "d_model": d_mod,
            "lr": lr_val, "t_steps": t_st, "dt": dt_val, "epsilon": eps_val,
            "temperature": temp, "repetition_penalty": pen, "eq_weight": eq_w,
            "ui_live": ui_live, "auto_pilot": ap, "pause": pause,
        })
        set_cfg(c)
        st.success("OK: Synced")

    if st.button("Save Checkpoint", width="stretch", key="sb_save"):
        c["save_now"] = True
        set_cfg(c)
        st.success("Saving...")

    # -- Alert summary in sidebar ------------------------------------------
    if alerts:
        st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
        for level, msg, hint in alerts:
            icon = "CRIT:" if level == "crit" else "WARN:"
            st.markdown(f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.55);margin:2px 0">{icon} {msg}</div>', unsafe_allow_html=True)

    if ap:
        st.markdown('<div class="section-header">Auto-Pilot</div>', unsafe_allow_html=True)
        ap_health = autopilot_state.get("health", "—")
        ap_quality = autopilot_state.get("quality", "—")
        ap_regime = ", ".join(autopilot_state.get("regime", [])[:3]) or "observe"
        ap_action = autopilot_state.get("last_action", "observe")
        st.markdown(
            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.6);line-height:1.5">'
            f'health: <span class="mono">{ap_health}</span><br>'
            f'sample: <span class="mono">{ap_quality}</span><br>'
            f'regime: <span class="mono">{ap_regime}</span><br>'
            f'last: <span class="mono">{ap_action}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Auto-refresh
# =============================================================================

if c.get("ui_live", True):
    time.sleep(LIVE_REFRESH_SECONDS)
    st.rerun()