#!/usr/bin/env python3
# =============================================================================
# FluidLM Lab - Unified Launcher 
# =============================================================================
#
# This script orchestrates the two main processes of the FluidLM research lab:
#   1. The training engine  (train_engine.py)  - runs the PDE-based model
#   2. The research dashboard (web_app.py)     - Streamlit-based live telemetry
#
# Architecture choice: we use subprocess.Popen rather than multiprocessing
# because Streamlit needs its own process context (it spawns a Tornado server)
# and PyTorch's CUDA context doesn't play well with fork(). Two separate
# processes communicate via atomic JSON file writes - a deliberately simple
# IPC mechanism that avoids the complexity of sockets or shared memory while
# remaining robust to race conditions.
#
# The config.json file acts as a shared "blackboard" between the two processes:
# the dashboard writes hyperparameter changes, the engine reads them every
# ~10 optimizer steps. This polling approach adds negligible overhead (~0.1ms
# per read) and keeps both processes fully decoupled.
#
# Usage:
#   python launch_lab.py
#   Then navigate to http://localhost:8501
#
# Author: Fabien POLLY (Infinition)
# =============================================================================

import subprocess
import sys
import time
import os
import json


def atomic_write(data, path, retries=3):
    """
    Write JSON data to disk atomically using a temporary file + os.replace().

    Why atomic writes matter here:
    Both the training engine and the Streamlit dashboard read/write the same
    config file concurrently. A naive open()+write() could produce a half-
    written file that the other process reads, causing json.JSONDecodeError.
    The atomic write pattern (write to .tmp, then os.replace) guarantees that
    the target file is either the old version or the new version - never a
    partial/corrupted intermediate state.

    The retry logic handles rare filesystem contention on Windows (where
    os.replace can fail if the target is momentarily locked by another process).
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


def launch_laboratory():
    """
    Main entry point: initializes the default configuration (if missing),
    then spawns both the training engine and the Streamlit dashboard as
    child processes. Monitors their health in a polling loop and performs
    clean shutdown on KeyboardInterrupt (Ctrl+C) or if either process dies.
    """
    print("🌊 FluidLM Lab - Initializing research environment...")

    # ── Bootstrap default configuration ──────────────────────────────────
    # These defaults are chosen for a consumer GPU (RTX 3060/4070 class):
    #   - batch_size=32, seq_len=128: fits comfortably in 8-12GB VRAM
    #   - d_model=512: 36M param model - small enough to iterate fast
    #   - t_steps=12, dt=0.1: 12 Euler steps × 0.1 = integration time of 1.2
    #   - epsilon=1e-4: convergence threshold for Turing equilibrium
    #   - warmup_steps=500: linear LR warmup to stabilize early training
    if not os.path.exists("config.json"):
        default_config = {
            "lr": 1e-4,
            "batch_size": 32,
            "seq_len": 128,
            "d_model": 512,
            "t_steps": 12,
            "dt": 0.1,
            "repetition_penalty": 1.5,
            "temperature": 0.8,
            "pause": False,
            "request_chat": False,
            "chat_prompt": "",
            "save_now": False,
            "ui_live": True,
            "auto_pilot": False,
            "last_decay_step": 0,
            "epsilon": 0.05,
        }
        atomic_write(default_config, "config.json")
        print("   ✅ Default config.json created.")

    train_proc, ui_proc = None, None

    try:
        # ── Launch the training engine ───────────────────────────────────
        print("🚀 Starting training engine (train_engine.py)...")
        train_proc = subprocess.Popen([sys.executable, "train_engine.py"])

        # Brief delay to let the engine acquire the GPU and start writing
        # telemetry before the dashboard tries to read it.
        time.sleep(2)

        # ── Launch the Streamlit dashboard ───────────────────────────────
        print("🌊 Starting research dashboard (web_app.py)...")
        ui_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "web_app.py"]
        )
        print("\n✅ Lab is live → http://localhost:8501")

        # ── Health monitoring loop ───────────────────────────────────────
        # We poll every second to check if either child has crashed.
        # In production, you might replace this with signal handlers or
        # a proper process supervisor (systemd, supervisord, etc.).
        while True:
            time.sleep(1)
            if (train_proc and train_proc.poll() is not None) or \
               (ui_proc and ui_proc.poll() is not None):
                print("⚠️ A child process exited unexpectedly.")
                break

    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown requested (Ctrl+C)...")

    finally:
        # ── Cleanup ──────────────────────────────────────────────────────
        # SIGTERM gives each process a chance to flush buffers and save
        # checkpoints before the OS reclaims resources.
        print("🧹 Terminating child processes...")
        if train_proc and train_proc.poll() is None:
            train_proc.terminate()
        if ui_proc and ui_proc.poll() is None:
            ui_proc.terminate()
        print("💤 Lab shut down cleanly.")


if __name__ == "__main__":
    launch_laboratory()