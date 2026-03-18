#!/usr/bin/env python3
# =============================================================================
# FluidLM Training Engine -- V4.5.0
# =============================================================================
#
# V4.5.0 (architecture upgrade):
#   [ARCH-10] SwiGLU replaces GELU MLP (LLaMA/PaLM proven activation)
#   [ARCH-11] Multi-Head CausalLongConv (K=33/65/129/257 per head)
#   [ARCH-12] RMSNorm replaces LayerNorm (more stable, fewer params)
#   [ARCH-13] Cross-channel mixing after depthwise conv
#   [ARCH-14] Persistent h-state across segments (optional)
#   [ARCH-15] Residual scaling 1/sqrt(num_layers) on PDE update
#   [TRAIN-8] seq_len 256, grad_accum 4 (effective batch 128)
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
import math
import numpy as np
from tqdm import tqdm
import tiktoken
from collections import deque

try:
    from src.core import FluidNet
except ImportError:
    from src.core import FluidNet

# -- Device selection ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- File paths ---------------------------------------------------------------
CONFIG_FILE = "config.json"
LOG_FILE = "live_logs.json"
STATS_FILE = "training_stats.json"
SAMPLE_HISTORY_FILE = "sample_history.jsonl"

SAVE_PATH = "./training"
MODEL_FILE = os.path.join(SAVE_PATH, "fluidlm_model.pth")


# =============================================================================
# Utility Functions
# =============================================================================

def atomic_write(data, path, retries=3):
    for attempt in range(retries):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"WARNING: Failed to write {path} after {retries} attempts: {e}")
            time.sleep(0.1 * (attempt + 1))
    return False


def append_jsonl(data, path):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"WARNING: Failed to append {path}: {e}")
        return False


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "lr": 3e-4,
            "batch_size": 32,
            "seq_len": 256,
            "d_model": 512,
            "t_steps": 8,
            "dt": 0.1,
            "repetition_penalty": 1.5,
            "temperature": 0.8,
            "grad_accum_steps": 4,
            "warmup_steps": 500,
            "epsilon": 0.05,
            "eq_weight": 0.01,
            "gate_reg_weight": 0.08,
            "total_steps": 50000,
            "grad_loss_weight": 0.005,
            "curriculum_steps": 5000,
        }


# =============================================================================
# Tokenizer
# =============================================================================

class BPETokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        return torch.tensor(
            self.enc.encode(text, allowed_special="all"), dtype=torch.long
        )

    def decode(self, tokens):
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
    def __init__(self, path, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.stride = seq_len // 2

        files = glob.glob(os.path.join(path, "*.txt"))
        if not files:
            raise ValueError(
                f"ERROR: No .txt files found in {path}/\n"
                f"   Please add text files to the /data/ directory."
            )

        raw = ""
        for f in files:
            with open(f, "r", encoding="utf-8") as fl:
                raw += fl.read() + "\n"

        print("Tokenizing dataset (BPE)...")
        self.data = self.tokenizer.encode(raw)
        print(f"Dataset loaded: {len(self.data):,} tokens.")

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        i = idx * self.stride
        return self.data[i : i + self.seq_len], self.data[i + 1 : i + self.seq_len + 1]


# =============================================================================
# Text Generation
# =============================================================================

def generate_text(model, tokenizer, config, start_str="The ", max_tokens=60):
    model.eval()

    with torch.no_grad():
        if not start_str:
            start_str = " "
        enc = tokenizer.encode(start_str)
        if enc.numel() == 0:
            enc = torch.tensor([0], dtype=torch.long)
        chars = enc.unsqueeze(0).to(DEVICE)

        history_window = deque(maxlen=int(config.get("repeat_window", 80)))
        top_k = max(0, int(config.get("top_k", 40)))
        top_p = float(config.get("top_p", 0.92))
        min_new_tokens = max(1, int(config.get("min_new_tokens", min(24, max_tokens))))
        no_repeat_ngram_size = max(0, int(config.get("no_repeat_ngram_size", 3)))
        inference_epsilon = float(config.get("inference_epsilon", min(float(config.get("epsilon", 1e-4)), 0.01)))
        eot_token_id = getattr(getattr(tokenizer, "enc", None), "eot_token", None)

        def apply_top_k_top_p(logits: torch.Tensor) -> torch.Tensor:
            filtered = logits

            if top_k > 0 and top_k < filtered.shape[-1]:
                threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
                filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False

                remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
                remove_mask.scatter_(1, sorted_indices, sorted_remove)
                filtered = filtered.masked_fill(remove_mask, float("-inf"))

            return filtered

        def banned_ngram_tokens(token_ids):
            if no_repeat_ngram_size <= 1 or len(token_ids) < no_repeat_ngram_size - 1:
                return set()

            prefix = tuple(token_ids[-(no_repeat_ngram_size - 1):])
            banned = set()
            limit = len(token_ids) - no_repeat_ngram_size + 1
            for idx in range(max(0, limit)):
                if tuple(token_ids[idx : idx + no_repeat_ngram_size - 1]) == prefix:
                    banned.add(token_ids[idx + no_repeat_ngram_size - 1])
            return banned

        for _ in range(max_tokens):
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Note: dt=None is correct — dt_val is learned via dt_gate,
                # the config dt only sets the initialization point [FIX-2]
                out = model(
                    chars[:, -config["seq_len"] :],
                    config["t_steps"],
                    None,   # dt is learned, not passed at inference
                    epsilon=inference_epsilon,
                )
                logits = out[0]
                logits = logits[:, -1, :] / max(float(config["temperature"]), 1e-4)

            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            for idx in set(history_window):
                if logits[0, idx] < 0:
                    logits[0, idx] *= config["repetition_penalty"]
                else:
                    logits[0, idx] /= config["repetition_penalty"]

            for idx in banned_ngram_tokens(chars[0].tolist()):
                logits[0, idx] = float("-inf")

            generated_tokens = chars.shape[1] - enc.numel()
            if eot_token_id is not None and generated_tokens < min_new_tokens:
                logits[0, eot_token_id] = float("-inf")

            logits = apply_top_k_top_p(logits)

            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                probs = F.softmax(torch.nan_to_num(logits, nan=-1e9, posinf=50.0, neginf=-1e9), dim=-1)

            next_c = torch.multinomial(probs, 1)
            chars = torch.cat([chars, next_c], dim=1)

            token_id = next_c.item()
            history_window.append(token_id)
            if eot_token_id is not None and token_id == eot_token_id and generated_tokens >= min_new_tokens:
                break

    model.train()
    return tokenizer.decode(chars[0].tolist())


# =============================================================================
# Data Loader Builder
# =============================================================================

def build_loader(cfg, tokenizer):
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
# [TRAIN-6] Laplacian Grad Loss (ported from FluidWorld)
# =============================================================================

def compute_grad_loss(model, d_model, device):
    """
    Compute 1D Laplacian smoothness loss on the final hidden representation.
    Penalizes high-frequency noise along the sequence dimension, acting as an
    implicit spatial regularizer. This is the mechanism behind FluidWorld's
    superior rollout coherence over Transformer and ConvLSTM baselines.
    """
    hidden = getattr(model, "_last_hidden", None)
    if hidden is None:
        return torch.tensor(0.0, device=device)
    h_t = hidden.transpose(1, 2)  # (B, D, L)
    kernel = torch.tensor([1.0, -2.0, 1.0], device=device).view(1, 1, 3)
    kernel = kernel.expand(d_model, 1, 3)
    h_padded = F.pad(h_t, (1, 1), mode="constant", value=0.0)
    lap = F.conv1d(h_padded, kernel, groups=d_model)
    return lap.abs().mean()


# =============================================================================
# [TRAIN-7] Curriculum Scheduler (ported from FluidWorld)
# =============================================================================

def get_curriculum_value(step, start_val, end_val, curriculum_steps):
    """
    Linear ramp from start_val to end_val over curriculum_steps.
    Returns end_val for all steps beyond curriculum_steps.
    """
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return end_val
    progress = step / curriculum_steps
    return start_val + (end_val - start_val) * progress


# =============================================================================
# [TRAIN-1] Cosine Learning Rate Schedule
# =============================================================================

def get_lr(step, max_lr, warmup_steps, total_steps=50000):
    """
    Linear warmup → Cosine annealing to max_lr / 10.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return max_lr * 0.1
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# =============================================================================
# Optimizer Builder — Differential LR per dilation
# =============================================================================

def build_optimizer(model, cfg):
    base_lr = cfg["lr"]

    diff_params_d1 = []
    diff_params_d4 = []
    diff_params_d16 = []
    ssm_params = []
    other_params = []

    for name, p in model.named_parameters():
        if "diff_coeffs.0" in name:
            diff_params_d1.append(p)
        elif "diff_coeffs.1" in name:
            diff_params_d4.append(p)
        elif "diff_coeffs.2" in name:
            diff_params_d16.append(p)
        elif "ssm" in name:
            ssm_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params,      "lr": base_lr,       "lr_mult": 1},
        {"params": diff_params_d1,    "lr": base_lr * 10,  "lr_mult": 10},
        {"params": diff_params_d4,    "lr": base_lr * 50,  "lr_mult": 50},
        {"params": diff_params_d16,   "lr": base_lr * 100, "lr_mult": 100},
        {"params": ssm_params,        "lr": base_lr * 2,   "lr_mult": 2},
    ]

    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def update_lr(opt, current_lr):
    for g in opt.param_groups:
        g["lr"] = current_lr * g.get("lr_mult", 1)


# =============================================================================
# Main Training Loop
# =============================================================================

def train():
    os.makedirs(SAVE_PATH, exist_ok=True)
    cfg = load_config()
    tokenizer = BPETokenizer()

    # [FIX-2] Pass init_dt from config so the dt_gate starts at the right value
    model = FluidNet(
        tokenizer.vocab_size,
        cfg["d_model"],
        init_dt=cfg.get("dt", 0.1),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ssm_param_count = sum(
        p.numel() for n, p in model.named_parameters() if "ssm" in n
    )

    print(f"\n{'='*60}")
    print(f"  FluidLM V4.5.0")
    print(f"{'='*60}")
    print(f"  d_model:        {cfg['d_model']}")
    print(f"  num_layers:     {len(model.layers)}")
    print(f"  Reaction:       SwiGLU (8/3 ratio)")
    print(f"  Normalization:  RMSNorm")
    print(f"  Positional:     Sinusoidal")
    print(f"  Long-range:     Selective SSM (Mamba, d_state=16, pure PyTorch)")
    print(f"  h-state:        (B, D) + Forget Gate + persistent (optional)")
    print(f"  Residual scale: 1/sqrt({len(model.layers)})")
    print(f"  Grad clip:      Layer-wise x{len(model.layers)+2}")
    print(f"  Grad loss:      Laplacian smoothness w={cfg.get('grad_loss_weight', 0.005)}")
    print(f"  LR schedule:    Cosine {cfg['lr']} -> {cfg['lr']*0.1:.1e}")
    print(f"  seq_len:        {cfg['seq_len']}")
    print(f"  grad_accum:     {cfg.get('grad_accum_steps', 1)} (eff. batch {cfg['batch_size'] * cfg.get('grad_accum_steps', 1)})")
    print(f"  t_steps:        {cfg['t_steps']}")
    print(f"  eq_weight:      {cfg.get('eq_weight', 0.01)}")
    print(f"  curriculum:     {cfg.get('curriculum_steps', 0)} steps")
    print(f"  init_dt:        {cfg.get('dt', 0.1)} (live recalibration active)")
    print(f"  Total params:      {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")
    print(f"  SSM params:        {ssm_param_count:,}")
    print(f"{'='*60}\n")

    opt = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    global_step = 0
    global_epoch = 0
    best_loss = float("inf")
    _last_dt = float(cfg.get("dt", 0.1))  # [FIX-7] last seen dt for live recalibration

    stats = {
        "step": [], "loss": [], "vram": [], "it_s": [],
        "lr": [], "temp": [], "penalty": [], "avg_steps": [],
        "main_loss": [], "eq_loss": [], "diff_loss": [], "total_loss": [],
        "diff_turb": [], "diff_reg": [], "gate_reg": [], "grad_loss": [],
        "tokens_seen": [],
    }

    # -- Resume from checkpoint -------------------------------------------
    if os.path.exists(MODEL_FILE):
        print("Found existing checkpoint...")
        ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
        old_state = ckpt["model_state"] if isinstance(ckpt, dict) else ckpt

        has_rope = any("rope" in k for k in old_state.keys())
        has_long_conv = any("long_conv" in k for k in old_state.keys())
        # V4.4.0 used memory_gate (single Sequential), V4.4.1+ uses memory_gate_x + memory_gate_h
        # V4.4.2: h shape (B,L,D) -> (B,D), but Linear shapes D->D unchanged -> compatible
        has_old_memory_gate = any(k.endswith("memory_gate.0.weight") for k in old_state.keys())

        if has_rope and not has_long_conv:
            print("  V4.3 checkpoint detected -- partial migration:")
            print("     OK: Embedding, Reaction MLP, Diffusion coefficients loaded")
            print("     SKIP: RoPE keys skipped (replaced by sinusoidal PE)")
            print("     SKIP: CausalLongConv initialized fresh (new component)")

            new_state = model.state_dict()
            loaded_count = 0
            for key in old_state:
                if "rope" in key:
                    continue
                if key in new_state and old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
                    loaded_count += 1
            model.load_state_dict(new_state)
            print(f"     Loaded {loaded_count} weight tensors from V4.3")

        elif has_old_memory_gate:
            print("  Warning: V4.4.0 checkpoint detected -- migrating to V4.4.2+:")
            print("     (Sequential memory_gate -> memory_gate_x / memory_gate_h)")
            print("     Compatible weights loaded")
            print("     memory_gate_x / memory_gate_h: shapes D->D unchanged")
            print("     h-state (B,L,D) -> (B,D): no weight migration needed")
            print("     [FIX-6]: global reservoir O(1) restored")

            new_state = model.state_dict()
            loaded_count = 0
            for key in old_state:
                # Map old memory_gate.0.weight/bias → memory_gate_x
                mapped_key = key
                if "memory_gate.0.weight" in key:
                    mapped_key = key.replace("memory_gate.0.weight", "memory_gate_x.weight")
                elif "memory_gate.0.bias" in key:
                    mapped_key = key.replace("memory_gate.0.bias", "memory_gate_x.bias")
                # Skip memory_gate.1 (was Sigmoid, now we have no Sequential)
                elif "memory_gate.1" in key:
                    continue

                if mapped_key in new_state and old_state[key].shape == new_state[mapped_key].shape:
                    new_state[mapped_key] = old_state[key]
                    loaded_count += 1
            model.load_state_dict(new_state)
            print(f"     Loaded {loaded_count} weight tensors from V4.4.0")
            if isinstance(ckpt, dict):
                global_step = ckpt.get("step", 0)
                best_loss = ckpt.get("best_loss", float("inf"))
            print(f"     Resuming from step {global_step}")

        else:
            try:
                # V4.5.0 migration: load shape-matched weights, skip mismatches
                new_state = model.state_dict()
                loaded_count = 0
                skipped = []
                for key in old_state:
                    if key in new_state and old_state[key].shape == new_state[key].shape:
                        new_state[key] = old_state[key]
                        loaded_count += 1
                    elif key in new_state:
                        skipped.append(f"{key} ({list(old_state[key].shape)} -> {list(new_state[key].shape)})")
                    else:
                        skipped.append(f"{key} (removed)")
                model.load_state_dict(new_state)
                if skipped:
                    print(f"  V4.5.0 migration: {loaded_count} tensors loaded, {len(skipped)} skipped:")
                    for s in skipped[:10]:
                        print(f"    - {s}")
                    if len(skipped) > 10:
                        print(f"    ... and {len(skipped)-10} more")
                    print("  New components (SwiGLU, MultiHeadLongConv, RMSNorm) initialized fresh.")
                    # Reset optimizer since architecture changed
                    opt = build_optimizer(model, cfg)
                else:
                    # Full V4.5.0 checkpoint
                    if isinstance(ckpt, dict):
                        if "optimizer_state" in ckpt:
                            try:
                                opt.load_state_dict(ckpt["optimizer_state"])
                            except ValueError:
                                print("  Optimizer state incompatible, resetting.")
                        if "scaler_state" in ckpt:
                            scaler.load_state_dict(ckpt["scaler_state"])
                if isinstance(ckpt, dict):
                    global_step = ckpt.get("step", 0)
                    best_loss = ckpt.get("best_loss", float("inf"))
                print(f"  Checkpoint loaded (step {global_step})\n")
            except Exception as e:
                print(f"  WARNING: Could not load checkpoint: {e}")
                print("  Starting fresh.\n")

    # -- Verify diffusion coefficients ------------------------------------
    print("Checking diffusion coefficients...")
    needs_reinit = False
    with torch.no_grad():
        for layer in model.layers:
            for coeff in layer.diff_coeffs:
                if coeff.abs().mean().item() < 0.02:
                    needs_reinit = True

    if needs_reinit:
        print("Forcing diff_coeffs reinitialization...")
        with torch.no_grad():
            for layer in model.layers:
                layer.diff_coeffs[0].fill_(0.15)
                layer.diff_coeffs[1].fill_(0.10)
                layer.diff_coeffs[2].fill_(0.08)
        print("Coefficients reinitialized: [0.15, 0.10, 0.08]")
    else:
        print("Diffusion coefficients OK.")

    loader, ds = build_loader(cfg, tokenizer)
    opt.zero_grad(set_to_none=True)

    total_steps = cfg.get("total_steps", 50000)

    # -- Infinite training loop -------------------------------------------
    while True:
        pbar = tqdm(loader, desc=f"Epoch {global_epoch}")

        for i, (x, y) in enumerate(pbar):
            batch_start = time.time()

            current_lr = get_lr(
                global_step, cfg["lr"],
                cfg.get("warmup_steps", 500),
                total_steps,
            )
            update_lr(opt, current_lr)

            is_log_step = global_step > 0 and global_step % 50 == 0

            # -- Hot-reload configuration ---------------------------------
            if global_step > 0 and global_step % 10 == 0:
                new_cfg = load_config()
                if (new_cfg["batch_size"] != cfg["batch_size"] or
                        new_cfg["seq_len"] != cfg["seq_len"]):
                    cfg = new_cfg
                    loader, ds = build_loader(cfg, tokenizer)
                    break
                cfg = new_cfg

                # [FIX-7] If the dt slider changed, recalibrate dt_gate live on all
                # layers. The model remains free to adapt afterward -- we just give
                # it a new starting point without blocking the gradient.
                new_dt = cfg.get("dt")
                if new_dt is not None and abs(float(new_dt) - _last_dt) > 1e-6:
                    _last_dt = float(new_dt)
                    dt_clamped = max(0.002, min(0.199, float(new_dt)))
                    dt_recal = math.log((dt_clamped - 0.001) / (0.2 - dt_clamped))
                    with torch.no_grad():
                        for layer in model.layers:
                            layer.dt_gate.copy_(
                                torch.tensor(dt_recal,
                                             device=layer.dt_gate.device,
                                             dtype=layer.dt_gate.dtype)
                            )

            # -- Pause / Chat handling ------------------------------------
            if cfg.get("pause"):
                if cfg.get("request_chat"):
                    res = generate_text(
                        model, ds.tokenizer, cfg,
                        cfg["chat_prompt"], max_tokens=100,
                    )
                    try:
                        with open(LOG_FILE, "r") as f:
                            log_packet = json.load(f)
                    except Exception:
                        log_packet = {
                            "loss": 0, "step": global_step, "vram": 0,
                            "waves": [], "w_hist": [], "w_bins": [],
                            "avg_steps": 12,
                        }
                    log_packet["chat_prompt"] = cfg.get("chat_prompt", "")
                    log_packet["chat_response"] = res
                    log_packet["chat_step"] = global_step
                    atomic_write(log_packet, LOG_FILE)
                    cfg["request_chat"] = False
                    cfg["pause"] = False
                    atomic_write(cfg, CONFIG_FILE)
                time.sleep(0.5)
                continue

            # -- Forward pass ---------------------------------------------
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out_model = model(
                    x, cfg["t_steps"], None,
                    return_history=is_log_step,
                    epsilon=cfg.get("epsilon", 1e-4),
                )

                if is_log_step:
                    logits_raw, waves, avg_steps, diff_turb, gate_stats = out_model
                else:
                    logits_raw, avg_steps, diff_turb, gate_stats = out_model

                main_loss = nn.CrossEntropyLoss()(
                    logits_raw.view(-1, tokenizer.vocab_size), y.view(-1)
                )

                diffusion_reg = torch.tensor(0.0, device=DEVICE)
                for layer in model.layers:
                    for coeff in layer.diff_coeffs:
                        diffusion_reg = diffusion_reg + torch.relu(0.05 - F.softplus(coeff)).mean()

                # [TRAIN-7] Curriculum: ramp eq_weight and grad_loss_weight
                curriculum_steps = cfg.get("curriculum_steps", 0)
                eq_weight_target = cfg.get("eq_weight", 0.01)
                grad_loss_weight_target = cfg.get("grad_loss_weight", 0.005)
                if curriculum_steps > 0:
                    eq_weight = get_curriculum_value(global_step, eq_weight_target * 0.1, eq_weight_target, curriculum_steps)
                    grad_loss_weight = get_curriculum_value(global_step, 0.0, grad_loss_weight_target, curriculum_steps)
                else:
                    eq_weight = eq_weight_target
                    grad_loss_weight = grad_loss_weight_target

                gate_reg_weight = cfg.get("gate_reg_weight", 0.08)
                eq_loss = eq_weight * diff_turb
                diff_loss = 0.01 * diffusion_reg
                gate_reg = gate_reg_weight * gate_stats["total_gate_reg"]

                # [TRAIN-6] Laplacian smoothness loss on hidden representations
                grad_loss_val = compute_grad_loss(model, cfg["d_model"], DEVICE)
                grad_loss_term = grad_loss_weight * grad_loss_val

                total_loss = main_loss + eq_loss + diff_loss + gate_reg + grad_loss_term

                grad_accum = cfg.get("grad_accum_steps", 1)
                loss = total_loss / grad_accum

            # -- Backward pass --------------------------------------------
            scaler.scale(loss).backward()

            # -- Optimizer step -------------------------------------------
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                # [TRAIN-4] Layer-wise gradient clipping (Euler integrator stability).
                # A global clip lets one unstable layer consume the entire budget,
                # starving the rest -> loss spikes. Each component gets its own
                # max_norm=1.0 ceiling: embedding, each FluidLayer, head.
                torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), max_norm=1.0)
                for _layer in model.layers:
                    torch.nn.utils.clip_grad_norm_(_layer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            # -- Telemetry logging ----------------------------------------
            if is_log_step:
                t_delta = time.time() - batch_start
                it_s = 1.0 / t_delta if t_delta > 0 else 0.1
                eta_sec = (len(loader) - i) / it_s

                loss_val = float(total_loss.item())
                vram_val = (
                    torch.cuda.memory_reserved() / 1e6
                    if torch.cuda.is_available()
                    else 0
                )

                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(
                        {"model_state": model.state_dict(), "step": global_step,
                         "best_loss": best_loss},
                        MODEL_FILE.replace(".pth", "_best.pth"),
                    )

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

                tokens_seen = global_step * cfg["batch_size"] * cfg["seq_len"] * cfg.get("grad_accum_steps", 1)

                with torch.no_grad():
                    ew = model.embedding.weight
                    emb_mean = ew.mean().item()
                    emb_std = ew.std().item()
                    emb_min = ew.min().item()
                    emb_max = ew.max().item()

                    # [FIX-2] Log actual learned dt_val for each layer
                    dt_vals = []
                    memory_gate_temps = []
                    alpha_local_vals = []
                    for layer in model.layers:
                        dt_v = (0.001 + (0.2 - 0.001) * torch.sigmoid(layer.dt_gate)).item()
                        dt_vals.append(round(dt_v, 4))
                        memory_gate_temps.append(round((layer.memory_gate_temp.abs() + 1.0).item(), 4))
                        alpha_local_vals.append(round(F.softplus(layer.alpha_local_param).item(), 4))

                    # Gate saturation batch-based (from FluidLayer.forward())
                    gate_sat_val      = round(gate_stats["gate_sat"].item(), 4)
                    gate_mean_val     = round(gate_stats["gate_mean"].item(), 4)
                    decay_mean_val    = round(gate_stats["decay_mean"].item(), 4)

                    # Per-layer decay (learned viscosity) for fine-grained debug
                    decay_vals = []
                    for layer in model.layers:
                        dv = torch.sigmoid(layer.decay_param).mean().item()
                        decay_vals.append(round(dv, 4))

                    # diff_coeffs per-layer per-dilation -- detects long-range
                    # diffusion collapse (dilation 16 collapse in practice)
                    diff_coeff_snapshot = []
                    for layer in model.layers:
                        layer_coeffs = [
                            round(F.softplus(c).mean().item(), 5)
                            for c in layer.diff_coeffs
                        ]
                        diff_coeff_snapshot.append(layer_coeffs)

                # -- Update rolling statistics ----------------------------
                stats["step"].append(global_step)
                stats["loss"].append(loss_val)
                stats["vram"].append(vram_val)
                stats["it_s"].append(it_s)
                stats["lr"].append(current_lr)  # actual cosine-decayed LR
                stats["temp"].append(cfg["temperature"])
                stats["penalty"].append(cfg["repetition_penalty"])
                stats["avg_steps"].append(float(avg_steps))
                stats["main_loss"].append(float(main_loss.item()))
                stats["eq_loss"].append(float(eq_loss.item()))
                stats["diff_loss"].append(float(diff_loss.item()))
                stats["total_loss"].append(float(total_loss.item()))
                stats["diff_turb"].append(float(diff_turb.item()))
                stats["diff_reg"].append(float(diffusion_reg.item()))
                stats["gate_reg"].append(float(gate_reg.item()))
                stats["grad_loss"].append(float(grad_loss_val.item()))
                stats["tokens_seen"].append(int(tokens_seen))

                for k in stats.keys():
                    stats[k] = stats[k][-1000:]

                sample = generate_text(model, ds.tokenizer, cfg, max_tokens=30)

                w_vals = model.embedding.weight.detach().flatten().cpu().float().numpy()
                counts, bins = np.histogram(w_vals, bins=30, range=(-0.3, 0.3))

                log_packet = {
                    "epoch": global_epoch,
                    "step": global_step,
                    "loss": loss_val,
                    "it_s": it_s,
                    "eta": eta_sec,
                    "vram": vram_val,
                    "sample": sample,
                    "waves": [w[0].tolist() for w in waves] if is_log_step else [],
                    "w_hist": counts.tolist(),
                    "w_bins": bins.tolist(),
                    "timestamp": time.time(),
                    "avg_steps": float(avg_steps),

                    # [FIX-3] Now includes current learning rate
                    "current_lr": current_lr,

                    # Loss decomposition + health
                    "main_loss": float(main_loss.item()),
                    "eq_loss": float(eq_loss.item()),
                    "diff_loss": float(diff_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "diff_turb": float(diff_turb.item()),
                    "diff_reg": float(diffusion_reg.item()),
                    "gate_reg": float(gate_reg.item()),
                    "grad_loss": float(grad_loss_val.item()),
                    "grad_loss_weight": grad_loss_weight,
                    "tokens_seen": int(tokens_seen),

                    "dt_learned": dt_vals,
                    "memory_gate_temps": memory_gate_temps,
                    "alpha_local_vals": alpha_local_vals,

                    "gate_mean":       gate_mean_val,
                    "gate_sat":        gate_sat_val,
                    "decay_mean":      decay_mean_val,
                    "decay_vals":      decay_vals,
                    "gate_reg_components": {
                        "gate": float(gate_stats["gate_reg_loss"].item()),
                        "decay": float(gate_stats["decay_reg_loss"].item()),
                    },

                    "emb_mean": float(emb_mean),
                    "emb_std": float(emb_std),
                    "emb_min": float(emb_min),
                    "emb_max": float(emb_max),

                    # diff_coeffs [layer][dilation] -- softplus'd actual values
                    # Monitors dilation-16 diffusion collapse
                    "diff_coeff_snapshot": diff_coeff_snapshot,
                }
                atomic_write(log_packet, LOG_FILE)
                atomic_write(stats, STATS_FILE)
                append_jsonl(
                    {
                        "timestamp": log_packet["timestamp"],
                        "epoch": global_epoch,
                        "step": global_step,
                        "loss": loss_val,
                        "main_loss": float(main_loss.item()),
                        "eq_loss": float(eq_loss.item()),
                        "diff_turb": float(diff_turb.item()),
                        "avg_steps": float(avg_steps),
                        "current_lr": current_lr,
                        "gate_sat": gate_sat_val,
                        "decay_mean": decay_mean_val,
                        "grad_loss": float(grad_loss_val.item()),
                        "sample": sample,
                    },
                    SAMPLE_HISTORY_FILE,
                )

        else:
            global_epoch += 1


if __name__ == "__main__":
    train()