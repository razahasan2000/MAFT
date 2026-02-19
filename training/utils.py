import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import random

def load_config(path="training/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_path(cfg, *parts):
    return os.path.join(cfg['paths']['base_dir'], *parts)

def log_print(msg, log_file):
    print(msg, flush=True)
    with open(log_file, "a") as f:
        f.write(f"{pd.Timestamp.now()} - {msg}\n")

# ------------------------------------------------------------------------------
# Advanced Augmentations
# ------------------------------------------------------------------------------
def mixup_data(x_dict, y, device, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_dict = {}
    for k, v in x_dict.items():
        if isinstance(v, torch.Tensor):
            v_index = index.to(v.device)
            if v.dim() == 1:
                mixed_dict[k] = lam * v + (1 - lam) * v[v_index]
            else:
                mixed_dict[k] = lam * v + (1 - lam) * v[v_index, :]
        else:
            mixed_dict[k] = v
            
    y_a, y_b = y, y[index]
    return mixed_dict, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# ------------------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------------------
def calculate_ece(probs, targets, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(targets)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def profile_model(model, loader, device):
    """Profiles inference latency and parameter count."""
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Warmup
    valid_keys = ['aerial', 'sar', 'acoustic', 'ais']
    dummy_input = {k: torch.randn(1, *(v.shape[1:])).to(device) for k, v in next(iter(loader)).items() if k in valid_keys}
    for _ in range(10): 
        with torch.no_grad(): _ = model(**dummy_input, force_no_dropout=True)
    
    # Latency check
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        with torch.no_grad(): _ = model(**dummy_input, force_no_dropout=True)
    latency = (time.time() - start_time) / iterations * 1000 # ms
    
    return num_params, latency
