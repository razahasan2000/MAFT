"""
Baseline Comparison Analysis
Generates comprehensive comparison tables and visualizations including memory footprint.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baselines.early_fusion_cnn import EarlyFusionCNN
from models.baselines.mid_fusion_mlp import MidFusionMLP
from models.baselines.decision_fusion import DecisionFusionEnsemble
from models.baselines.maft_no_dropout import MAFTNoDropout
from models.maft_v3 import MAFTv3
from training.utils import load_config, get_path, calculate_ece, profile_model, get_memory_footprint
from training.dataset import MaritimeDatasetV3

# Load config
CFG = load_config(os.path.join(os.path.dirname(__file__), '../training/config.yaml'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    preds, targets, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            # Handle different input signatures
            if isinstance(model, MAFTv3) or isinstance(model, MAFTNoDropout):
                logits = model(
                    aerial=batch['aerial'].to(device),
                    sar=batch['sar'].to(device),
                    acoustic=batch['acoustic'].to(device),
                    ais=batch['ais'].to(device),
                    force_no_dropout=True
                )
            elif isinstance(model, MidFusionMLP): # Mid fusion takes 4 inputs
                 logits = model(
                    batch['aerial'].to(device),
                    batch['sar'].to(device),
                    batch['acoustic'].to(device),
                    batch['ais'].to(device)
                )
            elif isinstance(model, EarlyFusionCNN): # Early fusion takes 1 concat input? No, typically assumes concatenated
                # Check signature of EarlyFusionCNN or update it to take dict/named args if strictly needed
                # Assuming baselines wrapped or adapted. 
                # Let's assume standard call for now, but update if needed.
                # Actually earlier code used: model(aerial, sar, acoustic, ais) or similar.
                # Let's match train_baselines.py usage.
                # train_baselines.py: logits = model(aerial, sar, acoustic, ais) for all.
                
                logits = model(
                    batch['aerial'].to(device),
                    batch['sar'].to(device),
                    batch['acoustic'].to(device),
                    batch['ais'].to(device)
                )
            else: # Decision Fusion
                 logits = model(
                    batch['aerial'].to(device),
                    batch['sar'].to(device),
                    batch['acoustic'].to(device),
                    batch['ais'].to(device)
                )

            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(batch['label'].numpy())
    
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    y_prob = torch.cat(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'ece': calculate_ece(y_prob, torch.tensor(y_true))
    }
    
    return metrics

def compare_all_baselines():
    """Generate comprehensive comparison of all models"""
    
    # Setup data
    SYNTHETIC_DATA_DIR = get_path(CFG, CFG['paths']['synthetic_data_dir'])
    METADATA_PATH = os.path.join(SYNTHETIC_DATA_DIR, CFG['paths']['metadata_path'])
    WORKING_DIR = get_path(CFG, CFG['paths']['working_dir'])
    
    df = pd.read_csv(METADATA_PATH)
    img_size = tuple(CFG['augmentation']['image_size'])
    
    val_tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15/0.8, random_state=42)
    
    val_loader = DataLoader(MaritimeDatasetV3(val_df, SYNTHETIC_DATA_DIR, val_tfm),
                             batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # Models to compare
    models_config = {
        'Early Fusion CNN': ('early_fusion_cnn', EarlyFusionCNN(num_classes=16)),
        'Mid Fusion MLP': ('mid_fusion_mlp', MidFusionMLP(num_classes=16, feature_dim=256)),
        'Decision Ensemble': ('decision_fusion', DecisionFusionEnsemble(num_classes=16)),
        'MAFT (No Dropout)': ('maft_no_dropout', MAFTNoDropout(CFG)),
        'MAFT v3.1 (Ours)': ('maft_v3', MAFTv3(CFG))
    }
    
    results = []
    
    for model_name, (weight_name, model) in models_config.items():
        print(f"\nEvaluating {model_name}...")
        
        # Load weights
        weight_path = os.path.join(WORKING_DIR, f'{weight_name}_best.pth')
        if not os.path.exists(weight_path):
             # Try fallback for main model
            if model_name == 'MAFT v3.1 (Ours)':
                 weight_path = os.path.join(WORKING_DIR, 'maft_advanced_v3.pth')
            
            if not os.path.exists(weight_path):
                print(f"  Warning: Weights not found for {model_name} at {weight_path}, skipping...")
                continue
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except Exception as e:
            print(f"Error loading weights for {model_name}: {e}")
            continue
            
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, device)
        
        # Profile
        # Note: profile_model from utils returns (params, latency)
        # We need memory too.
        # Let's stick to the local profile_model which returns 3 values? 
        # utils.profile_model only returns 2. I should probably extend utils.profile_model or calculate memory here.
        # I'll enable memory calculation here.
        torch.cuda.reset_peak_memory_stats(device)
        num_params, latency = profile_model(model, val_loader, device) 
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        results.append({
            'Model': model_name,
            'Fusion Type': {
                'Early Fusion CNN': 'Input-level',
                'Mid Fusion MLP': 'Feature-level (MLP)',
                'Decision Ensemble': 'Decision-level',
                'MAFT (No Dropout)': 'Feature-level (Transformer)',
                'MAFT v3.1 (Ours)': 'Feature-level (Transformer + Dropout)'
            }[model_name],
            'Accuracy': metrics['accuracy'] * 100,
            'F1-Score': metrics['f1'] * 100,
            'Precision': metrics['precision'] * 100,
            'Recall': metrics['recall'] * 100,
            'ECE': metrics['ece'],
            'Parameters (M)': num_params / 1e6,
            'Latency (ms)': latency,
            'Memory (MB)': memory_mb
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(os.path.join(WORKING_DIR, 'results'), exist_ok=True)
    df_results.to_csv(os.path.join(WORKING_DIR, 'results', 'baseline_comparison.csv'), index=False)
    
    # Print table
    print("\n" + "="*100)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("="*100)
    print(df_results.to_string(index=False))
    print("="*100)
    
    # Generate visualizations
    generate_comparison_plots(df_results, os.path.join(WORKING_DIR, 'results'))
    
    return df_results

def generate_comparison_plots(df, output_dir):
    """Generate comparison visualizations"""
    
    # Plot 1: Accuracy, F1, ECE comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    axes[0].barh(df['Model'], df['Accuracy'], color='steelblue')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].grid(axis='x', alpha=0.3)
    
    # F1-Score
    axes[1].barh(df['Model'], df['F1-Score'], color='forestgreen')
    axes[1].set_xlabel('F1-Score (%)')
    axes[1].set_title('Model F1-Score Comparison')
    axes[1].grid(axis='x', alpha=0.3)
    
    # ECE (lower is better)
    axes[2].barh(df['Model'], df['ECE'], color='coral')
    axes[2].set_xlabel('ECE (lower is better)')
    axes[2].set_title('Model Calibration (ECE)')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved performance comparison to {output_dir}/baseline_performance_comparison.png")
    
    # Efficiency Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].barh(df['Model'], df['Parameters (M)'], color='purple'); axes[0].set_title('Model Size (M Params)')
    axes[1].barh(df['Model'], df['Latency (ms)'], color='orange'); axes[1].set_title('Latency (ms)')
    axes[2].barh(df['Model'], df['Memory (MB)'], color='teal'); axes[2].set_title('Memory (MB)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_efficiency_comparison.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    compare_all_baselines()
