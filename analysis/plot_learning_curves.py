import re
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

def parse_log(log_path):
    epochs = []
    losses = []
    accs = []
    f1s = []
    eces = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    # Find the last "Starting MAFT v3 Training Pipeline" to ensure we get the latest run
    start_idx = 0
    for i, line in enumerate(lines):
        if "Starting MAFT v3 Training Pipeline" in line:
            start_idx = i
            
    print(f"Parsing log from line {start_idx}...")
    
    # Regex for parsing
    # Epoch 1 - Loss: 1.9527 - Acc: 0.1933 - F1: 0.1126 - ECE: 0.0777
    pattern = r"Epoch (\d+) - Loss: ([\d.]+) - Acc: ([\d.]+) - F1: ([\d.]+) - ECE: ([\d.]+)"
    
    for line in lines[start_idx:]:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            acc = float(match.group(3))
            f1 = float(match.group(4))
            ece = float(match.group(5))
            
            # Handle potential duplicates if log has restarts (not expected here based on last check)
            if epochs and epoch <= epochs[-1]:
                # Reset if epoch count restarts
                epochs = []
                losses = []
                accs = []
                f1s = []
                eces = []
                
            epochs.append(epoch)
            losses.append(loss)
            accs.append(acc)
            f1s.append(f1)
            eces.append(ece)
            
    return epochs, losses, accs, f1s, eces

def plot_curves(epochs, losses, accs, f1s, eces, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1.plot(epochs, losses, 'r-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss vs Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, accs, 'b-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Validation Accuracy vs Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_learning_curves.py <log_file> <output_image>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    output_file = sys.argv[2]
    
    epochs, losses, accs, f1s, eces = parse_log(log_file)
    print(f"Parsed {len(epochs)} epochs.")
    
    plot_curves(epochs, losses, accs, f1s, eces, output_file)
