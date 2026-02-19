import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import random

# Add parent directory to path to allow importing models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.maft_v3 import MAFTv3
from training.utils import load_config, get_path, log_print, mixup_data, mixup_criterion, calculate_ece, profile_model, EMA
from training.dataset import MaritimeDatasetV3

warnings.filterwarnings("ignore")

def main():
    CFG = load_config()
    
    # Path setup
    WORKING_DIR = get_path(CFG, CFG['paths']['working_dir'])
    LOG_FILE = os.path.join(WORKING_DIR, CFG['paths']['log_file'])
    SYNTHETIC_DATA_DIR = get_path(CFG, CFG['paths']['synthetic_data_dir'])
    METADATA_PATH = os.path.join(SYNTHETIC_DATA_DIR, CFG['paths']['metadata_path'])
    WEIGHTS_PATH = os.path.join(WORKING_DIR, CFG['paths']['weights_path'])
    
    os.makedirs(WORKING_DIR, exist_ok=True)
    log_print("Starting MAFT v3 Training Pipeline...", LOG_FILE)
    
    # Set seeds
    random.seed(CFG['training']['random_state'])
    np.random.seed(CFG['training']['random_state'])
    torch.manual_seed(CFG['training']['random_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(METADATA_PATH)
    
    img_size = tuple(CFG['augmentation']['image_size'])
    train_tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(CFG['augmentation']['rotation_range']),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    train_df, test_df = train_test_split(df, test_size=CFG['dataset']['test_set_size'], random_state=CFG['training']['random_state'])
    train_df, val_df = train_test_split(train_df, test_size=CFG['dataset']['validation_set_size']/(1-CFG['dataset']['test_set_size']), random_state=CFG['training']['random_state'])
    
    train_loader = DataLoader(
        MaritimeDatasetV3(train_df, SYNTHETIC_DATA_DIR, train_tfm, True), 
        batch_size=CFG['training']['batch_size'], 
        shuffle=True, 
        num_workers=CFG['training']['num_workers'], 
        pin_memory=CFG['training']['pin_memory']
    )
    val_loader = DataLoader(
        MaritimeDatasetV3(val_df, SYNTHETIC_DATA_DIR, val_tfm), 
        batch_size=CFG['training']['batch_size'], 
        shuffle=False, 
        num_workers=CFG['training']['num_workers'], 
        pin_memory=CFG['training']['pin_memory']
    )
    
    model = MAFTv3(CFG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(CFG['training']['learning_rate']), weight_decay=float(CFG['training']['weight_decay']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['training']['num_epochs'])
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, CFG['model']['ema_decay'])
    
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(CFG['training']['num_epochs']):
        model.train()
        total_loss = 0
        total_train_acc = 0
        pb = tqdm(train_loader, desc=f"Epoch {ep+1}/{CFG['training']['num_epochs']}")
        
        for batch in pb:
            optimizer.zero_grad()
            for k in ['aerial','sar','acoustic','ais']: batch[k] = batch[k].to(device)
            target = batch['label'].to(device)
            
            # MixUp
            mixed_inputs, target_a, target_b, lam = mixup_data(batch, target, device, CFG['augmentation']['mixup_alpha'])
            
            with torch.cuda.amp.autocast(enabled=CFG['training']['use_mixed_precision']):
                logits, aux, _ = model(
                    aerial=mixed_inputs['aerial'], 
                    sar=mixed_inputs['sar'], 
                    acoustic=mixed_inputs['acoustic'], 
                    ais=mixed_inputs['ais'], 
                    return_aux=True
                )
                
                loss_cls = mixup_criterion(criterion, logits, target_a, target_b, lam)
                
                if 'aerial_orientation' in aux:
                    pred_orient = aux['aerial_orientation'].view(-1, 1)
                    target_orient = batch['cog_norm'].to(device).view(-1, 1)
                    loss_orient = F.mse_loss(pred_orient, target_orient)
                else:
                    loss_orient = 0
                    
                loss = loss_cls + CFG['model']['orientation_loss_weight'] * loss_orient
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            total_loss += loss.item()
            
            # Track Train Acc
            train_preds = logits.argmax(1)
            train_acc = (train_preds == target).float().mean().item()
            total_train_acc += train_acc
            
            pb.set_postfix(loss=loss.item(), acc=train_acc)
            
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        
        ema.apply_shadow()
        model.eval()
        preds, targets, all_probs = [], [], []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                target = batch['label'].to(device)
                logits = model(
                    aerial=batch['aerial'].to(device),
                    sar=batch['sar'].to(device),
                    acoustic=batch['acoustic'].to(device),
                    ais=batch['ais'].to(device),
                    force_no_dropout=True
                )
                
                # Val Loss
                val_loss = criterion(logits, target)
                total_val_loss += val_loss.item()
                
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(batch['label'].numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        y_true = np.concatenate(targets)
        y_pred = np.concatenate(preds)
        y_prob = torch.cat(all_probs)
        
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')
        val_ece = calculate_ece(y_prob, torch.tensor(y_true))
        
        ema.restore()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            log_print(f"--- New Best Model Saved (Val Acc: {best_acc:.4f}) ---", LOG_FILE)
            
            log_print(f"--- New Best Model Saved (Val Acc: {best_acc:.4f}) ---", LOG_FILE)
            
        log_print(f"Epoch {ep+1} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - F1: {val_f1:.4f} - ECE: {val_ece:.4f}", LOG_FILE)
        scheduler.step()

    # Final Profiling
    num_params, latency = profile_model(model, val_loader, device)
    log_print(f"Final Model Profile: {num_params/1e6:.2f}M params, {latency:.2f}ms inference latency", LOG_FILE)

if __name__ == "__main__":
    main()
