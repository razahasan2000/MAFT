import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- MODEL ARCHITECTURE ---

class ImageEncoderB2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Using EfficientNet-B2 as the backbone
        self.backbone = timm.create_model('tf_efficientnet_b2.ns_jft_in1k', pretrained=True, num_classes=0, global_pool='')
        feat_dim = 1408
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_dim, output_dim)
        self.orient_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map).flatten(1)
        projected = self.proj(pooled)
        orientation = self.orient_head(pooled)
        return projected, orientation

class AISEncoder(nn.Module):
    def __init__(self, input_dim=12, output_dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MAFTNoDropout(nn.Module):
    def __init__(self, cfg, num_classes=16):
        super().__init__()
        f_dim = cfg['model']['feature_dim']
        self.encoders = nn.ModuleDict({
            'aerial': ImageEncoderB2(f_dim),
            'sar': ImageEncoderB2(f_dim),
            'acoustic': ImageEncoderB2(f_dim),
            'ais': AISEncoder(12, f_dim)
        })
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=f_dim,
            nhead=cfg['model']['n_heads'],
            dim_feedforward=f_dim * 4,
            dropout=cfg['model']['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['model']['n_layers'])
        self.classifier = nn.Linear(f_dim, num_classes)
    
    def forward(self, aerial, sar, acoustic, ais):
        tokens = []
        
        # Process modalities
        a_feat, _ = self.encoders['aerial'](aerial)
        s_feat, _ = self.encoders['sar'](sar)
        ac_feat, _ = self.encoders['acoustic'](acoustic)
        ai_feat = self.encoders['ais'](ais)
        
        tokens = torch.stack([a_feat, s_feat, ac_feat, ai_feat], dim=1) # [B, 4, f_dim]
        
        # Transformer Fusion (No Modality Dropout)
        fused = self.transformer(tokens)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits

# --- DATASET ---

class MaritimeDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
        self.label_map = {l:i for i,l in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        def load_img(p):
            return self.transform(Image.open(os.path.join(self.root, p)).convert('RGB'))
        
        ais = torch.tensor([row[c] for c in ['sog','cog','signal_present','rate_of_turn',
                                               'trajectory_consistency','speed_variance',
                                               'time_since_update','vessel_type_encoded',
                                               'vessel_length_proxy','nav_status',
                                               'proximity_to_shore']] + [0.0], dtype=torch.float32)
        
        return {
            'aerial': load_img(row['aerial_path']),
            'sar': load_img(row['sar_path']),
            'acoustic': load_img(row['acoustic_path']),
            'ais': ais,
            'label': torch.tensor(self.label_map[row['label']], dtype=torch.long)
        }

# --- TRAINING LOOP ---

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    data_dir = os.path.join(cfg['paths']['base_dir'], cfg['paths']['synthetic_data_dir'])
    metadata_path = os.path.join(data_dir, cfg['paths']['metadata_path'])
    
    # Load Data
    df = pd.read_csv(metadata_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    img_size = tuple(cfg['augmentation']['image_size'])
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(MaritimeDataset(train_df, data_dir, transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(MaritimeDataset(val_df, data_dir, transform), batch_size=16, shuffle=False)

    # Initialize Model
    num_classes = len(df['label'].unique())
    model = MAFTNoDropout(cfg, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Simple Train loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            logits = model(
                batch['aerial'].to(device),
                batch['sar'].to(device),
                batch['acoustic'].to(device),
                batch['ais'].to(device)
            )
            
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Simple Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch['aerial'].to(device),
                    batch['sar'].to(device),
                    batch['acoustic'].to(device),
                    batch['ais'].to(device)
                )
                preds = logits.argmax(dim=1)
                correct += (preds == batch['label'].to(device)).sum().item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} Val Acc: {correct/len(val_df):.4f}")

    # Save final model
    os.makedirs(cfg['paths']['working_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg['paths']['working_dir'], "maft_no_dropout_final.pth"))

if __name__ == "__main__":
    main()