"""
Early Fusion CNN Baseline
Concatenates all modalities at input level and processes with a single CNN backbone.
"""
import torch
import torch.nn as nn
import timm

class EarlyFusionCNN(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        # Strategy: Use a standard 3-channel CNN but process concatenated features early
        
        # Shared CNN backbone for all images (process sequentially then concatenate)
        self.backbone = timm.create_model(
            'tf_efficientnet_b0.ns_jft_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # AIS encoder
        self.ais_encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128)
        )
        
        # Get feature dimensions
        img_feat_dim = self.backbone.num_features  # 1280 for efficientnet-b0
        
        # Early fusion: concatenate all image features + AIS
        # 3 images × 1280 features + 128 AIS features = 3968 total
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim * 3 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, aerial, sar, acoustic, ais, **kwargs):
        # Process each image through the same backbone
        aerial_feat = self.backbone(aerial)      # [B, 1280]
        sar_feat = self.backbone(sar)           # [B, 1280]
        acoustic_feat = self.backbone(acoustic)  # [B, 1280]
        
        # Process AIS
        ais_feat = self.ais_encoder(ais)  # [B, 128]
        
        # Early fusion: concatenate all features
        fused = torch.cat([aerial_feat, sar_feat, acoustic_feat, ais_feat], dim=1)  # [B, 3968]
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
