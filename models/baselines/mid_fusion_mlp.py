"""
Mid Fusion MLP Baseline
Uses separate encoders for each modality, then fuses with a simple MLP (no transformer).
"""
import torch
import torch.nn as nn
import timm

class MidFusionMLP(nn.Module):
    def __init__(self, num_classes=16, feature_dim=256):
        super().__init__()
        
        # Separate encoders for each image modality
        self.aerial_encoder = self._make_image_encoder(feature_dim)
        self.sar_encoder = self._make_image_encoder(feature_dim)
        self.acoustic_encoder = self._make_image_encoder(feature_dim)
        
        # AIS encoder
        self.ais_encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        # MLP fusion (no transformer, just concatenation + MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),  # 4 modalities
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_image_encoder(self, output_dim):
        backbone = timm.create_model(
            'tf_efficientnet_b0.ns_jft_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = backbone.num_features  # 1280
        
        return nn.Sequential(
            backbone,
            nn.Linear(feat_dim, output_dim)
        )
    
    def forward(self, aerial, sar, acoustic, ais, **kwargs):
        # Encode each modality separately
        aerial_feat = self.aerial_encoder(aerial)      # [B, 256]
        sar_feat = self.sar_encoder(sar)              # [B, 256]
        acoustic_feat = self.acoustic_encoder(acoustic)  # [B, 256]
        ais_feat = self.ais_encoder(ais)              # [B, 256]
        
        # Concatenate all features
        fused = torch.cat([aerial_feat, sar_feat, acoustic_feat, ais_feat], dim=1)  # [B, 1024]
        
        # MLP classifier
        logits = self.fusion_mlp(fused)
        
        return logits
