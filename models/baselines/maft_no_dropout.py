"""
MAFT (No Dropout) Ablation
Same architecture as MAFT v3.1 but with modality_dropout_rate = 0.0
This proves the value of modality dropout for robustness.
"""
import torch
import torch.nn as nn
import timm

class ImageEncoderB2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
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
    """MAFT architecture WITHOUT modality dropout (ablation study)"""
    def __init__(self, cfg):
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
        self.classifier = nn.Linear(f_dim, 16)
        
        # NO modality dropout - this is the key difference
        self.modality_dropout_rate = 0.0
    
    def forward(self, aerial=None, sar=None, acoustic=None, ais=None, return_aux=False, force_no_dropout=False):
        tokens, aux_outputs = [], {}
        
        # Process aerial
        if aerial is not None:
            aerial_feat, aerial_orient = self.encoders['aerial'](aerial)
            tokens.append(aerial_feat.unsqueeze(1))
            aux_outputs['aerial_orientation'] = aerial_orient
        
        # Process SAR
        if sar is not None:
            sar_feat, _ = self.encoders['sar'](sar)
            tokens.append(sar_feat.unsqueeze(1))
        
        # Process acoustic
        if acoustic is not None:
            acoustic_feat, _ = self.encoders['acoustic'](acoustic)
            tokens.append(acoustic_feat.unsqueeze(1))
        
        # Process AIS
        if ais is not None:
            ais_feat = self.encoders['ais'](ais)
            tokens.append(ais_feat.unsqueeze(1))
        
        # Concatenate and fuse (NO DROPOUT)
        x = torch.cat(tokens, dim=1)
        fused = self.transformer(x)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        
        if return_aux:
            return logits, aux_outputs, fused
        return logits
