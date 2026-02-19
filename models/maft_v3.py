import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random

# ------------------------------------------------------------------------------
# Models (EfficientNet-B2 + Refined Cross-Attn)
# ------------------------------------------------------------------------------
class ImageEncoderB2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b2.ns_jft_in1k', pretrained=True, num_classes=0, global_pool='')
        feat_dim = 1408
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat_dim, output_dim)
        self.proj = nn.LayerNorm(output_dim)
        # Orientation head for auxiliary loss
        self.orient_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).flatten(1)
        return self.proj(self.fc(x)), self.orient_head(x)

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
    def forward(self, x): return self.net(x)

class MAFTv3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f_dim = cfg['model']['feature_dim']
        self.encoders = nn.ModuleDict({
            'aerial': ImageEncoderB2(f_dim),
            'sar': ImageEncoderB2(f_dim),
            'acoustic': ImageEncoderB2(f_dim), # Using B2 for acoustic spectrogram too
            'ais': AISEncoder(12, f_dim)
        })
        self.cls_token = nn.Parameter(torch.zeros(1, 1, f_dim))
        self.modality_names = ['cls', 'aerial', 'sar', 'acoustic', 'ais']
        self.mod_to_id = {n: i for i, n in enumerate(self.modality_names)}
        self.mod_embed = nn.Embedding(len(self.modality_names), f_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=f_dim, 
            nhead=cfg['model']['n_heads'], 
            dim_feedforward=f_dim*4, 
            dropout=cfg['model']['dropout'], 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['model']['n_layers'])
        self.classifier = nn.Sequential(
            nn.LayerNorm(f_dim),
            nn.Linear(f_dim, 16) # 4 anomalies * 4 ship types
        )
        self.modality_dropout_rate = cfg['augmentation']['modality_dropout_rate']

    def forward(self, aerial=None, sar=None, acoustic=None, ais=None, return_aux=False, force_no_dropout=False):
        sys_aux, embs, tokens, mod_names = {}, {}, [], []
        
        inputs = {'aerial': aerial, 'sar': sar, 'acoustic': acoustic, 'ais': ais}
        
        # Modality Dropout logic
        active_modalities = list(inputs.keys())
        if self.training and not force_no_dropout:
            # Ensure at least one modality remains
            keep_prob = 1.0 - self.modality_dropout_rate
            mods_to_keep = [m for m in active_modalities if random.random() < keep_prob]
            if not mods_to_keep: mods_to_keep = [random.choice(active_modalities)]
            active_modalities = mods_to_keep

        for m_name in active_modalities:
            if m_name == 'ais':
                f = self.encoders[m_name](inputs[m_name])
            else:
                f, o = self.encoders[m_name](inputs[m_name])
                if m_name == 'aerial': sys_aux['aerial_orientation'] = o
            
            tokens.append(f)
            mod_names.append(m_name)
            embs[m_name] = f
            
        x = torch.stack(tokens, dim=1)
        # Add CLS token
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        # Add modality embeddings
        mod_ids = torch.tensor([self.mod_to_id['cls']] + [self.mod_to_id[n] for n in mod_names], device=x.device)
        x = x + self.mod_embed(mod_ids)
        
        feat = self.transformer(x)
        logits = self.classifier(feat[:, 0])
        
        return (logits, sys_aux, embs) if return_aux else logits
