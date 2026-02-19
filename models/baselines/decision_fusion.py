"""
Decision-Level Fusion (Ensemble Voting) Baseline
Trains 4 independent single-modality classifiers and combines via majority voting.
"""
import torch
import torch.nn as nn
import timm

class SingleModalityClassifier(nn.Module):
    """Single modality classifier (for aerial, sar, or acoustic)"""
    def __init__(self, num_classes=16, input_type='image'):
        super().__init__()
        self.input_type = input_type
        
        if input_type == 'image':
            self.backbone = timm.create_model(
                'tf_efficientnet_b0.ns_jft_in1k',
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            feat_dim = self.backbone.num_features
            self.classifier = nn.Linear(feat_dim, num_classes)
        else:  # AIS
            self.backbone = nn.Sequential(
                nn.Linear(12, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class DecisionFusionEnsemble(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        
        # 4 independent classifiers
        self.aerial_clf = SingleModalityClassifier(num_classes, 'image')
        self.sar_clf = SingleModalityClassifier(num_classes, 'image')
        self.acoustic_clf = SingleModalityClassifier(num_classes, 'image')
        self.ais_clf = SingleModalityClassifier(num_classes, 'ais')
        
        self.num_classes = num_classes
        
    def forward(self, aerial, sar, acoustic, ais, **kwargs):
        # Get predictions from each classifier
        aerial_logits = self.aerial_clf(aerial)
        sar_logits = self.sar_clf(sar)
        acoustic_logits = self.acoustic_clf(acoustic)
        ais_logits = self.ais_clf(ais)
        
        # Average logits (soft voting)
        # This is more stable than hard voting during training
        avg_logits = (aerial_logits + sar_logits + acoustic_logits + ais_logits) / 4.0
        
        return avg_logits
    
    def predict_hard_voting(self, aerial, sar, acoustic, ais):
        """Hard voting for inference (optional)"""
        with torch.no_grad():
            aerial_pred = self.aerial_clf(aerial).argmax(1)
            sar_pred = self.sar_clf(sar).argmax(1)
            acoustic_pred = self.acoustic_clf(acoustic).argmax(1)
            ais_pred = self.ais_clf(ais).argmax(1)
            
            # Stack predictions and take mode
            votes = torch.stack([aerial_pred, sar_pred, acoustic_pred, ais_pred], dim=1)
            # Simple majority voting (take most common prediction)
            final_pred = torch.mode(votes, dim=1)[0]
            
            return final_pred
