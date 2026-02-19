import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MaritimeDatasetV3(Dataset):
    def __init__(self, df, root, transform, is_training=False):
        self.df, self.root, self.transform, self.is_training = df.reset_index(drop=True), root, transform, is_training
        self.classes = sorted(self.df['label'].unique())
        self.label_map = {l:i for i,l in enumerate(self.classes)}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        def load_img(p): return self.transform(Image.open(os.path.join(self.root, p)).convert('RGB'))
        ais = torch.tensor([row[c] for c in ['sog','cog','signal_present','rate_of_turn','trajectory_consistency','speed_variance','time_since_update','vessel_type_encoded','vessel_length_proxy','nav_status','proximity_to_shore']] + [0.0], dtype=torch.float32)
        if self.is_training: ais += torch.randn_like(ais) * 0.02
        return {
            'aerial': load_img(row['aerial_path']), 
            'sar': load_img(row['sar_path']), 
            'acoustic': load_img(row['acoustic_path']), 
            'ais': ais, 
            'label': torch.tensor(self.label_map[row['label']], dtype=torch.long), 
            'cog_norm': torch.tensor(row['cog']/360.0, dtype=torch.float32)
        }
