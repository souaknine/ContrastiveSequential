import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import os.path as opt
from tqdm import tqdm
import numpy as np



## Module totale pour le modèle d'apprentissage Contrastive (sans mise à jour des paramètres d'augmentation)
class ContrastiveModel_fixed(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveModel_fixed, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def forward(self, x):
        batch_size, n_frames, n_joints, coord = x.shape

        x = x.view(batch_size, n_frames, n_joints*coord).to(torch.float32)
        h = self.encoder(x) 
        z = self.projection_head(h)
        #z = F.normalize(z, dim=-1)
        return z, h
    


## Module totale avec mise à jour des paramètres d'augmentation 
class ContrastiveModel_learnable(nn.Module):
    def __init__(self, augmentation, encoder):
        super(ContrastiveModel_learnable, self).__init__()
        self.augmentation = augmentation
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        x_aug = self.augmentation(x)
        h = self.encoder(x_aug) 
        z = self.projection_head(h)
        #z = F.normalize(z, dim=-1)
        return z, h