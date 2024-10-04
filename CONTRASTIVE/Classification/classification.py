
import torch
from ..loss.info_nce import info_nce_loss, entropy_loss
from ..ContrastiveArchitecture.ContrastiveModels import ContrastiveModel_learnable
from ..augmentations.augmentations_fixed import augmentations_sequence_video, augmentations_sequence_video_trainable
from ..encoders.encoders import LSTMEncoder
from ..datasets.shrek_dataloader import trainloader_shrek
import numpy as np
from copy import deepcopy
import torch.nn as nn


## Modèle simple de classification pour les tâches en aval supervisées
class ClassificationModel(nn.Module):
    def __init__(self, contrastive_arch, num_classes=16):
        super(ClassificationModel, self).__init__()
        self.contrastive = contrastive_arch
        self.classifier = nn.Linear(512, num_classes)  

    def forward(self, x):
        with torch.no_grad():
            z, h = self.contrastive(x)  
        logits = self.classifier(h)
        return logits