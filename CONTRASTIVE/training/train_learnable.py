import torch
from ..loss.info_nce import info_nce_loss, entropy_loss
from ..ContrastiveArchitecture.ContrastiveModels import ContrastiveModel_learnable
from ..augmentations.augmentations_fixed import augmentations_sequence_video, augmentations_sequence_video_trainable
from ..encoders.encoders import LSTMEncoder
from ..datasets.shrek_dataloader import trainloader_shrek
import numpy as np
from copy import deepcopy


## Entraînement avec augmentation NON-FIXE

def training_model_learnable(model, augmentations, n_epochs, trainloader, temperature, coeff):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        for batch, reste in trainloader:


            z1, h1 = model(batch)
            z2, h2 = model(batch)

            loss = info_nce_loss(z1, z2, temperature=temperature) - coeff*entropy_loss(model.augmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss)
    model.eval()
    return model


### Exemples d'utilisation


temperatures = np.linspace(start = 0.001, stop = 1, num = 100)
trainloader = trainloader_shrek(batch_size=700)

batch_size, n_frames, n_joints, coord = next(iter(trainloader))
models = []
augmentation_architecture = augmentations_sequence_video_trainable()

n_epochs = 2000
coeff = 0.01

for temperature in temperatures:
    model = ContrastiveModel_learnable(augmentation=augmentation_architecture ,encoder=LSTMEncoder(input_dim=coord*n_joints, hidden_dim=512))
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in n_epochs:
        for batch, reste in trainloader:
            vid1 = augmentations_sequence_video(batch)
            vid2 = augmentations_sequence_video(batch)
            
            z1, h1 = model(vid1)
            z2, h2 = model(vid2)

            loss = info_nce_loss(z1, z2, temperature=temperature) - coeff*entropy_loss(model.augmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
    models.append(deepcopy(model))



