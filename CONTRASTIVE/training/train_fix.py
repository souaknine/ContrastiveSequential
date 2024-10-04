
import torch
from ..loss.info_nce import info_nce_loss
from ..ContrastiveArchitecture.ContrastiveModels import ContrastiveModel_fixed
from ..augmentations.augmentations_fixed import augmentations_sequence_video
from ..encoders.encoders import LSTMEncoder
from ..datasets.shrek_dataloader import trainloader_shrek
import numpy as np
from copy import deepcopy


## Entraînement avec augmentation FIXE

def training_model_fixed(model, augmentations, n_epochs, trainloader, temperature):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        for batch, reste in trainloader:

        
            vid1 = augmentations(batch)
            vid2 = augmentations(batch)

            z1, h1 = model(vid1)
            z2, h2 = model(vid2)

            loss = info_nce_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss)
    model.eval()
    return model


### Exemples d'utilisation


temperatures = np.linspace(start = 0.01, stop = 1, num = 100)
trainloader = trainloader_shrek(batch_size=12)



batch_size, n_frames, n_joints, coord = next(iter(trainloader))["Sequence"].shape
models = []

n_epochs = 2000

augmentations = augmentations_sequence_video()



for temperature in temperatures:
    encoder = LSTMEncoder(input_size=coord*n_joints, hidden_dim=128, num_layers=1, output_dim=64)
    model = ContrastiveModel_fixed(encoder=encoder)
    
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        for dico in trainloader:
            batch = dico["Sequence"]
            vid1 = augmentations(batch)
            vid2 = augmentations(batch)
            
            
            vid1 = vid1.to(torch.float32)
            
            z1, h1 = model(vid1)


            
            vid2 = vid2.to(torch.float32)
            z2, h2 = model(vid2)



            loss = info_nce_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
    models.append(deepcopy(model))











    





    