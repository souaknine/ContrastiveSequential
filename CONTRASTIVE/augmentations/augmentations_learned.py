import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import v2


## Classes d'augmentation avec paramètres pouvant être mises à jour pendant la propagation (ici : amplitude de la couche + probabilité de la couche d'être appliquée)


## ________________________________________________________________________________________________________________________ ## 

## Rotation Séquence 


class RandomRotation3D_forward(torch.nn.Module):
    def __init__(self, degrees=30, axis='z'):

        self.axis = axis

    def forward(self, sequence, degrees):
        
        batch_size = sequence.size(0)
        epsilon = torch.rand(batch_size)
        
        angle = -degrees + 2*degrees*epsilon
        angle_rad = torch.deg2rad(torch.tensor(angle))
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)

        rotation_matrix = torch.zeros((batch_size, 3, 3))
        

        if self.axis == 'x':
            rotation_matrix[:, 0, 0] = 1
            rotation_matrix[: ,1, 1] = cos
            rotation_matrix[:, 1, 2] = -sin
            rotation_matrix[:, 2, 1] = sin
            rotation_matrix[:, 2, 2] = cos
        elif self.axis == 'y':
            rotation_matrix[:, 1, 1] = 1
            rotation_matrix[:,0, 0] = cos
            rotation_matrix[:,0, 2] = sin
            rotation_matrix[:,2, 0] = -sin
            rotation_matrix[:,2, 2] = cos
        elif self.axis == 'z':
            rotation_matrix[:, 2, 2] = 1
            rotation_matrix[:,0, 0] = cos
            rotation_matrix[:,0, 1] = -sin
            rotation_matrix[:,1, 0] = sin
            rotation_matrix[:,1, 1] = cos

        
        batch_size, n_frames, n_joints, _ = sequence.shape
        sequence = sequence.view(batch_size, -1, 3)
        sequence = torch.bmm(sequence, rotation_matrix.transpose(1, 2))
       
        sequence = sequence.view(batch_size, n_frames, n_joints, 3)
        return sequence

class Rotation3D_trainable(torch.nn.Module):
    def __init__(self, proba = 0.5, amplitude=30.0, axis='z'):
        super().__init__()
        self.proba = torch.nn.Parameter(torch.tensor([1 - proba,proba]))
        self.amplitude = torch.nn.Parameter(torch.tensor(amplitude))
        self.next = RandomRotation3D_forward(axis= axis)

    def forward(self, x):

        batch_size = x.size()[0]
        proba = self.proba.unsqueeze(0).expand(batch_size, -1)
        gumbel_sample = self.gumbel_softmax_sample(proba)[:, 1]



        degrees = self.amplitude*gumbel_sample

        
        return self.next(x, degrees=degrees)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature=0.5):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)
    


    
## ________________________________________________________________________________________________________________________ ## 

## Ajout bruit gaussien


class AddGaussianNoise_trainable(torch.nn.Module):
    def __init__(self, proba, amplitude):
          
        super().__init__()
        self.proba = torch.nn.Parameter(torch.tensor([1 - proba,proba]))
        self.amplitude = torch.nn.Parameter(torch.tensor(amplitude))

    def forward(self, x):

        batch_size = x.size()[0]
        proba = self.proba.unsqueeze(0).expand(batch_size, -1)
        gumbel_sample = self.gumbel_softmax_sample(proba)[:, 1]
        gumbel_sample = gumbel_sample.view(batch_size, 1, 1, 1)



        noises = torch.randn_like(x)*self.amplitude*gumbel_sample
        
        return x + noises

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature=0.5):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)
    

## ________________________________________________________________________________________________________________________ ## 

## Changement de Perspective

class RandomPerspectiveBatch_forward(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.ok = True


    def forward(self, sequences, max_angles):

        epsilon = torch.rand(batch_size, 3)

        angles = -max_angles + 2 * max_angles * epsilon

        angles_rad = torch.deg2rad(angles)
        cos_angles = torch.cos(angles_rad)
        sin_angles = torch.sin(angles_rad)

        ones = torch.ones(batch_size, 1, device=sequences.device)
        zeros = torch.zeros(batch_size, 1, device=sequences.device)

        Rx = torch.stack([
        torch.stack([ones.squeeze(), zeros.squeeze(), zeros.squeeze()], dim=1),
        torch.stack([zeros.squeeze(), cos_angles[:, 0], -sin_angles[:, 0]], dim=1),
        torch.stack([zeros.squeeze(), sin_angles[:, 0], cos_angles[:, 0]], dim=1)
    ], dim=2)
        
        Ry = torch.stack([
        torch.stack([cos_angles[:, 1], zeros.squeeze(), sin_angles[:, 1]], dim=1),
        torch.stack([zeros.squeeze(), ones.squeeze(), zeros.squeeze()], dim=1),
        torch.stack([-sin_angles[:, 1], zeros.squeeze(), cos_angles[:, 1]], dim=1)
    ], dim=2)
        
        Rz = torch.stack([
        torch.stack([cos_angles[:, 2], -sin_angles[:, 2], zeros.squeeze()], dim=1),
        torch.stack([sin_angles[:, 2], cos_angles[:, 2], zeros.squeeze()], dim=1),
        torch.stack([zeros.squeeze(), zeros.squeeze(), ones.squeeze()], dim=1)
    ], dim=2)
        

        R = torch.bmm(torch.bmm(Rz, Ry), Rx)
        
        batch_size, n_frames, n_joints, _ = sequences.shape
        sequences = sequences.view(-1, 3) 
        sequences_rotated = torch.bmm(sequences, R.transpose(1, 2))
        sequences_rotated = sequences_rotated.view(batch_size, n_frames, n_joints, 3)
        
        return sequences_rotated


class Perspective_trainable(torch.nn.Module):
    def __init__(self, proba = 0.5, amplitude=30.0, axis='z'):
        super().__init__()
        self.proba = torch.nn.Parameter(torch.tensor([1 - proba,proba]))
        self.amplitude = torch.nn.Parameter(torch.tensor(amplitude))
        self.next = RandomPerspectiveBatch_forward()

    def forward(self, x):

        batch_size = x.size()[0]
        proba = self.proba.unsqueeze(0).expand(batch_size, -1)
        gumbel_sample = self.gumbel_softmax_sample(proba)[:, 1]



        max_angles = self.amplitude*gumbel_sample

        
        return self.next(x, max_angle=max_angles)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature=0.5):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)
    
## ________________________________________________________________________________________________________________________ ## 

## On rajoute les autres augmentations qu'on a laissé fixe (accélération, décélération, cacher une partie de la vidéo) : moins facile de déterminer ici l'amplitude ici mais on peut le faire

class VideoRalentie(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ok = True

    def forward(self, sequences):

        
        batch_size, n_frames, n_joints, n_coords = sequences.shape
        
        sequences = sequences.permute(0, 2, 3, 1)  
        new_n_frames = n_frames * self.factor
        
        sequences = torch.nn.functional.interpolate(
            sequences,
            size=new_n_frames,
            mode='linear',
            align_corners=False
        )
        
        sequences = sequences.permute(0, 3, 1, 2) 
        
        indices = torch.linspace(0, new_n_frames - 1, steps=n_frames).long()
        sequences = sequences[:, indices, :, :]
        return sequences
    

class VideoAcceleree(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ok = True


    def forward(self, sequences):

        batch_size, n_frames, n_joints, n_coords = sequences.shape
        
        indices = torch.arange(0, n_frames, 2)
        sequences = sequences[:, indices, :, :]
        
        n_missing = n_frames - sequences.shape[1]
        if n_missing > 0:
            last_frames = sequences[:, -1:, :, :].repeat(1, n_missing, 1, 1)
            sequences = torch.cat([sequences, last_frames], dim=1)

        return sequences
    
class HorizontalFlipBatch(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.ok = True

    def forward(self, sequences):


        sequences = sequences.clone()
        sequences[..., 0] *= -1  

        return sequences
    
class HideSequenceSegment(torch.nn.Module):
    def __init__(self, min_ratio=0.1, max_ratio=0.3):
        super().__init__()

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio


    def forward(self, sequence):
        
        n_frames = 90
        batch_size = sequence.shape
        for i in range(batch_size):
            hide_len = int(n_frames * random.uniform(self.min_ratio, self.max_ratio))
            start = random.randint(0, n_frames - hide_len)
            sequence[i, start:start + hide_len] = 0
        
        return sequence


## ________________________________________________________________________________________________________________________ ## 

## Composée de toutes les transformations, à changer si on veut appliquer en fonction d'un batch, changer l'amplitude de certaines, la probabilité, etc...


def augmentations_sequence_video_trainable():
    augmentations = v2.Compose([
        HideSequenceSegment(),
        HorizontalFlipBatch(),
        VideoAcceleree(),
        Perspective_trainable(),
        Rotation3D_trainable(),
        AddGaussianNoise_trainable()
    ])
    return augmentations
