import torch
from torchvision.transforms import v2
import random
#import tsaug

## Module pour pouvoir appliquer différentes couches avec des probabilités différentes pour chaque élément du batch

class RandomApplyTransform(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()

        self.transform = transform
        self.p = p

    def forward(self, batch):

        return torch.stack([self.transform(img.unsqueeze(0))[0] if random.random() < self.p else img for img in batch])
    

## ________________________________________________________________________________________________________________________ ## 


## Augmentations d'images avec stratégie d'augmentation fixée

def augmentations_images():

    ## On peut ajouter d'autres modules de la bibliothèque v2

    augmentations = v2.Compose([
        RandomApplyTransform(v2.RandomCrop(size=(32,32))),
        RandomApplyTransform(v2.RandomPerspective(distortion_scale=0.5)),
        RandomApplyTransform(v2.RandomRotation(degrees=(-90, 90))),
        RandomApplyTransform(v2.RandomAffine(degrees=90)),
        RandomApplyTransform(v2.ColorJitter(brightness=3, hue=0.2)),
        RandomApplyTransform(v2.GaussianBlur(kernel_size=1)),
    ])

    return augmentations


## ________________________________________________________________________________________________________________________ ## 

## Stratégies d'augmentation pour les landmark data


## Ajouter bruit à la séquence

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std


    def __call__(self, sequence):

        noise = torch.randn_like(sequence) * self.std

        return sequence + noise
    
## On cache une partie de la séquence

class HideSequenceSegment(torch.nn.Module):
    def __init__(self, min_ratio=0.1, max_ratio=0.3):
        super().__init__()

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio


    def forward(self, sequence):
        
        n_frames = 90
        batch_size = sequence.shape[0]
        for i in range(batch_size):
            hide_len = int(n_frames * random.uniform(self.min_ratio, self.max_ratio))
            start = random.randint(0, n_frames - hide_len)
            sequence[i, start:start + hide_len] = 0
        
        return sequence
    


## On fait une rotation selon un axe

## A améliorer : la rotation est soit la même pour tous les éléments du batch je crois si on fait sans la boucle ? dépend de comment on utise par batch pour rapidité ou en faisant une boucle (pour toutes les augmentations sur les repères en fait)
class RandomRotation3D(torch.nn.Module):
    def __init__(self, degrees=30, axis='z'):
        super().__init__()

        self.degrees = degrees
        self.axis = axis

    def forward(self, sequence):
        dtype = sequence.dtype


        angle = random.uniform(-self.degrees, self.degrees)
        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=dtype))
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)

        rotation_matrix = torch.eye(3, dtype=dtype)
        

        if self.axis == 'x':
            rotation_matrix[1, 1] = cos
            rotation_matrix[1, 2] = -sin
            rotation_matrix[2, 1] = sin
            rotation_matrix[2, 2] = cos
        elif self.axis == 'y':
            rotation_matrix[0, 0] = cos
            rotation_matrix[0, 2] = sin
            rotation_matrix[2, 0] = -sin
            rotation_matrix[2, 2] = cos
        elif self.axis == 'z':
            rotation_matrix[0, 0] = cos
            rotation_matrix[0, 1] = -sin
            rotation_matrix[1, 0] = sin
            rotation_matrix[1, 1] = cos

        
        batch_size, n_frames, n_joints, _ = sequence.shape
        sequence = sequence.reshape(-1, 3) @ rotation_matrix.T
        sequence = sequence.view(batch_size, n_frames, n_joints, 3)
        return sequence
    

## Effet miroir sur les vidéos

class HorizontalFlipBatch(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.ok = True

    def forward(self, sequences):


        sequences = sequences.clone()
        sequences[..., 0] *= -1  

        return sequences
    

## Changement de perspective sur les vidéos

class RandomPerspectiveBatch(torch.nn.Module):
    
    def __init__(self, max_angle=30):
        super().__init__()
        self.ok = True
        self.max_angle= max_angle

    def forward(self, sequences):
        dtype = sequences.dtype

        angles = torch.tensor([
            random.uniform(-self.max_angle, self.max_angle) for _ in range(3)
        ])
        angles_rad = torch.deg2rad(angles)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles_rad[0]), -torch.sin(angles_rad[0])],
            [0, torch.sin(angles_rad[0]), torch.cos(angles_rad[0])]
        ], dtype=dtype)
        Ry = torch.tensor([
            [torch.cos(angles_rad[1]), 0, torch.sin(angles_rad[1])],
            [0, 1, 0],
            [-torch.sin(angles_rad[1]), 0, torch.cos(angles_rad[1])]
        ], dtype=dtype)
        Rz = torch.tensor([
            [torch.cos(angles_rad[2]), -torch.sin(angles_rad[2]), 0],
            [torch.sin(angles_rad[2]), torch.cos(angles_rad[2]), 0],
            [0, 0, 1]
        ], dtype=dtype)
        

        R = Rz @ Ry @ Rx
        
        batch_size, n_frames, n_joints, _ = sequences.shape
        sequences = sequences.view(-1, 3) @ R.T
        sequences = sequences.view(batch_size, n_frames, n_joints, 3)
        
        return sequences




## On accélère la séquence en prenant une frame sur deux et on repète la dernière frame sur les frames manquantes à la fin
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
    

## On ralentit la vidéo en interpolant entre deux frames à chaque fois pour doubler la longueur de la vidéo, puis on prend ensuite 90 frames parmi la vidéo "interpolée" pour retrouver le nombre original. En fonction du padding initial, ça peut faire l'effet inverse mais on suppose que c'est juste une autre augmentation réaliste. Problème sur l'interpolation d'un batch ?

# class VideoRalentie(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ok = True

#     def forward(self, sequences):

        
#         batch_size, n_frames, n_joints, n_coords = sequences.shape
        
        
#         new_n_frames = n_frames * 2
        
#         for i in range(batch_size):
#             sequence = sequences[i]
#             sequence = sequence.permute(1, 2, 0)
#             sequence = sequence.reshape(-1, n_frames)
#         sequences = torch.nn.functional.interpolate(
#             sequences,
#             size=new_n_frames,
#             mode='linear',
#             align_corners=False
#         )
        
#         sequences = sequences.permute(0, 3, 1, 2) 
        
#         indices = torch.linspace(0, new_n_frames - 1, steps=n_frames).long()
#         sequences = sequences[:, indices, :, :]
#         return sequences
    


## Si on fait accélération + décélération, ça peut créer un effet "nul" ? Pas testé
def augmentations_sequence_video():
    augmentations = v2.Compose([
        RandomApplyTransform(HideSequenceSegment(min_ratio=0.1, max_ratio=0.3), p=0.5),
        RandomApplyTransform(RandomRotation3D(degrees=30, axis='z'), p=0.5),
        RandomApplyTransform(RandomPerspectiveBatch(), p = 0.5),
        RandomApplyTransform(HorizontalFlipBatch(torch.nn.Module), p=0.5),
        RandomApplyTransform(VideoAcceleree(), p=0.5)
    ])
    return augmentations
