import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

## ________________________________________________________________________________________________________________________ ## 

## Pour Images, encoder avec plusieurs couches CNN

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        self.encoder_network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # Convolution layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample the image
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.flatten = nn.Flatten()
        
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), 
            nn.ReLU(),
            nn.Linear(512, 128)  
        )

    def forward(self, x):
        x = self.encoder_network(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

## ________________________________________________________________________________________________________________________ ## 

## Pour images, modèle pré-entraîné ResNet où on enlève la dernière couche de classification

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  
        

    def forward(self, x):
        x = self.resnet(x)
        return x
    
## ________________________________________________________________________________________________________________________ ## 

### Pour vidéos, GRU

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=False):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction, hidden_dim)

    def forward(self, x):
        batch_size, n_frames, n_joints, coord = x.shape
        x = x.view(batch_size, n_frames, n_joints*3)
        
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_dim,
            device=x.device,
        )

        
        out, hn = self.gru(x, h0)  

        
        if self.bidirectional:
            
            final_hidden_state = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            final_hidden_state = hn[-1]

        
        embedding = self.fc(final_hidden_state)
        return embedding  


## ________________________________________________________________________________________________________________________ ##  

## Pour vidéos, LSTM


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Propagate input through the LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        # Return the output of the last time step
        return out


## ________________________________________________________________________________________________________________________ ## 

### Pour vidéos, transformers (pas testé)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 

        self.register_buffer("pe", pe)
    

class TransformersEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=5000):

        super(TransformersEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, src):
        
        src = self.input_proj(src) * (self.embed_dim ** 0.5)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) 

        output = self.transformer_encoder(src)
        output = output.permute(1, 2, 0)  
        output = self.pool(output).squeeze(-1)  

        return output 