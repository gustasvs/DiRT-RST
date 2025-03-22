import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize

from functions.settings import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet = resnet50()
        if GRAYSCALE:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, TARGET_CLASS_COUNT),
            # nn.Softmax(1)
        )
        self.model = resnet

    def forward(self, x):
        return self.model(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, num_frames=TEMPORAL_FRAME_WINDOW, num_classes=TARGET_CLASS_COUNT, d_model=2048, nhead=8, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.resnet = resnet50()
        # Remove final classification layer to extract features
        self.resnet.fc = nn.Identity()
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnet(x)  # shape: (batch_size * num_frames, d_model)
        features = features.view(batch_size, num_frames, -1)  # shape: (batch_size, num_frames, d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        # Transformer expects (seq_len, batch_size, d_model)
        features = features.transpose(0, 1)
        transformed = self.transformer_encoder(features)
        # Aggregate features (e.g., mean pooling over time)
        aggregated = transformed.mean(dim=0)
        out = self.fc(aggregated)
        return out
