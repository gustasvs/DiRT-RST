import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from torchvision.models import resnet50, ResNet50_Weights, resnet18, efficientnet_b0, efficientnet_b1, efficientnet_b3, efficientnet_v2_s, efficientnet_b2
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.video.s3d import s3d
from torchvision.models.video.resnet import r2plus1d_18

from functions.settings import *


class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()

        efficientnet = efficientnet_b2()

        self.feature_dim = efficientnet.classifier[1].in_features

        efficientnet.features[0][0] = nn.Conv2d(TEMPORAL_FRAME_WINDOW, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove LSTM layers and update classifier:
        self.classifier = nn.Linear(self.feature_dim, TARGET_CLASS_COUNT)

        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])

        # # resnet = resnet18()
        # # efficientnet = efficientnet_v2_s() # 24
        # # efficientnet = efficientnet_b4() # 48
        # # efficientnet = efficientnet_b3() # 40
        # # efficientnet = efficientnet_b1() # 32
        # # efficientnet = efficientnet_b2() # 32
        # efficientnet = efficientnet_b0()
        # # print(efficientnet.features[0][0])
        
        # # print(resnet)
        # if GRAYSCALE:
        #     # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #     efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # # self.feature_dim = resnet.fc.in_features  # typically 512
        # self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        # self.feature_dim = efficientnet.classifier[1].in_features
        
        # # LSTM to process sequential features from each frame
        # self.lstm1 = nn.LSTM(input_size=self.feature_dim, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=128, hidden_size=96, num_layers=1, batch_first=True)
        # self.lstm3 = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)
        # # Final classifier layer
        # self.classifier = nn.Linear(64, TARGET_CLASS_COUNT)


    def forward(self, x):
        # # x shape: (batch_size, num_frames, C, H, W)
        x = x.squeeze(2)  # Now x shape: (batch_size, num_frames, H, W)
        # Pass the tensor directly; num_frames is used as the channel dimension
        features = self.feature_extractor(x)  # output shape: (batch_size, self.feature_dim, 1, 1)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

        # # x shape: (batch_size, num_frames, C, H, W)
        # batch_size, num_frames, C, H, W = x.size()
        # # Reshape to (batch_size*num_frames, C, H, W) for CNN processing
        # x = x.view(batch_size * num_frames, C, H, W)
        # features = self.feature_extractor(x)  # output shape: (batch_size*num_frames, feature_dim, 1, 1)
        # features = features.view(batch_size, num_frames, self.feature_dim)
        # # Process the sequence of features with LSTM
        # out, _ = self.lstm1(features)
        # out, _ = self.lstm2(out)
        # lstm_out, _ = self.lstm3(out)
        # # Use the output from the final time step for classification
        # out = self.classifier(lstm_out[:, -1, :])
        # return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet = resnet18()
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
    

class r21d(nn.Module):
    def __init__(self):
        super(r21d, self).__init__()
        r2plus1d = r2plus1d_18()

        self.feature_dim = r2plus1d.fc.in_features

        if GRAYSCALE:
            r2plus1d.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        
        r2plus1d.fc = nn.Identity()  # Remove the final classification layer
        self.model = nn.Sequential(
            r2plus1d,
            nn.Linear(self.feature_dim, TARGET_CLASS_COUNT)
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, C, H, W)
        # Reshape to (batch_size, C, num_frames, H, W) for 3D CNN processing
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size, C, num_frames, H, W)
        return self.model(x)

class VideoModel3sD(nn.Module):
    def __init__(self):
        super(VideoModel3sD, self).__init__()
        self.model = s3d(dropout = 0.5)
        
        if GRAYSCALE:
            self.model.features[0][0][0] = nn.Conv3d(1, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 5, 5), bias=False)
        
        # overwrite the final classifier layer
        self.model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model.classifier = nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Conv3d(1024, TARGET_CLASS_COUNT, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, C, H, W)
        # Reshape to (batch_size, C, num_frames, H, W) for 3D CNN processing
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size, C, num_frames, H, W)
        x = self.model(x)
        return x


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
    def __init__(self, num_frames=TEMPORAL_FRAME_WINDOW, d_model=256, nhead=8, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.resnet = resnet18()
        # Remove final classification layer to extract features
        if GRAYSCALE:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Identity()
        
        self.feature_proj = nn.Linear(512, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, TARGET_CLASS_COUNT)
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnet(x)  # shape: (batch_size * num_frames, 2048)
        features = features.view(batch_size, num_frames, -1)  # shape: (batch_size, num_frames, 2048)
        features = self.feature_proj(features) # shape: (batch_size, num_frames, d_model)
        features = self.norm(features)

        # Add positional encoding
        features = self.pos_encoder(features)
        # Transformer expects (seq_len, batch_size, d_model)
        features = features.transpose(0, 1)
        transformed = self.transformer_encoder(features)
        # Aggregate features (e.g., mean pooling over time)
        aggregated = transformed.mean(dim=0)
        out = self.fc(aggregated)
        return out
