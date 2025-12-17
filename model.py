import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Conv(nn.Module):
    def __init__(self, out_channels, init_channels=3, input_size=6):
        super(Conv, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(0.3)
            )
        
        self.conv_features = nn.Sequential(
            conv_block(init_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
        self.output_H = input_size
        self.output_W = input_size
    
    def forward(self, x):
        x = self.conv_features(x)
        return x
    
class MultiFeatureFusion(nn.Module):
    def __init__(self, input_size, embed_dim=128, hidden_dim=256, num_features=1, num_classes=3):
        super(MultiFeatureFusion, self).__init__()

        self.embed_dim = embed_dim
        dim_per_cnn = embed_dim // num_features

        if num_features == 1:
            self.cnn_f1 = Conv(input_size=input_size, out_channels=dim_per_cnn)
            
        elif num_features == 2:
            self.cnn_f1 = Conv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f2 = Conv(input_size=input_size, out_channels=dim_per_cnn)

        else:
            self.cnn_f1 = Conv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f2 = Conv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f3 = Conv(input_size=input_size, out_channels=dim_per_cnn)

        self.f_h = self.cnn_f1.output_H
        self.f_w = self.cnn_f1.output_W
    
        total_in_channels = dim_per_cnn * num_features
        self.fusion_block = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        final_vec_size = hidden_dim * self.f_h * self.f_w

        self.classifier = nn.Sequential(
            nn.Linear(final_vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, f_1, f_2=None, f_3=None):
        f_list = []
        if f_2 == None and f_3 == None:
            f1 = self.cnn_f1(f_1)
            f_list.append(f1)

        elif f_2 != None and f_3 == None:            
            f1 = self.cnn_f1(f_1)
            f2 = self.cnn_f2(f_2)
            f_list.append(f1)
            f_list.append(f2)

        else:
            f1 = self.cnn_f1(f_1)
            f2 = self.cnn_f2(f_2)
            f3 = self.cnn_f3(f_3)
            f_list.append(f1)
            f_list.append(f2)
            f_list.append(f3)

        x = torch.cat(f_list, dim=1)
        x = self.fusion_block(x)
        x = x.flatten(start_dim=1)
        
        return self.classifier(x)

class MFCT_Net(nn.Module):
    def __init__(self, input_size, input_dim=48, embed_dim=384, hidden_dim=512, num_heads=8, num_layers=3, num_classes=2):
        super(MFCT_Net, self).__init__()

        self.embed_dim = embed_dim
        channel_per_cnn = embed_dim // 3

        self.cnn_gasf = Conv(input_size=input_size, out_channels=channel_per_cnn)
        self.cnn_gadf = Conv(input_size=input_size, out_channels=channel_per_cnn)
        self.cnn_rp = Conv(input_size=input_size, out_channels=channel_per_cnn)

        self.feature_h = self.cnn_gasf.output_H
        self.feature_w = self.cnn_gasf.output_W
        self.seq_len = self.feature_h * self.feature_w

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.positional_embedding = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))

        self.fc_head = nn.Linear(embed_dim, num_classes)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, gasf, gadf, rp):
        f_gasf = self.cnn_gasf(gasf)
        f_gadf = self.cnn_gadf(gadf)
        f_rp = self.cnn_rp(rp)

        # Prepare to input into Transformer
        x = torch.cat([f_gasf, f_gadf, f_rp], dim=1)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        x = x + self.positional_embedding

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        x = self.fc_head(x)
        # x = self.fc(x)a
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def single_feature_model(input_size):
    return MultiFeatureFusion(input_size=input_size, num_features=1)

def double_features_model(input_size):
    return MultiFeatureFusion(input_size=input_size, num_features=2)

def multi_features_model(input_size):
    return MultiFeatureFusion(input_size=input_size, num_features=3)

def multi_features_mfct_net(input_size):
    return MFCT_Net(input_size=input_size, num_classes=3)

if __name__ == '__main__':
    input_size = 24 

    img1 = torch.randn(6, 3, input_size, input_size)
    img2 = torch.randn(6, 3, input_size, input_size)
    img3 = torch.randn(6, 3, input_size, input_size)

    single_model = single_feature_model(input_size)
    double_model = double_features_model(input_size)
    _multi_model = multi_features_model(input_size)

    logits = single_model(img1)
    print(logits.shape)
    print(count_parameters(single_model))

    logits = double_model(img1, img2)
    print(logits.shape)
    print(count_parameters(double_model))

    logits = _multi_model(img1, img2, img3)
    print(logits.shape)
    print(count_parameters(_multi_model))