import torch
import torch.nn as nn

import sys

class Conv(nn.Module):
    def __init__(self, out_channels, init_channels=3, input_size=48):
        super(Conv, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.4)
            )
        
        self.conv_features = nn.Sequential(
            conv_block(init_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
        self.output_H = input_size // (2 * 2 * 2)
        self.output_W = input_size // (2 * 2 * 2)
    
    def forward(self, x):
        x = self.conv_features(x)
        return x
    
class MuffinCNN(nn.Module):
    def __init__(self, input_size, embed_dim=128, hidden_dim=256, num_features=1, num_classes=3):
        super(MuffinCNN, self).__init__()

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
        
        self.agap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
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
        x = self.agap(x)
        x = x.view(x.size(0), -1) 

        return self.classifier(x)

################# 2026-03-23 ↓
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_att = self.channel_att(x).view(x.size(0), x.size(1), 1, 1)
        x = x * c_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        s_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * s_att

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x) # Skip Connection
        out = self.relu(out)
        return out
    
class ResidualConv(nn.Module):
    def __init__(self, out_channels, init_channels=3, input_size=48):
        super(ResidualConv, self).__init__()

        self.layer1 = ResidualBlock(init_channels, 32, stride=2, dilation=1)
        self.layer2 = ResidualBlock(32, 64, stride=1, dilation=2)
        self.layer3 = ResidualBlock(64, 128, stride=2, dilation=1)
        
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        
        self.output_H = input_size // 4
        self.output_W = input_size // 4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        return x
    
class ResidualMuffinCNN(nn.Module):
    def __init__(self, input_size, embed_dim=128, hidden_dim=256, num_features=1, num_classes=3):
        super(ResidualMuffinCNN, self).__init__()

        self.embed_dim = embed_dim
        dim_per_cnn = embed_dim // num_features

        if num_features == 1:
            self.cnn_f1 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)
            
        elif num_features == 2:
            self.cnn_f1 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f2 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)

        else:
            self.cnn_f1 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f2 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)
            self.cnn_f3 = ResidualConv(input_size=input_size, out_channels=dim_per_cnn)

        self.f_h = self.cnn_f1.output_H
        self.f_w = self.cnn_f1.output_W
    
        total_in_channels = dim_per_cnn * num_features

        self.fusion = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.se = SEBlock(hidden_dim, reduction=16)
        # self.cbam = CBAM(hidden_dim, reduction=16)
        self.residual_fusion = ResidualBlock(hidden_dim, hidden_dim, stride=1)
        self.agap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
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
        x = self.fusion(x)
        x = self.se(x)
        # x = self.cbam(x)
        x = self.residual_fusion(x)
        x = self.agap(x)
        x = x.view(x.size(0), -1) 

        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def muffincnn_singlefeature(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=1, num_classes=num_classes)

def muffincnn_dualfusion(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=2, num_classes=num_classes)

def muffincnn_triplefusion(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=3, num_classes=num_classes)

def residualmuffincnn(input_size, num_features, num_classes):
    return ResidualMuffinCNN(
        input_size=input_size, 
        num_features=num_features, 
        num_classes=num_classes
    )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage >> python model.py <input_size> <num_classes>')
    else:
        input_size = int(sys.argv[1])
        num_classes = int(sys.argv[2])

        img1 = torch.randn(6, 3, input_size, input_size)
        img2 = torch.randn(6, 3, input_size, input_size)
        img3 = torch.randn(6, 3, input_size, input_size)

        model1 = muffincnn_singlefeature(input_size=input_size, num_classes=num_classes)
        model2 = muffincnn_dualfusion(input_size=input_size, num_classes=num_classes)
        model3 = muffincnn_triplefusion(input_size=input_size, num_classes=num_classes)

        logits = model1(img1)

        print(logits.shape)
        print(count_parameters(model1))

        logits = model2(img1, img2)

        print(logits.shape)
        print(count_parameters(model2))
        
        logits = model3(img1, img2, img3)

        print(logits.shape)
        print(count_parameters(model3))

        res_model1 = residualmuffincnn(input_size=input_size, num_features=1, num_classes=num_classes)
        res_model2 = residualmuffincnn(input_size=input_size, num_features=2, num_classes=num_classes)
        res_model3 = residualmuffincnn(input_size=input_size, num_features=3, num_classes=num_classes)

        logits = res_model1(img1)

        print(logits.shape)
        print(count_parameters(res_model1))

        logits = res_model2(img1, img2)

        print(logits.shape)
        print(count_parameters(res_model2))
        
        logits = res_model3(img1, img2, img3)

        print(logits.shape)
        print(count_parameters(res_model3))

        print(res_model3)