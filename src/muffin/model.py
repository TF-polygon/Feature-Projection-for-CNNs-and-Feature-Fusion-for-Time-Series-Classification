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

        # self.classifier = nn.Sequential(
        #     nn.Linear(final_vec_size, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        
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
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def muffincnn_singlefeature(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=1, num_classes=num_classes)

def muffincnn_dualfusion(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=2, num_classes=num_classes)

def muffincnn_triplefusion(input_size, num_classes):
    return MuffinCNN(input_size=input_size, num_features=3, num_classes=num_classes)

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