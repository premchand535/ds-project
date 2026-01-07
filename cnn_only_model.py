import torch
import torch.nn as nn
from torchvision import models

class CNNOnlyModel(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()

        if backbone == "resnet18":
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        n_feats = cnn.fc.in_features
        cnn.fc = nn.Identity()
        self.cnn = cnn

        self.regressor = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img):
        x = self.cnn(img)
        out = self.regressor(x).squeeze(1)
        return out
