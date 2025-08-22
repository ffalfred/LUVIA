import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
from torch import nn
import os

class CNNEncoder_old(nn.Module):
    def __init__(self, encoded_dim=256):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 12 * 14, encoded_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Define the CNNEncoder model
class CNNEncoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64 * 12 * 14, encoded_dim)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x1 = x.clone()
        x = self.relu2(self.bn2(self.conv2(x)))
        x2 = x.clone()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, x1, x2


# Guided Backpropagation
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class GuidedBackpropModel(nn.Module):
    def __init__(self, model):
        super(GuidedBackpropModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = GuidedReLU.apply(self.model.bn1(self.model.conv1(x)))
        x = GuidedReLU.apply(self.model.bn2(self.model.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x
