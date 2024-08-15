import torch.nn.functional as F
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


class FED_EMNISTCNN(nn.Module):

    def __init__(self):
        super(FED_EMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # self.layer2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # self.bn = nn.BatchNorm2d(64)

        # self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc = nn.Linear(3136, 10)


 


    def forward(self, x):
        #activations = []
        x = self.layer1(x)
        x = self.layer2(x)
        #activations.append(x)
        # x = F.relu(self.bn(x))
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x=F.softmax(x,1)

        return x


