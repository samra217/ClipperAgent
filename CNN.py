import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, c_in=14, h=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(c_in,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,h,5,padding=5), nn.ReLU(), nn.MaxPool1d(5)
        )
        self.head = nn.Sequential(
            nn.Conv1d(h,h,5,padding=2), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(h,1,1)
        )
        self.pos_bias = nn.Parameter(torch.zeros(100))

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)+self.pos_bias
