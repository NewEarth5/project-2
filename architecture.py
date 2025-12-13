import numpy as np

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.input = tensor.shape[0]
        self.layer1 = nn.Linear(
            self.input,
            2**round(np.log2(self.input * 3))
        )
        self.layer2 = nn.Linear(
            self.layer1.out_features,
            self.layer1.out_features / 2
        )
        self.layer3 = nn.Linear(
            self.layer2.out_features,
            self.layer2.out_features / 2
        )
        self.output = nn.Linear(
            self.layer3.out_features,
            5
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)
