import numpy as np

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """

    def __init__(self, input_size):
        super().__init__()
        layer1_size = 2**round(np.log2(input_size * 3))
        layer2_size = layer1_size // 2
        layer3_size = layer2_size // 2
        output_size = 5

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, layer3_size)
        self.output = nn.Linear(layer3_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)
