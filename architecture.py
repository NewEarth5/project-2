import numpy as np

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """

    def __init__(self):
        super().__init__()
        input_size = 24
        layer1_size = 128
        layer2_size = layer1_size // 2
        layer3_size = layer2_size // 2
        output_size = 5

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.normal1 = nn.BatchNorm1d(layer1_size)
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.normal2 = nn.BatchNorm1d(layer2_size)
        self.dropout2 = nn.Dropout(0.3)

        self.layer3 = nn.Linear(layer2_size, layer3_size)
        self.normal3 = nn.BatchNorm1d(layer3_size)
        self.dropout3 = nn.Dropout(0.2)

        self.output = nn.Linear(layer3_size, output_size)

        # self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.activation = nn.GELU()
        # self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.normal1(self.layer1(x)))
        x = self.dropout1(x)

        x = self.activation(self.normal2(self.layer2(x)))
        x = self.dropout2(x)

        x = self.activation(self.normal3(self.layer3(x)))
        x = self.dropout3(x)

        return self.output(x)


if __name__ == "__main__":
    from train import Pipeline

    pipeline = Pipeline("datasets/pacman_dataset.pkl", save=False)
    pipeline.train()
