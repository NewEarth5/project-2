from collections import OrderedDict
import torch.nn as nn

from data import TENSOR_SIZE


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """

    def __init__(self, layersNum=3):
        super().__init__()
        inputSize = TENSOR_SIZE
        outputSize = 5
        ordDict = OrderedDict()
        layerSizePrev = inputSize
        for i in range(layersNum):
            if i == 0:
                layerSize = inputSize * 3
            else:
                layerSize = layerSizePrev * 2 // 3

            ordDict[f"layer{i + 1}"] = nn.Linear(layerSizePrev, layerSize)
            ordDict[f"normal{i + 1}"] = nn.BatchNorm1d(layerSize)
            ordDict[f"action{i + 1}"] = nn.GELU()
            ordDict[f"dropout{i + 1}"] = nn.Dropout(0.3)

            layerSizePrev = layerSize

        ordDict["output"] = nn.Linear(layerSizePrev, outputSize)

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(ordDict)

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)


if __name__ == "__main__":
    from train import Pipeline

    pipeline = Pipeline("datasets/pacman_dataset.pkl", save=False)
    pipeline.train()
