from collections import OrderedDict
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """

    def __init__(
        self,
        inputSize,
        outputSize,
        layersNum,
        layer1Size,
        layer_size_fun,
        layer_fun,
        doNormal,
        normal_fun,
        action,
        doDropout,
        dropoutRate
    ):
        super().__init__()
        ordDict = OrderedDict()
        layerSizePrev = inputSize
        for i in range(layersNum):
            if i == 0:
                layerSize = layer1Size
            else:
                layerSize = layer_size_fun(layerSizePrev, i, layersNum)

            ordDict[f"layer{i + 1}"] = layer_fun(layerSizePrev, layerSize)
            if doNormal:
                ordDict[f"normal{i + 1}"] = normal_fun(layerSize)
            ordDict[f"action{i + 1}"] = action()
            if doDropout:
                ordDict[f"dropout{i + 1}"] = nn.Dropout(dropoutRate)

            layerSizePrev = layerSize

        ordDict["output"] = nn.Linear(layerSizePrev, outputSize)

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(ordDict)

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)
