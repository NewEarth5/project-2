import numpy as np

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pacman_module.game import Directions
from architecture import PacmanNetwork
from data import PacmanDataset

ACTION_INDEX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}


def actions_convert(actions):
    actionsInd = []

    for action in actions:
        actionsInd.append(ACTION_INDEX[action])

    return torch.tensor(actionsInd, dtype=torch.long)


class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.path = path
        self.dataset = PacmanDataset(self.path)
        self.model = PacmanNetwork()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        print("Beginning of the training of your network...")

        self.model.train()
        trainLoader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True
        )

        epochsNum = 100

        for epoch in range(epochsNum):
            lossTotal = 0.0
            batchesNum = 0

            for batchInputs, batchActions in trainLoader:
                batchActionsInd = actions_convert(batchActions)

                self.optimizer.zero_grad()
                batchOutputs = self.model(batchInputs)
                batchLoss = self.criterion(batchOutputs, batchActionsInd)
                batchLoss.backward()
                self.optimizer.step()

                lossTotal += batchLoss.item()
                batchesNum += 1

            lossAvg = lossTotal / batchesNum
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochsNum}], Average Loss: {lossAvg:.4f}")

        torch.save(self.model.state_dict(), "models/pacman_model.pth")
        print("Model saved !")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
