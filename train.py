import numpy as np

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pacman_module.game import Directions
from architecture import PacmanNetwork
from data import PacmanDataset

VERSION = 6.1
EPOCHSNUM = 100

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
    def __init__(self, path, save=True, validSplit=0.2):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.save = save

        self.path = path
        self.dataset = PacmanDataset(self.path)
        self.model = PacmanNetwork()

        datasetSize = len(self.dataset)
        validSize = int(datasetSize * validSplit)
        trainSize = datasetSize - validSize
        self.datasetTrain, self.datasetValid = torch.utils.data.random_split(
            self.dataset, [trainSize, validSize]
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def train(self, version=1, epochsNum=100, precision=0):
        print("Beginning of the training of your network...")

        batchSize = 64
        lossPrev = np.inf

        trainLoader = DataLoader(
            self.datasetTrain,
            batch_size=batchSize,
            shuffle=True
        )
        validLoader = DataLoader(
            self.datasetValid,
            batch_size=batchSize,
            shuffle=False
        )

        print(f"Dataset size: {len(trainLoader) + len(validLoader)}")  # Dataset size 235

        for epoch in range(epochsNum):
            self.model.train()
            trainLossTot = 0.0
            trainCor = 0
            trainTot = 0

            for batchInputs, batchActions in trainLoader:
                batchActionsInd = actions_convert(batchActions)

                self.optimizer.zero_grad()
                batchOutputs = self.model(batchInputs)
                batchLoss = self.criterion(batchOutputs, batchActionsInd)
                batchLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                trainLossTot += batchLoss.item()
                _, predicted = batchOutputs.max(1)
                trainTot += batchActionsInd.size(0)
                trainCor += predicted.eq(batchActionsInd).sum().item()

            trainLossAvg = trainLossTot / len(trainLoader)
            trainAcc = 100. * trainCor / trainTot

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochsNum}]")
                print(f"  Train Loss: {trainLossAvg:.4f}, Train Acc: {trainAcc:.5f}%")

            if len(validLoader) != 0:
                self.model.eval()
                validLossTot = 0.0
                validCor = 0
                validTot = 0

                with torch.no_grad():
                    for batchInputs, batchActions in validLoader:
                        batchActionsInd = actions_convert(batchActions)
                        batchOutputs = self.model(batchInputs)
                        batchLoss = self.criterion(batchOutputs, batchActionsInd)

                        validLossTot += batchLoss.item()
                        _, predicted = batchOutputs.max(1)
                        validTot += batchActionsInd.size(0)
                        validCor += predicted.eq(batchActionsInd).sum().item()

                validLossAvg = validLossTot / len(validLoader)
                validAcc = 100. * validCor / validTot

                self.scheduler.step(validLossAvg)

                if (epoch + 1) % 10 == 0:
                    print(f"  Valid Loss: {validLossAvg:.4f}, Valid Acc: {validAcc:.5f}%")

                if abs(lossPrev - validLossAvg) < precision:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                lossPrev = validLossAvg
            else:
                self.scheduler.step(trainLossAvg)

                if abs(lossPrev - trainLossAvg) < precision:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                lossPrev = trainLossAvg

        if (self.save):
            torch.save(self.model.state_dict(), f"models/pacman_model_V{VERSION}-{epochsNum}.pth")
            print("Model saved !")
        print("Finished training your network model...")


if __name__ == "__main__":
    pipeline = Pipeline("datasets/pacman_dataset.pkl", validSplit=0.2)
    pipeline.train(version=VERSION, epochsNum=EPOCHSNUM)
