import numpy as np

import pickle
import torch
from torch.utils.data import Dataset


def state_to_tensor(state):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    walls = state.getWalls()
    food = state.getFood()

    pacmanPos = np.array(state.getPacmanPosition())

    ghostsPos = np.array(state.getGhostPositions())
    ghostCloInd = np.argmin(sum(abs(ghostsPos - pacmanPos).T))
    ghostCloPos = ghostsPos[ghostCloInd]

    wallN = float(walls[pacmanPos[0]][pacmanPos[1] + 1])
    wallE = float(walls[pacmanPos[0] + 1][pacmanPos[1]])
    wallS = float(walls[pacmanPos[0]][pacmanPos[1] - 1])
    wallW = float(walls[pacmanPos[0] - 1][pacmanPos[1]])

    foodN = float(food[pacmanPos[0]][pacmanPos[1] + 1])
    foodE = float(food[pacmanPos[0] + 1][pacmanPos[1]])
    foodS = float(food[pacmanPos[0]][pacmanPos[1] - 1])
    foodW = float(food[pacmanPos[0] - 1][pacmanPos[1]])

    tensor = torch.tensor([
        pacmanPos[0],    # Pacman's x position
        pacmanPos[1],    # Pacman's y position
        ghostCloPos[0],  # Closest Ghost's x position
        ghostCloPos[1],  # Closest Ghost's y position
        wallN,           # Whether there is a wall north
        wallE,           # Whether there is a wall east
        wallS,           # Whether there is a wall south
        wallW,           # Whether there is a wall west
        foodN,           # Whether there is food north
        foodE,           # Whether there is food east
        foodS,           # Whether there is food south
        foodW,           # Whether there is food west
    ], dtype=torch.float32)
    return tensor


class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]
