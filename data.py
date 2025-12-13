import numpy as np

import pickle
import torch
from torch.utils.data import Dataset


def position_normalize(position, walls):
    max_x = walls.width - 1
    max_y = walls.height - 1
    max_pos = np.array((max_x, max_y))
    return position / max_pos


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

    pacmanPos = np.array(state.getPacmanPosition())
    pacmanPosNorm = position_normalize(pacmanPos, walls)

    ghostsPos = np.array(state.getGhostPositions())
    ghostCloInd = np.argmin(sum(abs(ghostsPos - pacmanPos).T))
    ghostCloPos = ghostsPos[ghostCloInd]
    ghostCloPosNorm = position_normalize(ghostCloPos, walls)

    wallN = float(walls[pacmanPos[0]][pacmanPos[1] + 1])
    wallE = float(walls[pacmanPos[0] + 1][pacmanPos[1]])
    wallS = float(walls[pacmanPos[0]][pacmanPos[1] - 1])
    wallW = float(walls[pacmanPos[0] - 1][pacmanPos[1]])

    tensor = torch.tensor([
        pacmanPosNorm[0],    # Pacman's x position
        pacmanPosNorm[1],    # Pacman's y position
        ghostCloPosNorm[0],  # Closest Ghost's x position
        ghostCloPosNorm[1],  # Closest Ghost's y position
        wallN,               # Whether there is a wall north
        wallE,               # Whether there is a wall east
        wallS,               # Whether there is a wall south
        wallW,               # Whether there is a wall west
    ])
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
