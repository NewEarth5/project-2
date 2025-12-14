import numpy as np

import pickle
import torch
from torch.utils.data import Dataset

from pacman_module.game import Directions

ACTION_INDEX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}

DIRECTION_MAPPING = {
    Directions.NORTH: [1, 0, 0, 0],
    Directions.EAST: [0, 1, 0, 0],
    Directions.SOUTH: [0, 0, 1, 0],
    Directions.WEST: [0, 0, 0, 1],
    Directions.STOP: [0, 0, 0, 0]
}

TENSOR_SIZE = 24


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
    pacmanState = state.getPacmanState()

    walls = state.getWalls()
    food = state.getFood()

    pacmanPos = np.array(state.getPacmanPosition())
    pacmanPosNorm = position_normalize(pacmanPos, walls)

    ghostsPos = np.array(state.getGhostPositions())
    ghostCloInd = np.argmin(sum(abs(ghostsPos - pacmanPos).T))
    ghostCloPos = ghostsPos[ghostCloInd]
    ghostCloPosNorm = position_normalize(ghostCloPos, walls)

    wallNN = float(walls[pacmanPos[0] + 0][pacmanPos[1] + 1])
    wallNE = float(walls[pacmanPos[0] + 1][pacmanPos[1] + 1])
    wallEE = float(walls[pacmanPos[0] + 1][pacmanPos[1] + 0])
    wallSE = float(walls[pacmanPos[0] + 1][pacmanPos[1] - 1])
    wallSS = float(walls[pacmanPos[0] + 0][pacmanPos[1] - 1])
    wallSW = float(walls[pacmanPos[0] - 1][pacmanPos[1] - 1])
    wallWW = float(walls[pacmanPos[0] - 1][pacmanPos[1] + 0])
    wallNW = float(walls[pacmanPos[0] - 1][pacmanPos[1] + 1])

    foodNN = float(food[pacmanPos[0] + 0][pacmanPos[1] + 1])
    foodNE = float(food[pacmanPos[0] + 1][pacmanPos[1] + 1])
    foodEE = float(food[pacmanPos[0] + 1][pacmanPos[1] + 0])
    foodSE = float(food[pacmanPos[0] + 1][pacmanPos[1] - 1])
    foodSS = float(food[pacmanPos[0] + 0][pacmanPos[1] - 1])
    foodSW = float(food[pacmanPos[0] - 1][pacmanPos[1] - 1])
    foodWW = float(food[pacmanPos[0] - 1][pacmanPos[1] + 0])
    foodNW = float(food[pacmanPos[0] - 1][pacmanPos[1] + 1])

    currentDir = pacmanState.configuration.direction
    currentDirVec = DIRECTION_MAPPING[currentDir]

    tensor = torch.tensor([
        pacmanPosNorm[0],    # Pacman's normalized x position
        pacmanPosNorm[1],    # Pacman's normalized y position
        ghostCloPosNorm[0],  # Closest Ghost's normalized x position
        ghostCloPosNorm[1],  # Closest Ghost's normalized y position
        wallNN,              # Whether there is a wall north
        wallNE,              # Whether there is a wall north east
        wallEE,              # Whether there is a wall east
        wallSE,              # Whether there is a wall south east
        wallSS,              # Whether there is a wall south
        wallSW,              # Whether there is a wall south west
        wallWW,              # Whether there is a wall west
        wallNW,              # Whether there is a wall north west
        foodNN,              # Whether there is food north
        foodNE,              # Whether there is food north east
        foodEE,              # Whether there is food east
        foodSE,              # Whether there is food south east
        foodSS,              # Whether there is food south
        foodSW,              # Whether there is food south west
        foodWW,              # Whether there is food west
        foodNW,              # Whether there is food north west
        currentDirVec[0],    # Whether pacman is moving north
        currentDirVec[1],    # Whether pacman is moving east
        currentDirVec[2],    # Whether pacman is moving south
        currentDirVec[3],    # Whether pacman is moving west
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
            self.actions.append(ACTION_INDEX[a])

        self.inputs = torch.stack(self.inputs)
        self.actions = torch.tensor(self.actions, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]


if __name__ == "__main__":
    from train import Pipeline

    pipeline = Pipeline("datasets/pacman_dataset.pkl", save=False)
    pipeline.train()
