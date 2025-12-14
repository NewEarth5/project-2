import numpy as np
from collections import deque

import pickle
import torch
from torch.utils.data import Dataset

from pacman_module.pacman import manhattanDistance
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


DIRECTIONS_COORD = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0)
]


def get_tensor_size(viewDistance):
    size = 0
    size += 2                                       # Pacman's position
    size += 2                                       # Ghost's position
    size += 2 * viewDistance * (viewDistance + 1)   # Wall radar
    size += 2 * viewDistance * (viewDistance + 1)   # Food radar
    size += 1                                       # Number of food
    size += 4                                       # Pacman's direction
    size += 4                                       # Wall in direction
    size += 4                                       # Food amount in direction
    size += 4                                       # Ghost amount in direction
    size += 5                                       # Legal moves
    size += 5                                       # Danger detection
    size += 2                                       # Food cluster
    size += 2                                       # BFS food dist and num
    return size


def position_normalize(position, walls, doNormalPos):
    if doNormalPos:
        max_x = walls.width - 1
        max_y = walls.height - 1
        max_pos = np.array((max_x, max_y))
        return position / max_pos
    return position


def count_direction(truthFun, pacmanPos, walls, normalise=1):
    output = []

    for dx, dy in DIRECTIONS_COORD:
        count = 0
        dist = 0
        while True:
            dist += 1
            x = pacmanPos[0] + dx * dist
            y = pacmanPos[1] + dy * dist

            if is_outbounds(x, y, walls):
                break

            if truthFun(x, y):
                count += 1

        output.extend([count / normalise])

    return output


def diamond_indexes(n):
    """
    Indexes to access a 2D array in a diamond pattern of size n.
    """
    output = []

    for r in range(n):
        for c in range(1, n - r + 1):
            output.append((r, c))
            output.append((c, -r))
            output.append((-r, -c))
            output.append((-c, r))
    return output


def is_outbounds(x, y, walls):
    return (x > walls.width - 1 or x < 0 or y > walls.height - 1 or y < 0)


def is_illegal(x, y, walls):
    return is_outbounds(x, y, walls) or walls[x][y]


def do_if_legal(legalActions, trueFun, falseFun):
    output = []

    for action in list(ACTION_INDEX.keys()):
        if action in legalActions:
            output.extend(trueFun(action))
        else:
            output.extend(falseFun(action))

    return output


def food_cluster(foodPos, pacmanPos, walls):
    output = []
    if foodPos:
        foodDist = [manhattanDistance(pacmanPos, fpos) for fpos in foodPos]
        output.extend([
            min(foodDist) / max(walls.width, walls.height),
            np.mean(foodDist[:min(5, len(foodPos))]) / max(walls.width, walls.height),
        ])

    return output


def distance_bfs(start, targets, walls, max_depth=10):
    queue = deque([(start, 0)])
    visited = [tuple(start)]
    closest = float('inf')
    count = 0

    while queue:
        pos, dist = queue.popleft()

        if dist > max_depth:
            break

        if tuple(pos) in targets:
            closest = min(closest, dist)
            count += 1

        for dx, dy in DIRECTIONS_COORD:
            posNext = (pos[0] + dx, pos[1] + dy)

            if posNext in visited:
                continue
            if is_illegal(posNext[0], posNext[1], walls):
                continue

            visited.append(tuple(posNext))
            queue.append((posNext, dist + 1))

    return closest, count


def state_to_tensor(state, doNormalPos, viewDistance):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    features = []

    diamondInd = diamond_indexes(viewDistance)

    pacmanState = state.getPacmanState()
    legalActions = state.getLegalPacmanActions()
    walls = state.getWalls()

    food = state.getFood()
    foodPos = food.asList()
    foodNum = 0
    for i in range(food.width):
        for j in range(food.height):
            foodNum += int(food[i][j])

    pacmanPos = np.array(state.getPacmanPosition())
    pacmanPosNorm = position_normalize(pacmanPos, walls, doNormalPos)

    features.extend([
        pacmanPosNorm[0],
        pacmanPosNorm[1],
    ])

    ghostsPos = np.array(state.getGhostPositions())
    ghostNum = len(ghostsPos)
    ghostCloInd = np.argmin(sum(abs(ghostsPos - pacmanPos).T))
    ghostCloPos = ghostsPos[ghostCloInd]
    ghostCloPosNorm = position_normalize(ghostCloPos, walls, doNormalPos)

    features.extend([
        ghostCloPosNorm[0],
        ghostCloPosNorm[1],
    ])

    for i, j in diamondInd:
        x = pacmanPos[0] + i
        y = pacmanPos[1] + j
        if is_outbounds(x, y, walls):
            features.append(float(False))
        else:
            features.append(float(walls[x][y]))

    for i, j in diamondInd:
        x = pacmanPos[0] + i
        y = pacmanPos[1] + j
        if is_outbounds(x, y, walls):
            features.append(float(False))
        else:
            features.append(float(food[x][y]))

    features.append(foodNum)

    currentDir = pacmanState.configuration.direction
    currentDirVec = DIRECTION_MAPPING[currentDir]

    features.extend([
        currentDirVec[0],    # Whether pacman is moving north
        currentDirVec[1],    # Whether pacman is moving east
        currentDirVec[2],    # Whether pacman is moving south
        currentDirVec[3],    # Whether pacman is moving west
    ])

    for direction in count_direction(
        lambda x, y: walls[x][y],
        pacmanPos, walls
    ):
        features.append(int(bool(direction)))

    features.extend(
        count_direction(
            lambda x, y: food[x][y],
            pacmanPos,
            walls,
            normalise=foodNum
        )
    )

    features.extend(
        count_direction(
            lambda x, y: ghostCloPos[0] == x and ghostCloPos[1] == y,
            pacmanPos,
            walls,
            normalise=ghostNum
        )
    )

    legal = do_if_legal(
        legalActions,
        lambda action: [1],
        lambda action: [0]
    )
    features.extend(legal)

    danger = do_if_legal(
        legalActions,
        lambda action: [1] if manhattanDistance(
            state.generatePacmanSuccessor(action).getPacmanPosition(),
            ghostCloPos
        ) <= 2 else [0],
        lambda action: [0]
    )
    features.extend(danger)

    features.extend(food_cluster(foodPos, pacmanPos, walls))

    foodDistClo, foodReachNum = distance_bfs(pacmanPos, foodPos, walls)
    features.extend([
        foodDistClo if foodDistClo != float('inf') else (walls.height + walls.width),
        foodReachNum / foodNum,
    ])

    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    def __init__(self, path, doNormalPos, viewDistance):
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

        for state, action in data:
            x = state_to_tensor(
                state,
                doNormalPos=doNormalPos,
                viewDistance=viewDistance
            )
            self.inputs.append(x)
            self.actions.append(ACTION_INDEX[action])

        self.inputs = torch.stack(self.inputs)
        self.actions = torch.tensor(self.actions, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]


if __name__ == "__main__":
    from train import search_training

    search_training("models", numTrials=5, save=False)
