import numpy as np
import random
import json

import torch
from torch import nn

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent
from train import VERSION, get_layer_size_fun, get_action, get_layer_fun, get_normal_fun

USEDVERSION = f"V{VERSION}-{500}"
# USEDVERSION = "V1-100"

modelPath = f"models/pacman_model_{USEDVERSION}.pth"


with open(f"models/pacman_model_{USEDVERSION}.json", "r") as file:
    config = json.load(file)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Neural Network
inputSize = config['network']['inputSize']
outputSize = config['network']['outputSize']
layersNum = config['network']['layersNum']
layer1Size = config['network']['layer1Size']
layer_size_fun = get_layer_size_fun(config['network']['layerSizeFunName'])
layer_fun = get_layer_fun(config['network']['layerFunName'])
doNormal = config['network']['doNormal']
normal_fun = get_normal_fun(config['network']['normalFunName'])
action = get_action(config['network']['actionName'])
doDropout = config['network']['doDropout']
dropoutRate = config['network']['dropoutRate']

model = PacmanNetwork(
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
)
model.load_state_dict(torch.load(modelPath, map_location="cpu"))
model.eval()

# Pacman Agent
doNormalPos = config['dataset']['doNormalPos']
viewDistance = config['dataset']['viewDistance']

pacman_agent = PacmanAgent(model, doNormalPos, viewDistance)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
