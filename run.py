import numpy as np
import random
import json

import torch
from torch import nn

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent
from train import get_layer_size_fun, get_action, get_layer_fun, get_normal_fun, get_tensor_size, get_best


def get_config(folder, version):
    with open(f"models/{folder}/pacman_model_V{version}.json", "r") as file:
        config = json.load(file)

    print(f"{'=' * 60}")
    print(f"Best model from folder {folder}:")
    print(f"  Model {config['trial_id']}")
    print(f"  - Loss of {config['performance']['loss']:.4f}")
    print(f"  - Accuracy of {config['performance']['accuracy']:.2f}%")
    print(f"{'=' * 60}")

    return config


def get_PacmanNetwork(config):
    inputSize = get_tensor_size(config['dataset']['viewDistance'])
    outputSize = 5
    layersNum = config['network']['layersNum']
    layer1Size = int(inputSize * config['network']['layer1SizeMultiplier'])
    layer_size_fun = get_layer_size_fun(config['network']['layerSizeFun'])
    layer_fun = get_layer_fun(config['network']['layerFun'])
    doNormal = config['network']['doNormal']
    normal_fun = get_normal_fun(config['network']['normalFun'])
    action = get_action(config['network']['action'])
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
    return model


def get_PacmanAgent(model, config):
    doNormalPos = config['dataset']['doNormalPos']
    viewDistance = config['dataset']['viewDistance']
    pacman_agent = PacmanAgent(model, doNormalPos, viewDistance)
    return pacman_agent


def run(folder=1, version=1):
    config = get_config(folder, version)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    model = get_PacmanNetwork(config)
    model.load_state_dict(torch.load(f"models/{folder}/pacman_model_V{version}.pth", map_location="cpu"))
    model.eval()
    pacman_agent = get_PacmanAgent(model, config)

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


if __name__ == "__main__":
    folder = 5
    version = get_best(folder, index=0)
    run(folder, version)
