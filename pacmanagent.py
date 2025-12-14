import numpy as np
from numpy.random import choice

import torch

from pacman_module.game import Agent

from data import ACTION_INDEX, state_to_tensor


def invert_dictionnary(dictionnary, value):
    return list(dictionnary.keys())[list(dictionnary.values()).index(value)]


class PacmanAgent(Agent):
    def __init__(self, model):
        """
        Initialize the neural network Pacman agent.

        Arguments:
            model: The trained neural network model.
        """
        super().__init__()

        self.model = model.eval()

    def get_prediction(self, output, creativity=0, selectMax=None):
        prob = output[0].detach().cpu().numpy()
        best = np.argmax(prob)

        for i in range(len(prob)):
            prob[i] = max(0, prob[i])

        sort = np.argsort(prob)

        if selectMax is not None:
            for i in range(len(prob) - selectMax):
                prob[sort[i]] = 0

        prob = prob / sum(prob)
        prob *= creativity
        prob[best] += 1 - creativity

        return choice(list(ACTION_INDEX.keys()), p=prob, size=1)[0]

    def get_action(self, state):
        """
        Return the action chosen by the neural network given the
        current state.

        Arguments:
            state: a GameState object
        """
        with torch.no_grad():
            x = state_to_tensor(state).unsqueeze(0)
            output = self.model(x)

            return self.get_prediction(output)
