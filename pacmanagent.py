from pacman_module.game import Agent, Directions

from data import state_to_tensor
from train import PacmanDataset
from train import ACTION_INDEX


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

    def get_action(self, state):
        """
        Return the action chosen by the neural network given the
        current state.

        Arguments:
            state: a GameState object
        """
        x = state_to_tensor(state).unsqueeze(0)

        output = self.model(x)
        predictedInd = output.argmax(dim=1).item()
        action = invert_dictionnary(ACTION_INDEX, predictedInd)

        return action
