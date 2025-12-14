import pickle
import torch
import pandas as pd

from run import get_config, get_best, get_PacmanNetwork, get_PacmanAgent


class SubmissionWriter:
    def __init__(self, test_set_path, folder, version):
        """
        Initialize the writing of your submission.
        Pay attention that the test set only contains GameState objects,
        it's no longer (GameState, action) pairs.

        Arguments:
            test_set_path: The file path to the pickled test set.
            model_path: The file path to the trained model.
        """
        with open(test_set_path, "rb") as f:
            self.test_set = pickle.load(f)

        config = get_config(folder, version)
        self.model = get_PacmanNetwork(config)
        self.pacman = get_PacmanAgent(self.model, config)
        self.model.load_state_dict(torch.load(f"models/{folder}/pacman_model_V{version}.pth", map_location="cpu"))
        self.model.eval()

    def predict_on_testset(self):
        """
        Generate predictions for the test set.

        Your predicted actions should follow the same order
        as the test set provided.
        """
        actions = []
        for state in self.test_set:
            action = self.pacman.get_action(state)
            actions.append(action)
        return actions

    def write_csv(self, actions, file_name="submission"):
        """
        ! Do not modify !

        Write the predicted actions (North, South, ...)
        to a CSV file.

        """
        submission = pd.DataFrame(
            data={
                'ACTION': actions,
            },
            columns=["ACTION"]
        )

        submission.to_csv(file_name + ".csv", index=False)


if __name__ == "__main__":
    folder = 4
    version = get_best(folder, index=1)
    modelPath = f"models/{folder}/pacman_model_V{version}"
    writer = SubmissionWriter(
        "datasets/pacman_test.pkl",
        folder,
        version
    )
    predictions = writer.predict_on_testset()
    writer.write_csv(predictions)
