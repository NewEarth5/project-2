import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architecture import PacmanNetwork
from data import TENSOR_SIZE, N, PacmanDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
SCALER = torch.amp.GradScaler(enabled=USE_CUDA)
print(f"Using device: {DEVICE}")
if USE_CUDA:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

VERSION = 8.5


def get_layer_size_fun(name):
    if name == "two_thirds":
        return lambda prev, i, num: prev * 2 // 3
    raise ValueError()


def get_action(name):
    if name == "GELU":
        return nn.GELU()
    raise ValueError()


class Pipeline(nn.Module):
    def __init__(
        self,
        path,
        dataset,
        model,
        criterion,
        optimizer,
        doScheduler,
        scheduler,
        validSplit,
        patienceLimit=15
    ):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.path = path
        self.dataset = dataset
        self.model = model

        datasetSize = len(self.dataset)
        validSize = int(datasetSize * validSplit)
        trainSize = datasetSize - validSize
        self.datasetTrain, self.datasetValid = torch.utils.data.random_split(
            self.dataset, [trainSize, validSize]
        )

        self.criterion = criterion
        self.optimizer = optimizer
        if doScheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = None

        self.bestLoss = float('inf')
        self.patienceCount = 0
        self.patienceLimit = patienceLimit

    def train(
        self,
        epochsNum,
        batchSize,
        doNormal,
        normal,
        precision=0,
        precisionBest=False,
        version=1,
        save=True
    ):
        print("Beginning of the training of your network...")
        print(f"Device used: {next(self.model.parameters()).device}")

        lossPrev = float('inf')

        trainLoader = DataLoader(
            self.datasetTrain,
            batch_size=batchSize,
            shuffle=True,
            pin_memory=USE_CUDA,
            num_workers=2,
            persistent_workers=True
        )
        validLoader = DataLoader(
            self.datasetValid,
            batch_size=batchSize,
            shuffle=False,
            pin_memory=USE_CUDA,
            num_workers=2,
            persistent_workers=True
        )

        print(f"Dataset size: {len(self.dataset)}")  # Dataset size 15018

        for epoch in range(epochsNum):
            self.model.train()
            trainLossTot = 0.0
            trainCor = 0
            trainTot = 0

            for batchInputs, batchActions in trainLoader:
                batchInputs = batchInputs.to(DEVICE, non_blocking=USE_CUDA)
                batchActions = batchActions.to(DEVICE, non_blocking=USE_CUDA)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=USE_CUDA):
                    batchOutputs = self.model(batchInputs)
                    batchLoss = self.criterion(batchOutputs, batchActions)

                SCALER.scale(batchLoss).backward()
                SCALER.unscale_(self.optimizer)
                if doNormal:
                    normal
                SCALER.step(self.optimizer)
                SCALER.update()

                trainLossTot += batchLoss.item()
                _, predicted = batchOutputs.max(1)
                trainTot += batchActions.size(0)
                trainCor += predicted.eq(batchActions).sum().item()

            trainLossAvg = trainLossTot / len(trainLoader)
            trainAcc = 100. * trainCor / trainTot

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{epochsNum}]")
                print(f"  Train Loss: {trainLossAvg:.4f}, Train Acc: {trainAcc:.2f}%")

            if len(validLoader) != 0:
                self.model.eval()
                validLossTot = 0.0
                validCor = 0
                validTot = 0

                with torch.no_grad():
                    for batchInputs, batchActions in validLoader:
                        batchInputs = batchInputs.to(DEVICE, non_blocking=USE_CUDA)
                        batchActions = batchActions.to(DEVICE, non_blocking=USE_CUDA)

                        with torch.amp.autocast(device_type="cuda", enabled=USE_CUDA):
                            batchOutputs = self.model(batchInputs)
                            batchLoss = self.criterion(batchOutputs, batchActions)

                        validLossTot += batchLoss.item()
                        _, predicted = batchOutputs.max(1)
                        validTot += batchActions.size(0)
                        validCor += predicted.eq(batchActions).sum().item()

                validLossAvg = validLossTot / len(validLoader)
                validAcc = 100. * validCor / validTot
                lossAvg = validLossAvg

                if (epoch + 1) % 50 == 0:
                    print(f"  Valid Loss: {validLossAvg:.4f}, Valid Acc: {validAcc:.2f}%")
            else:
                lossAvg = trainLossAvg

            if self.scheduler is not None:
                self.scheduler.step(lossAvg)

            if precisionBest:
                if lossAvg < self.bestLoss:
                    self.bestLoss = lossAvg
                    self.patienceCount = 0
                    if save:
                        torch.save(self.model.state_dict(), f"models/pacman_model_V{version}-{epochsNum}.pth")
                        print("Model saved !")
                else:
                    self.patienceCount += 1
                    if self.patienceCount >= self.patienceLimit:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                if abs(lossPrev - trainLossAvg) < precision:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            lossPrev = lossAvg

        if save and precisionBest is not True:
            torch.save(self.model.state_dict(), f"models/pacman_model_V{version}-{.pth")
            print("Model saved !")

        print("Finished training your network model...")


if __name__ == "__main__":
    import json
    # TODO tweaking:
    #   - Data:
    #       - View Distance
    #       - Normalise positions
    #   - Neural Network:
    #       - Number of Layers
    #       - Layer Size
    #       - Normalisation
    #       - Activation
    #       - Dropout Rate
    #   - Training:
    #       - Criterion
    #       - Optimizer
    #       - Learning Rate
    #       - Scheduler
    #       - Batchsize
    #       - Normalisation
    #       - Epoch
    path = "datasets/pacman_dataset.pkl"

    # Data
    doNormalPos = True
    viewDistance = N

    dataset = PacmanDataset(
        path,
        doNormalPos=doNormalPos,
        viewDistance=viewDistance
    )

    # Neural Network
    inputSize = TENSOR_SIZE
    outputSize = 5
    layersNum = 5
    layer1Size = TENSOR_SIZE * 3
    layerSizeFunName = "two_thirds"
    layer_size_fun = get_layer_size_fun(layerSizeFunName)
    doNormal = True
    actionName = "GELU"
    action = get_action(actionName)
    doDropout = True
    dropoutRate = 0.3

    model = PacmanNetwork(
        inputSize,
        outputSize,
        layersNum=layersNum,
        layer1Size=layer1Size,
        layer_size_fun=layer_size_fun,
        doNormal=doNormal,
        action=action,
        doDropout=doDropout,
        dropoutRate=dropoutRate
    ).to(DEVICE)

    dict = {
        'dataset': {
            'doNormalPos': doNormalPos,
            'viewDistance': viewDistance,
        },
        'network': {
            'inputSize': inputSize,
            'outputSize': outputSize,
            'layersNum': layersNum,
            'layer1Size': layer1Size,
            'layerSizeFunName': layerSizeFunName,
            'doNormal': doNormal,
            'actionName': actionName,
            'doDropout': doDropout,
            'dropoutRate': dropoutRate,
        },
    }

    # Training model
    learningRate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01)
    doScheduler = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    validSplit = 0.2
    patienceLimit = 15

    pipeline = Pipeline(
        path,
        dataset,
        model,
        criterion,
        optimizer,
        doScheduler,
        scheduler,
        validSplit,
        patienceLimit=patienceLimit
    )

    # Training
    epochsNum = 500
    batchSize = 512
    doNormal = True
    normal = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    precision = 0
    precisionBest = False
    version = VERSION
    save = True

    pipeline.train(
        epochsNum,
        batchSize,
        doNormal,
        normal,
        precision=precision,
        precisionBest=precisionBest,
        version=version,
        save=save
    )

    if save:
        print("Writing model config...")
        with open(f"models/pacman_model_V{version}-{epochsNum}.json", "w") as file:
            json.dump(dict, file)
        print("Finished writing model config...")
