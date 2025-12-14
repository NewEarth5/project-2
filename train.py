import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architecture import PacmanNetwork
from data import PacmanDataset

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


def get_layer_fun(name):
    if name == "Linear":
        return nn.Linear
    raise ValueError()


def get_normal_fun(name):
    if name == "BatchNorm1d":
        return nn.BatchNorm1d
    raise ValueError()


def get_tensor_size(viewDistance):
    return 2 + 2 + (2 * viewDistance * (viewDistance + 1)) + (2 * viewDistance * (viewDistance + 1)) + 4


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
            # num_workers=2,
            # persistent_workers=True
        )
        validLoader = DataLoader(
            self.datasetValid,
            batch_size=batchSize,
            shuffle=False,
            pin_memory=USE_CUDA,
            # num_workers=2,
            # persistent_workers=True
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
            torch.save(self.model.state_dict(), f"models/pacman_model_V{version}-{epochsNum}.pth")
            print("Model saved !")

        print("Finished training your network model...")


if __name__ == "__main__":
    import json

    path = "datasets/pacman_dataset.pkl"
    testNum = 10

    try:
        # Dataset
        allDoNormalPos = [True, False]
        for doNormalPos in allDoNormalPos:
            for viewDistance in range(1, 10 + 1):
                dataset = PacmanDataset(
                    path,
                    doNormalPos=doNormalPos,
                    viewDistance=viewDistance
                )

                # Neural Network
                inputSize = get_tensor_size(viewDistance)
                outputSize = 5
                for layersNum in range(1, 10 + 1):
                    for layer1Fact in range(100, 1000 + 1):
                        layer1Size = inputSize * layer1Fact // 100
                        allLayerSizeFunName = ["two_thirds"]
                        for layerSizeFunName in allLayerSizeFunName:
                            layer_size_fun = get_layer_size_fun(layerSizeFunName)
                            allLayerFunName = ["Linear"]
                            for layerFunName in allLayerFunName:
                                layer_fun = get_layer_fun(layerFunName)
                                allDoNormalNet = [True, False]
                                for doNormalNet in allDoNormalNet:
                                    if doNormalNet:
                                        allNormalFunName = ["BatchNorm1d"]
                                    else:
                                        allNormalFunName = ["BatchNorm1d"]
                                    for normalFunName in allNormalFunName:
                                        normal_fun = get_normal_fun(normalFunName)
                                        allActionName = ["GELU"]
                                        for actionName in allActionName:
                                            action = get_action(actionName)
                                            allDoDropout = [True, False]
                                            for doDropout in allDoDropout:
                                                if doDropout:
                                                    rangeDropoutRate = 99
                                                else:
                                                    rangeDropoutRate = 0
                                                for dropoutRateFact in range(rangeDropoutRate + 1):
                                                    dropoutRate = dropoutRateFact / 100
                                                    model = PacmanNetwork(
                                                        inputSize,
                                                        outputSize,
                                                        layersNum,
                                                        layer1Size,
                                                        layer_size_fun,
                                                        layer_fun,
                                                        doNormalNet,
                                                        normal_fun,
                                                        action,
                                                        doDropout,
                                                        dropoutRate
                                                    ).to(DEVICE)

                                                    # Training model
                                                    for learningRateFact in range(1, 100 + 1):
                                                        learningRate = learningRateFact / 10000
                                                        allCriterion = [nn.CrossEntropyLoss()]
                                                        for criterion in allCriterion:
                                                            allOptimizer = []
                                                            allOptimizer.extend([optim.AdamW(model.parameters(), lr=learningRate, weight_decay=decay / 1000)] for decay in range(0, 100 + 1, 5))
                                                            for optimizer in allOptimizer:
                                                                allDoScheduler = [True, False]
                                                                for doScheduler in allDoScheduler:
                                                                    if doScheduler:
                                                                        allScheduler = []
                                                                        allScheduler.extend([optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor / 10, patience=patience) for factor in range(10) for patience in range(10)])
                                                                    else:
                                                                        allScheduler = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)]
                                                                    for scheduler in allScheduler:
                                                                        for validSplitFact in range(9 + 1):
                                                                            validSplit = validSplitFact / 10
                                                                            allPrecisionBest = [True, False]
                                                                            for precisionBest in allPrecisionBest:
                                                                                if precisionBest:
                                                                                    rangePatienceLimit = 100
                                                                                else:
                                                                                    rangePatienceLimit = 0
                                                                                for patienceLimit in range(0, rangePatienceLimit + 1, 10):
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
                                                                                    for epochsNum in range(0, 1000 + 1, 5):
                                                                                        if epochsNum == 0:
                                                                                            continue
                                                                                        for batchSizeFact in range(15 + 1):
                                                                                            batchSize = 2**batchSizeFact
                                                                                            allDoNormalTrain = [True, False]
                                                                                            for doNormalTrain in allDoNormalTrain:
                                                                                                if doNormalTrain:
                                                                                                    allNormal = [torch.nn.utils.clip_grad_norm_(model.parameters(), i / 10) for i in range(100 + 1)]
                                                                                                else:
                                                                                                    allNormal = [torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)]
                                                                                                for normal in allNormal:
                                                                                                    for precisionFact in range(10):
                                                                                                        precision = precisionFact / 1000000
                                                                                                        for i in range(testNum + 1):
                                                                                                            version = VERSION
                                                                                                            save = True

                                                                                                            pipeline.train(
                                                                                                                epochsNum,
                                                                                                                batchSize,
                                                                                                                doNormalTrain,
                                                                                                                normal,
                                                                                                                precision=precision,
                                                                                                                precisionBest=precisionBest,
                                                                                                                version=version,
                                                                                                                save=save
                                                                                                            )

                                                                                                            if save:
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
                                                                                                                        'layerFunName': layerFunName,
                                                                                                                        'doNormal': doNormalNet,
                                                                                                                        'normalFunName': normalFunName,
                                                                                                                        'actionName': actionName,
                                                                                                                        'doDropout': doDropout,
                                                                                                                        'dropoutRate': dropoutRate,
                                                                                                                    },
                                                                                                                }

                                                                                                                print("Writing model config...")
                                                                                                                with open(f"models/pacman_model_V{version}-{epochsNum}.json", "w") as file:
                                                                                                                    json.dump(dict, file)
                                                                                                                print("Finished writing model config...")
    except:
        print()
