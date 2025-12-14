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
    if name == "half":
        return lambda prev, i, num: prev // 2
    if name == "diamond":
        return lambda prev, i, num: prev * 2 if i <= num // 2 else prev // 2
    if name == "linear":
        return lambda prev, i, num: prev
    raise ValueError()


def get_layer_fun(name):
    if name == "Linear":
        return nn.Linear
    raise ValueError()


def get_action(name):
    if name == "ELU":
        return nn.ELU()
    if name == "Hardshrink":
        return nn.Hardshrink()
    if name == "Hardsigmoid":
        return nn.Hardsigmoid()
    if name == "Hardtanh":
        return nn.Hardtanh()
    if name == "Hardswish":
        return nn.Hardswish()
    if name == "LeakyReLU":
        return nn.LeakyReLU()
    if name == "LogSigmoid":
        return nn.LogSigmoid()
    if name == "PReLU":
        return nn.PReLU()
    if name == "ReLU":
        return nn.ReLU()
    if name == "ReLU6":
        return nn.ReLU6()
    if name == "RReLU":
        return nn.RReLU()
    if name == "SELU":
        return nn.SELU()
    if name == "CELU":
        return nn.CELU()
    if name == "GELU":
        return nn.GELU()
    if name == "Sigmoid":
        return nn.Sigmoid()
    if name == "SiLU":
        return nn.SiLU()
    if name == "Mish":
        return nn.Mish()
    if name == "Sofplus":
        return nn.Softplus()
    if name == "Softshrink":
        return nn.Softshrink()
    if name == "Softsign":
        return nn.Softsign()
    if name == "Tanh":
        return nn.Tanh()
    if name == "Tanhshrink":
        return nn.Tanhshrink()
    if name == "GLU":
        return nn.GLU()
    if name == "Softmin":
        return nn.Softmin()
    if name == "Softmax":
        return nn.Softmax()
    if name == "Softmax2d":
        return nn.Softmax2d()
    if name == "LogSoftmax":
        return nn.LogSoftmax()
    raise ValueError()


def get_normal_fun(name):
    if name == "BatchNorm1d":
        return nn.BatchNorm1d
    if name == "BatchNorm2d":
        return nn.BatchNorm2d
    if name == "BatchNorm3d":
        return nn.BatchNorm3d
    if name == "GroupNorm":
        return lambda val: nn.GroupNorm(5, val)
    if name == "SyncBatchNorm":
        return nn.SyncBatchNorm
    if name == "InstanceNorm1d":
        return nn.InstanceNorm1d
    if name == "InstanceNorm2d":
        return nn.InstanceNorm2d
    if name == "InstanceNorm3d":
        return nn.InstanceNorm3d
    if name == "LayerNorm":
        return nn.LayerNorm
    if name == "LocalResponseNorm":
        return nn.LocalResponseNorm
    if name == "RMSNorm":
        return nn.RMSNorm
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
            torch.save(self.model.state_dict(), f"models/pacman_model_V{version}.pth")
            print("Model saved !")

        print("Finished training your network model...")


if __name__ == "__main__":
    import json

    path = "datasets/pacman_dataset.pkl"
    testNum = 10

    try:
        # Dataset
        allDoNormalPos = [True, False]
        for d1 in range(len(allDoNormalPos)):
            doNormalPos = allDoNormalPos[d1]
            for d2 in range(1, 10 + 1):
                viewDistance = d2
                dataset = PacmanDataset(
                    path,
                    doNormalPos=doNormalPos,
                    viewDistance=viewDistance
                )

                # Neural Network
                inputSize = get_tensor_size(viewDistance)
                outputSize = 5
                for n1 in range(1, 10 + 1):
                    layersNum = n1
                    for n2 in range(100, 1000 + 1):
                        layer1Size = inputSize * n2 // 100
                        allLayerSizeFunName = [
                            "two_thirds",
                            "half",
                            "diamond",
                            "linear"
                        ]
                        for n3 in range(len(allLayerSizeFunName)):
                            layerSizeFunName = allLayerSizeFunName[n3]
                            layer_size_fun = get_layer_size_fun(layerSizeFunName)
                            allLayerFunName = [
                                "Linear",
                            ]
                            for n4 in range(len(allLayerFunName)):
                                layerFunName = allLayerFunName[n4]
                                layer_fun = get_layer_fun(layerFunName)
                                allDoNormalNet = [True, False]
                                for n5 in range(len(allDoNormalNet)):
                                    doNormalNet = allDoNormalNet[n5]
                                    if doNormalNet:
                                        allNormalFunName = [
                                            "BatchNorm1d",
                                            "BatchNorm2d",
                                            "BatchNorm3d",
                                            "GroupNorm",
                                            "SyncBatchNorm",
                                            "InstanceNorm1d",
                                            "InstanceNorm2d",
                                            "InstanceNorm3d",
                                            "LayerNorm",
                                            "LocalResponseNorm",
                                            "RMSNorm",
                                        ]
                                    else:
                                        allNormalFunName = ["BatchNorm1d"]
                                    for n6 in range(len(allNormalFunName)):
                                        normalFunName = allNormalFunName[n6]
                                        normal_fun = get_normal_fun(normalFunName)
                                        allActionName = [
                                            "ELU",
                                            "Hardshrink",
                                            "Hardsigmoid",
                                            "Hardtanh",
                                            "Hardswish",
                                            "LeakyReLU",
                                            "LogSigmoid",
                                            "PReLU",
                                            "ReLU",
                                            "ReLU6",
                                            "RReLU",
                                            "SELU",
                                            "CELU",
                                            "GELU",
                                            "Sigmoid",
                                            "SiLU",
                                            "Mish",
                                            "Sofplus",
                                            "Softshrink",
                                            "Softsign",
                                            "Tanh",
                                            "Tanhshrink",
                                            "GLU",
                                            "Softmin",
                                            "Softmax",
                                            "Softmax2d",
                                            "LogSoftmax",
                                        ]
                                        for n7 in range(len(allActionName)):
                                            actionName = allActionName[n7]
                                            action = get_action(actionName)
                                            allDoDropout = [True, False]
                                            for n8 in range(len(allDoDropout)):
                                                doDropout = allDoDropout[n8]
                                                if doDropout:
                                                    rangeDropoutRate = 99
                                                else:
                                                    rangeDropoutRate = 0
                                                for n9 in range(rangeDropoutRate + 1):
                                                    dropoutRate = n9 / 100
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
                                                    for p1 in range(1, 100 + 1):
                                                        learningRate = p1 / 10000
                                                        allCriterion = [
                                                            nn.L1Loss(),
                                                            nn.MSELoss(),
                                                            nn.CrossEntropyLoss(),
                                                            nn.CTCLoss(),
                                                            nn.NLLLoss(),
                                                            nn.PoissonNLLLoss(),
                                                            nn.GaussianNLLLoss(),
                                                            nn.KLDivLoss(),
                                                            nn.BCELoss(),
                                                            nn.BCEWithLogitsLoss(),
                                                            nn.MarginRankingLoss(),
                                                            nn.HingeEmbeddingLoss(),
                                                            nn.MultiLabelMarginLoss(),
                                                            nn.HuberLoss(),
                                                            nn.SmoothL1Loss(),
                                                            nn.SoftMarginLoss(),
                                                            nn.MultiLabelSoftMarginLoss(),
                                                            nn.CosineEmbeddingLoss(),
                                                            nn.MultiMarginLoss(),
                                                            nn.TripletMarginLoss(),
                                                            nn.TripletMarginWithDistanceLoss(),
                                                        ]
                                                        for p2 in range(len(allCriterion)):
                                                            criterion = allCriterion[p2]
                                                            allOptimizer = []
                                                            allOptimizer.extend([optim.Adadelta(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.Adafactor(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.Adagrad(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.Adam(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.AdamW(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.SparseAdam(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.Adamax(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.ASGD(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.LBFGS(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.NAdam(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.RAdam(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.RMSprop(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend([optim.Rprop(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)])
                                                            allOptimizer.extend(([optim.SGD(model.parameters(), lr=learningRate, weight_decay=decay / 100) for decay in range(10 + 1)]))
                                                            for p3 in range(len(allOptimizer)):
                                                                optimizer = allOptimizer[p3]
                                                                allDoScheduler = [True, False]
                                                                for p4 in range(len(allDoScheduler)):
                                                                    doScheduler = allDoScheduler[p4]
                                                                    if doScheduler:
                                                                        allScheduler = []
                                                                        allScheduler.extend([optim.lr_scheduler.LRScheduler(optimizer)])
                                                                        allScheduler.extend([optim.lr_scheduler.MultiplicativeLR(optimizer)])
                                                                        allScheduler.extend([optim.lr_scheduler.StepLR(optimizer, step) for step in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.ConstantLR(optimizer, factor=factor / 10) for factor in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.LinearLR(optimizer, start_factor=factor / 10) for factor in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.ExponentialLR(optimizer, gamma / 10) for gamma in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.PolynomialLR(optimizer, power=power) for power in range(5)])
                                                                        allScheduler.extend([optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max) for T_max in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor / 10, patience=patience) for factor in range(10) for patience in range(10)])
                                                                        allScheduler.extend([optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0) for T_0 in range(10)])
                                                                    else:
                                                                        allScheduler = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)]
                                                                    for p5 in range(len(allScheduler)):
                                                                        scheduler = allScheduler[p5]
                                                                        for p6 in range(9 + 1):
                                                                            validSplit = p6 / 10
                                                                            allPrecisionBest = [True, False]
                                                                            for p7 in range(len(allPrecisionBest)):
                                                                                precisionBest = allPrecisionBest[p7]
                                                                                if precisionBest:
                                                                                    rangePatienceLimit = 100
                                                                                else:
                                                                                    rangePatienceLimit = 0
                                                                                for p8 in range(0, rangePatienceLimit + 1, 10):
                                                                                    patienceLimit = p8
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
                                                                                    for t1 in range(0, 1000 + 1, 5):
                                                                                        if t1 == 0:
                                                                                            continue
                                                                                        epochsNum = t1
                                                                                        for t2 in range(15 + 1):
                                                                                            batchSize = 2**t2
                                                                                            for t3 in range(10):
                                                                                                precision = t3 / 1000000
                                                                                                for test in range(testNum + 1):
                                                                                                    version = f"{d1}.{d2}-{n1}.{n2}.{n3}.{n4}.{n5}.{n6}.{n7}.{n8}.{n9}-{p1}.{p2}.{p3}.{p4}.{p5}.{p6}.{p7}.{p8}-{t1}.{t2}.{t3}-{test}"
                                                                                                    save = True

                                                                                                    pipeline.train(
                                                                                                        epochsNum,
                                                                                                        batchSize,
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
                                                                                                        with open(f"models/pacman_model_V{version}.json", "w") as file:
                                                                                                            json.dump(dict, file)
                                                                                                        print("Finished writing model config...")
    except:
        print()
