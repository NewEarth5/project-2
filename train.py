import os
import json
import random
from random import choice
import traceback

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architecture import PacmanNetwork
from data import PacmanDataset, get_tensor_size

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
SCALER = torch.amp.GradScaler(enabled=USE_CUDA)

PARAM = {
    'dataset': {
        'doNormalPos': [False],
        'viewDistance': [4],
    },
    'network': {
        'layersNum': [4],
        'layer1SizeMultiplier': [4],
        'layerSizeFun': ['linear'],
        'layerFun': ['Linear'],
        'doNormal': [True],
        'normalFun': ['BatchNorm1d'],
        'action': ['GELU'],
        'doDropout': [True],
        'dropoutRate': [0.2],
    },
    'training': {
        'learningRate': [0.0005],
        'criterion': ['CrossEntropyLoss'],
        'optimizer': ['AdamW'],
        'weightDecay': [0.0],
        'doScheduler': [False],
        'schedulerType': ['StepLR'],
        'validSplit': [0.2],
        'precision': [0.000001],
        'precisionBest': [2],
        'patienceLimit': [50],
        'epochsNum': [1000],
        'batchSize': [512],
    }
}


def get_layer_size_fun(name):
    layerSize = {
        "two_thirds": lambda pre, i, num: pre * 2 // 3,
        "half": lambda pre, i, num: pre // 2,
        "diamond": lambda pre, i, num: pre * 2 if i <= num // 2 else pre // 2,
        "linear": lambda pre, i, num: pre,
    }
    if name in layerSize:
        return layerSize[name]
    raise ValueError(f"Unknown layer size function: {name}")


def get_layer_fun(name):
    layer = {
        "Linear": nn.Linear,
    }
    if name in layer:
        return layer[name]
    raise ValueError(f"Unknown layer function: {name}")


def get_action(name):
    actions = {
        "ELU": nn.ELU,
        "Hardshrink": nn.Hardshrink,
        "Hardsigmoid": nn.Hardsigmoid,
        "Hardtanh": nn.Hardtanh,
        "Hardswish": nn.Hardswish,
        "LeakyReLU": nn.LeakyReLU,
        "LogSigmoid": nn.LogSigmoid,
        "PReLU": nn.PReLU,
        "ReLU": nn.ReLU,
        "ReLU6": nn.ReLU6,
        "RReLU": nn.RReLU,
        "SELU": nn.SELU,
        "CELU": nn.CELU,
        "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish,
        "Softplus": nn.Softplus,
        "Softshrink": nn.Softshrink,
        "Softsign": nn.Softsign,
        "Tanh": nn.Tanh,
        "Tanhshrink": nn.Tanhshrink,
        "GLU": nn.GLU,
        "Softmin": nn.Softmin,
        "Softmax": nn.Softmax,
        "Softmax2d": nn.Softmax2d,
        "LogSoftmax": nn.LogSoftmax,
    }
    if name in actions:
        return actions[name]
    raise ValueError(f"Unknown activation function: {name}")


def get_normal_fun(name):
    normal = {
        "BatchNorm1d": nn.BatchNorm1d,
        "GroupNorm": lambda val: nn.GroupNorm(val, val),
        "InstanceNorm1d": nn.InstanceNorm1d,
        "LayerNorm": nn.LayerNorm,
        "LocalResponseNorm": nn.LocalResponseNorm,
        "RMSNorm": nn.RMSNorm,
    }
    if name in normal:
        return normal[name]
    raise ValueError(f"Unknown normalisation function: {name}")


def create_criterion(name):
    criterion = {
        "L1Loss": nn.L1Loss,
        "MSELoss": nn.MSELoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "CTCLoss": nn.CTCLoss,
        "NLLLoss": nn.NLLLoss,
        "PoissonNLLLoss": nn.PoissonNLLLoss,
        "GaussianNLLLoss": nn.GaussianNLLLoss,
        "KLDivLoss": nn.KLDivLoss,
        "BCELoss": nn.BCELoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "MarginRankingLoss": nn.MarginRankingLoss,
        "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
        "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
        "HuberLoss": nn.HuberLoss,
        "SmoothL1Loss": nn.SmoothL1Loss,
        "SoftMarginLoss": nn.SoftMarginLoss,
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
        "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        "MultiMarginLoss": nn.MultiMarginLoss,
        "TripletMarginLoss": nn.TripletMarginLoss,
        "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    }
    if name in criterion:
        return criterion[name]
    raise ValueError(f"Unknown criterion: {name}")


def create_optimiser(name, parameters, learningRate, decay):
    """Factory function to create optimizer - creates on demand"""
    if name == "Adadelta":
        return optim.Adadelta(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "Adafactor":
        return optim.Adafactor(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "Adagrad":
        return optim.Adagrad(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "Adam":
        return optim.Adam(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "AdamW":
        return optim.AdamW(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "SparseAdam":
        return optim.SparseAdam(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "Adamax":
        return optim.Adamax(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "ASGD":
        return optim.ASGD(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "LBFGS":
        return optim.LBFGS(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "NAdam":
        return optim.NAdam(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "RAdam":
        return optim.RAdam(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "RMSprop":
        return optim.RMSprop(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "Rprop":
        return optim.Rprop(
            parameters,
            lr=learningRate,
            weight_decay=decay
        )
    if name == "SGD":
        return optim.SGD(
            parameters,
            lr=learningRate,
            weight_decay=decay,
            momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(name, optimizer, doScheduler):
    """Factory function to create scheduler - creates on demand"""
    if not doScheduler:
        return None

    if name == "LRScheduler":
        return optim.lr_scheduler.LRScheduler(optimizer)
    if name == "MultiplicativeLR":
        return optim.lr_scheduler.MultiplicativeLR(optimizer),
    if name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    if name == "ConstantLR":
        return optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=0.5
        )
    if name == "LinearLR":
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5
        )
    if name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.9
        )
    if name == "PolynomialLR":
        return optim.lr_scheduler.PolynomialLR(
            optimizer,
            power=3
        )
    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50
        )
    if name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    if name == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50
        )
    raise ValueError(f"Unknown scheduler: {name}")


class Pipeline(nn.Module):
    def __init__(
        self,
        path,
        dataset,
        model,
        criterion,
        optimizer,
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

        self.criterion = criterion()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clipping = torch.nn.utils.clip_grad_norm_

        self.bestLoss = float('inf')
        self.bestAcc = 0
        self.patienceCount = 0
        self.patienceLimit = patienceLimit

    def train(
        self,
        epochsNum,
        batchSize,
        precision=0,
        precisionBest=0,
        save=True,
        silent=False
    ):
        if not silent:
            print("Beginning of the training of your network...")
            print(f"Device used: {next(self.model.parameters()).device}")

        lossPrev = float('inf')
        accPrev = 0

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

        if not silent:
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
                self.clipping(self.model.parameters(), max_norm=1.0)
                SCALER.unscale_(self.optimizer)
                SCALER.step(self.optimizer)
                SCALER.update()

                trainLossTot += batchLoss.item()
                _, predicted = batchOutputs.max(1)
                trainTot += batchActions.size(0)
                trainCor += predicted.eq(batchActions).sum().item()

            trainLossAvg = trainLossTot / len(trainLoader)
            trainAcc = 100. * trainCor / trainTot

            if (epoch + 1) % 10 == 0:
                if not silent:
                    print(f"Epoch {epoch + 1}/{epochsNum}")
                    print(f"  Train Loss: {trainLossAvg:.4f}", end="")
                    print(f", Train Acc: {trainAcc:.2f}%")

            if len(validLoader) != 0:
                self.model.eval()
                validLossTot = 0.0
                validCor = 0
                validTot = 0

                with torch.no_grad():
                    for batchInputs, batchActions in validLoader:
                        batchInputs = batchInputs.to(
                            DEVICE,
                            non_blocking=USE_CUDA
                        )
                        batchActions = batchActions.to(
                            DEVICE,
                            non_blocking=USE_CUDA
                        )

                        with torch.amp.autocast(
                            device_type="cuda",
                            enabled=USE_CUDA
                        ):
                            batchOutputs = self.model(batchInputs)
                            batchLoss = self.criterion(
                                batchOutputs,
                                batchActions
                            )

                        validLossTot += batchLoss.item()
                        _, predicted = batchOutputs.max(1)
                        validTot += batchActions.size(0)
                        validCor += predicted.eq(batchActions).sum().item()

                validLossAvg = validLossTot / len(validLoader)
                validAcc = 100. * validCor / validTot
                lossAvg = validLossAvg
                acc = validAcc

                if (epoch + 1) % 10 == 0:
                    if not silent:
                        print(f"  Valid Loss: {validLossAvg:.4f}", end="")
                        print(f", Valid Acc: {validAcc:.2f}%")
            else:
                lossAvg = trainLossAvg
                acc = trainAcc

            if self.scheduler is not None:
                try:
                    self.scheduler.step(metrics=lossAvg)
                except Exception:
                    self.scheduler.step()

            if precisionBest != 0:
                if ((lossAvg < self.bestLoss and precisionBest == 1)
                   or (acc > self.bestAcc and precisionBest == 2)):
                    self.bestLoss = lossAvg
                    self.bestAcc = acc
                    self.patienceCount = 0

                    if not silent:
                        print("Model improved...", end="")

                    if save:
                        torch.save(self.model.state_dict(), self.path)

                        if not silent:
                            print(" saved !", end="")

                    if not silent:
                        print()
                else:
                    self.patienceCount += 1
                    if self.patienceCount >= self.patienceLimit:
                        if not silent:
                            print(f"Early stopping at epoch {epoch + 1}")

                        break
            else:
                if abs(lossPrev - trainLossAvg) < precision:
                    if not silent:
                        print(f"Early stopping at epoch {epoch + 1}")

                    break

            lossPrev = lossAvg
            accPrev = acc

        if save and precisionBest == 0:
            torch.save(self.model.state_dict(), self.path)

            if not silent:
                print("Model saved !")

        if not silent:
            print("Finished training your network model...")

        if precisionBest == 0:
            return [lossPrev, accPrev]
        else:
            return [self.bestLoss, self.bestAcc]


def search_training(folderPath, numTrials, save=True, silent=False):
    results = []
    failed = []
    files = os.listdir(folderPath)
    folders = np.array(
        [f for f in files if os.path.isdir(os.path.join(folderPath, f))],
        dtype=int
    )
    if folders.size == 0:
        folder = 1
    else:
        folder = max(folders) + 1
    if save:
        os.makedirs(os.path.join(folderPath, str(folder)))

    for trial in range(numTrials):
        if silent:
            print(f"Trial [{trial + 1}/{numTrials}]", end="")
        else:
            print()
            print(f"{'='*60}")
            print(f"Trial {trial + 1}/{numTrials}")
            print(f"{'='*60}")

        viewDistance = choice(PARAM['dataset']['viewDistance'])
        inputSize = get_tensor_size(viewDistance)
        layer1SizeMult = choice(PARAM['network']['layer1SizeMultiplier']),

        config = config = {
            'dataset': {
                'doNormalPos': choice(PARAM['dataset']['doNormalPos']),
                'viewDistance': viewDistance,
            },
            'network': {
                'inputSize': inputSize,
                'outputSize': 5,
                'layersNum': choice(PARAM['network']['layersNum']),
                'layer1Size': int(inputSize * layer1SizeMult[0]),
                'layer1SizeMultiplier': layer1SizeMult[0],
                'layerSizeFun': choice(PARAM['network']['layerSizeFun']),
                'layerFun': choice(PARAM['network']['layerFun']),
                'doNormal': choice(PARAM['network']['doNormal']),
                'normalFun': choice(PARAM['network']['normalFun']),
                'action': choice(PARAM['network']['action']),
                'doDropout': choice(PARAM['network']['doDropout']),
                'dropoutRate': choice(PARAM['network']['dropoutRate']),
            },
            'training': {
                'learningRate': choice(PARAM['training']['learningRate']),
                'criterion': choice(PARAM['training']['criterion']),
                'optimizer': choice(PARAM['training']['optimizer']),
                'weightDecay': choice(PARAM['training']['weightDecay']),
                'doScheduler': choice(PARAM['training']['doScheduler']),
                'schedulerType': choice(PARAM['training']['schedulerType']),
                'validSplit': choice(PARAM['training']['validSplit']),
                'precision': choice(PARAM['training']['precision']),
                'precisionBest': choice(PARAM['training']['precisionBest']),
                'patienceLimit': choice(PARAM['training']['patienceLimit']),
                'epochsNum': choice(PARAM['training']['epochsNum']),
                'batchSize': choice(PARAM['training']['batchSize']),
            },
            'trial_id': trial,
        }

        if not silent:
            print(f"Configuration: {json.dumps(config, indent=2)}")

        try:
            dataset = PacmanDataset(
                os.path.join("datasets", "pacman_dataset.pkl"),
                config['dataset']['doNormalPos'],
                config['dataset']['viewDistance'],
            )

            model = PacmanNetwork(
                config['network']['inputSize'],
                config['network']['outputSize'],
                config['network']['layersNum'],
                config['network']['layer1Size'],
                get_layer_size_fun(config['network']['layerSizeFun']),
                get_layer_fun(config['network']['layerFun']),
                config['network']['doNormal'],
                get_normal_fun(config['network']['normalFun']),
                get_action(config['network']['action']),
                config['network']['doDropout'],
                config['network']['dropoutRate'],
            ).to(DEVICE)

            criterion = create_criterion(config['training']['criterion'])
            optimizer = create_optimiser(
                config['training']['optimizer'],
                model.parameters(),
                config['training']['learningRate'],
                config['training']['weightDecay']
            )
            scheduler = create_scheduler(
                config['training']['schedulerType'],
                optimizer,
                config['training']['doScheduler']
            )

            pipeline = Pipeline(
                os.path.join(
                    folderPath,
                    str(folder),
                    f"pacman_model_V{trial + 1}.pth"
                ),
                dataset,
                model,
                criterion,
                optimizer,
                scheduler,
                config['training']['validSplit'],
                config['training']['patienceLimit']
            )

            bestLoss, bestAcc = pipeline.train(
                config['training']['epochsNum'],
                config['training']['batchSize'],
                config['training']['precision'],
                config['training']['precisionBest'],
                save=save,
                silent=silent
            )

            result = {
                **config,
                'performance': {
                    'loss': float(bestLoss),
                    'accuracy': float(bestAcc),
                },
                'status': 'success'
            }
            results.append(result)

            if silent:
                print(f" - success: Loss: {bestLoss:.4f}, Acc: {bestAcc:.2f}%")
            else:
                print(f"SUCCESS - Loss: {bestLoss:.4f}, Acc: {bestAcc:.2f}%")

            if save:
                with open(
                    os.path.join(
                        folderPath,
                        str(folder),
                        f"pacman_model_V{trial + 1}.json"
                    ),
                    "w"
                ) as f:
                    json.dump(result, f, indent=2)

        except Exception as e:
            error = {
                'trial_id': trial,
                'config': config,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'status': 'failed',
            }
            failed.append(error)

            if silent:
                print(f" - fail: {str(e)}")
            else:
                print(f"FAILED - Trial {trial}")
                print(f"Error: {str(e)}")

        finally:
            if save:
                with open(
                    os.path.join(
                        folderPath,
                        f"training_results_V{folder}.json"
                    ),
                    "w"
                ) as f:
                    json.dump({
                        'results': results,
                        'failed': failed,
                        'completed_trials': trial + 1,
                        'total_trials': numTrials
                    }, f, indent=2)

            if USE_CUDA:
                torch.cuda.empty_cache()

    if silent:
        print(f"Finished: {len(results)}/{numTrials} succeeded.")
    else:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"  Successful trials: {len(results)}/{numTrials}")
        print(f"  Failed trials: {len(failed)}/{numTrials}")
        print(f"{'='*60}")

    return results, failed


def get_best(folder, index=0, byAcc=True):
    with open(f"models/training_results_V{folder}.json", "r") as file:
        dictJson = json.load(file)

    results = dictJson['results']

    if results:
        loss = np.array([res['performance']['loss'] for res in results])
        lossInd = np.argsort(loss)
        acc = np.array([res['performance']['accuracy'] for res in results])
        accInd = np.argsort(-acc)

        if byAcc:
            return int(results[accInd[index]]['trial_id']) + 1
        else:
            return int(results[lossInd[index]]['trial_id']) + 1


if __name__ == "__main__":
    torch.manual_seed(420)
    random.seed(420)
    silent = True

    results, failed = search_training("models", 1000, silent=True)

    if results:
        bestAcc = max(
            results,
            key=lambda x: x['performance']['accuracy']
        )
        bestLoss = min(
            results,
            key=lambda x: x['performance']['loss']
        )
        if silent:
            print("Best: ", end="")
            print(f"Loss by model {bestLoss['trial_id']} ", end="")
            print(f"[{bestLoss['performance']['loss']:.2f}]", end="")
            print(f"Acc by model {bestAcc['trial_id']} ", end="")
            print(f"[{bestAcc['performance']['accuracy']:.2f}], ", end="")
            print()
        else:
            print(f"\nBest by Accuracy: {bestAcc['performance']['accuracy']:.2f}%")
            print(f"Config: {json.dumps(bestAcc, indent=2)}")

            print(f"\nBest by Loss: {bestLoss['performance']['loss']:.4f}")
            print(f"Config: {json.dumps(bestLoss, indent=2)}")
