import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import copy
from FLArch.Client import Client
import numpy as np
from FLArch.Server.AggregationAlgo import AggregationAlgo
from FLArch.Utlis.test import test_img

class Server:
    def __init__(
            self,
            dataset:datasets.VisionDataset,
            globalModel:nn.Module,
            clients:[Client],
            device:str='cuda'
    ):
        self.dataset = dataset
        self.globalModel = globalModel
        self.clients = clients
        self.device = device

    def globalModelInitTrain(self):
        self.globalModel.to(self.device)
        self.globalModel.train()

    def copyGlobalModelWeights(self):
        # copy weights
        global_weights = self.globalModel.state_dict()
        return global_weights

    def clientSelect(
            self
    )->[Client]:
        return self.clients

    def getLocalUpdates(
            self,
            args,
            epoch:int,
            dataset_train,
            dataset_test
    ):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        self.globalModel.train()
        participants = self.clientSelect()
        for participant in participants:
            localWeightUpdate, localLoss = participant.localUpdate(
                args=args,
                epoch=epoch,
                global_model=self.globalModel
            )
            local_weights.append(localWeightUpdate)
            local_losses.append(localLoss)
        # update global weights
        aggregationAlgo = AggregationAlgo()
        global_weights = aggregationAlgo.average_weights(local_weights)
        # update global weights
        self.globalModel.load_state_dict(global_weights)

        acc_train, loss_train = test_img(self.globalModel, dataset_train)
        acc_test, loss_test = test_img(self.globalModel, dataset_test)
        print("Global Round {:3d},Training accuracy: {:.2f}".format(epoch, acc_train))
        print("Global Testing accuracy: {:.2f}".format(acc_test))