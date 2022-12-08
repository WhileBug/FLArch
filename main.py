import dataclasses

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FLArch.Client.Client import Client
from FLArch.Server.Server import Server
from FLArch.Model.Model import CNNMnist

CLIENT_NUM = 10
EPOCH_NUM = 10

class FLArgs:
    local_bs = 10
    optimizer = "sgd"
    lr = 0.01
    local_ep=10
    verbose=True

def get_mnist_dataset():
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    root_dataset_dir = "./Dataset/MNIST"
    batch_size = 32
    # 读取测试数据，train=True读取训练数据；train=False读取测试数据
    train_dataset = datasets.MNIST(root=root_dataset_dir, train=True, transform=data_tf)
    test_dataset = datasets.MNIST(root=root_dataset_dir, train=False, transform=data_tf)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = get_mnist_dataset()

    cnnmnist = CNNMnist()
    fl_args = FLArgs()
    clients = []
    for i in range(CLIENT_NUM):
        fl_client = Client(
            clientDataset=train_dataset,
            datasetIndexList=list(range(1000))
        )
        clients.append(fl_client)
    fl_server = Server(
        dataset=train_dataset,
        globalModel=cnnmnist,
        clients=clients,
        device='cuda'
    )
    for epoch in range(EPOCH_NUM):
        fl_server.getLocalUpdates(
            args=fl_args,
            epoch=epoch
        )