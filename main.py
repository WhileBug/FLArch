from torchvision import datasets, transforms
from FLArch.Client.Client import Client
from FLArch.Server.Server import Server
from FLArch.Model.Model import CNNMnist
import numpy as np
import torch

class FLConfig:
    num_sample_per_client = 1000
    num_clients = 10
    epoch_num = 100

class FLArgs:
    local_bs = 10
    optimizer = "sgd"
    lr = 0.01
    local_ep=10
    verbose=True

    num_classes=10 # how many classes are in current task
    num_channels=1

    gpu = 'cuda'

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

def mnist_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

if __name__ == '__main__':
    train_dataset, test_dataset = get_mnist_dataset()

    user_groups = mnist_iid(train_dataset, FLConfig.num_clients, FLConfig.num_sample_per_client)
    fl_args = FLArgs()
    cnnmnist = CNNMnist(fl_args)
    cnnmnist = cnnmnist.to(torch.device(fl_args.gpu))
    clients = []
    for i in range(FLConfig.num_clients):
        fl_client = Client(
            clientDataset=train_dataset,
            datasetIndexList=user_groups[i]
        )
        clients.append(fl_client)
    fl_server = Server(
        dataset=train_dataset,
        globalModel=cnnmnist,
        clients=clients,
        device=fl_args.gpu
    )
    for epoch in range(FLConfig.epoch_num):
        fl_server.getLocalUpdates(
            args=fl_args,
            epoch=epoch
        )