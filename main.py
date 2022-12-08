import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
root_dataset_dir = "./Dataset/MNIST"
batch_size = 32
# 读取测试数据，train=True读取训练数据；train=False读取测试数据
train_dataset = datasets.MNIST(root=root_dataset_dir, train=True, transform=data_tf)
test_dataset = datasets.MNIST(root=root_dataset_dir, train=False, transform=data_tf)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

examples = enumerate(test_loader) #img&label
batch_idx, (imgs, labels) = next(examples) #读取数据,batch_idx从0开始

print(labels) #读取标签数据
print(labels.shape) #torch.Size([32])，因为batch_size为32