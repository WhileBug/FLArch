import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics


def test_img(net_g, datatest, bs):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=bs)
    l = len(data_loader)
    iter_num = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        #print("y_pred",y_pred)
        #print("target",target)
        #correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        #print(metrics.accuracy_score(target.tolist(), y_pred.tolist()))
        correct +=  metrics.accuracy_score(target.tolist(), y_pred.tolist())
        iter_num += 1

    test_loss /= len(data_loader.dataset)
    #accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy = 100.00 * correct / iter_num
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_img_2(net_g, datatest, bs=1024):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    accuracy_val = 0
    data_loader = DataLoader(datatest, batch_size=bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):

        data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        predict_result = np.argmax(log_probs.tolist(), axis=0)
        accuracy_val += metrics.accuracy_score(target.tolist(), predict_result)
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy_val /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy_val, test_loss