import torch
from torch import nn
from tqdm import tqdm
import ML


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type_as(y) == y
    return float(cmp.clone().detach().sum())


def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
    metric = ML.Accumulator(2)
    with torch.no_grad():
        pbar = tqdm(data_iter, total=len(data_iter), desc="accuracy")
        for X, y in pbar:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
            pbar.set_postfix(accuracy=metric[0] / metric[1])
        pbar.close()
    return metric[0] / metric[1]
