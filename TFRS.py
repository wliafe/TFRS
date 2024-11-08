# 导入所需库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from d2l import torch as d2l


# 定义类
class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 数据获取
def load_image_classification_data(path, transform, batch_size, num_workers=0, pin_memory=False):
    data = datasets.ImageFolder(path, transform=transform)
    iters = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                       drop_last=True)
    return iters


# 准确率评估
def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        pbar = tqdm(data_iter, total=len(data_iter), desc="accuracy")
        for X, y in pbar:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
            pbar.set_postfix(accuracy=metric[0] / metric[1])
        pbar.close()
    print(metric[0] / metric[1])
    return metric[0] / metric[1]


# 定义图像分类训练函数
def image_classification_train(net, train_iter, test_iter, num_epochs, lr, device):
    # 初始化网络权重
    train_loss, train_acc, test_acc = 0.0, 0.0, 0.0

    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 初始化训练损失，训练准确率，样本数
        metric = Accumulator(3)
        net.train()
        pbar = tqdm(train_iter, total=len(train_iter), desc=f'epoch {epoch + 1}')
        for X, y in pbar:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            lost = loss(y_hat, y)
            lost.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(lost * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            pbar.set_postfix(train_loss=train_loss)
        pbar.close()
        # 计算测试准确率
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f"loss {train_loss:.3f}, trains acc {train_acc:.3f}, tests acc {test_acc:.3f}")
    return train_loss, train_acc, test_acc


# 测试
def image_classification_test(net, labels, test_iter, device):
    net.to(device)
    net.eval()
    X, y = next(iter(test_iter))
    X, y = X.to(device), y.to(device)
    trues = [labels[i] for i in y]
    predicts = [labels[i] for i in net(X).argmax(dim=1)]
    print(trues[0:6])
    print(predicts[0:6])
    return trues, predicts


# 预测
def image_classification_predict(net, labels, image, transform, device):
    net.to(device)
    net.eval()
    with torch.no_grad():
        image_tensor = transform(Image.open(image).convert('RGB')).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        predict = labels[net(image_tensor).argmax(dim=1)[0]]
    return predict
