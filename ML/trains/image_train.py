import torch
from torch import nn
from tqdm import tqdm
import ML


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
        metric = ML.Accumulator(3)
        net.train()
        pbar = tqdm(enumerate(train_iter), total=len(train_iter), desc=f'epoch {epoch + 1}')
        for i, (X, y) in pbar:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            lost = loss(y_hat, y)
            lost.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(lost * X.shape[0], ML.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            pbar.set_postfix(train_loss=train_loss)
            pbar.update(1)
        # 计算测试准确率
        test_acc = ML.evaluate_accuracy(net, test_iter, device)
    print(f"loss {train_loss:.3f}, trains acc {train_acc:.3f}, tests acc {test_acc:.3f}")
    return train_loss, train_acc, test_acc
