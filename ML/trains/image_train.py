import torch
from torch import nn
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
    # 定义训练过程图
    animator = ML.Animator(
        legend=['trains loss', 'trains acc', 'tests acc'], x_label='epoch', x_lim=[0, num_epochs], y_lim=[0, 1.1])
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 初始化训练损失，训练准确率，样本数
        metric = ML.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
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
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, [train_loss, train_acc, None])
        # 计算测试准确率
        test_acc = ML.evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, [None, None, test_acc])
    print(f"loss {train_loss:.3f}, trains acc {train_acc:.3f}, tests acc {test_acc:.3f} on {str(device)}")
    return train_loss, train_acc, test_acc
