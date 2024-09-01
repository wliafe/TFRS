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
