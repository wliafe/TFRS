import torch
import ML
from TFRS import net, train_iter, test_iter, device, model_path

if __name__ == '__main__':
    lr, num_epochs = 0.01, 3
    train_loss, train_acc, test_acc = ML.image_classification_train(net, train_iter, test_iter, num_epochs, lr, device)
    torch.save(net.state_dict(), model_path)
