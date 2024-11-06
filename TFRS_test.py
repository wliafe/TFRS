import torch
import ML
from TFRS import net, model_path, device, labels, test_iter

if __name__ == '__main__':
    net.load_state_dict(torch.load(model_path, map_location=device))
    ML.image_classification_test(net, labels, test_iter, device)
    print(ML.evaluate_accuracy(net, test_iter, device))
