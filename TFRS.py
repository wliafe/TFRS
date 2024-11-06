# 导入所需库
import torch
from torchvision import models, transforms
import ML

# 变量定义
train_path = 'data/train'
test_path = 'data/test'
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=0.5, std=0.5)
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=0.5, std=0.5)
])
model_path = 'model/TFRS_model.pth'
# 数据处理
labels = ML.load_image_labels_data(train_path)
net = models.resnet18(num_classes=len(labels))
train_iter = ML.load_image_classification_data(train_path, train_transform, 50)
test_iter = ML.load_image_classification_data(test_path, test_transform, 50)
device = ML.try_gpu()
