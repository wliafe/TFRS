from PIL import Image


def image_classification_predict(net, labels, image, transform, device):
    net.to(device)
    net.eval()
    image_tensor = transform(Image.open(image).convert('RGB')).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return labels[net(image_tensor).argmax(dim=1)[0]]
