
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def load_model(version='default'):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes (CIFAR-10)
    if version != 'default':
        model.load_state_dict(torch.load(f'models/{version}.pth'))
    model.eval()
    return model

def classify_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return int(predicted)
