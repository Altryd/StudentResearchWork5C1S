import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Resize((260, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=r'CEDAR\signatures', transform=transform)
loader = DataLoader(dataset=dataset, shuffle=True)
train_ratio = 0.8
train_size = int(len(dataset)*train_ratio)
test_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, test_size])
train_dataset = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_dataset = DataLoader(dataset=val_data, batch_size=32, shuffle=True)
models_path = ["densenet121_handwritten_2fc_trained.pth", "efficientnet_b4_2fc_trained.pth",
               "resnet101_2fc_trained.pth", "vgg16_2fc_trained.pth", ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_path in models_path:
    model = torch.load(model_path, weights_only=False)
    model.eval()
    total_correct = 0
    with torch.no_grad():
        predictions = []
        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            predictions.append(predicted)
            total_correct += (predicted == labels).sum().item()
        print(f"predictions for first batch: {predictions[0]}...")

    accuracy = total_correct / len(val_data)
    print(f'Test Accuracy for model {model_path}: {(100 * accuracy):.2f}%')