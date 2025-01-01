import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from utility import load_dataset, train_model

resnet101 = models.resnet101(pretrained=False, num_classes=2)
resnet101.name = "resnet101"
# num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов
# resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet101.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((260, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset, val_dataset, train_data, val_data = load_dataset(r"Samara",
                                                                transform=models.ResNet101_Weights.IMAGENET1K_V2.transforms())

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet101.to(device)

train_model(resnet101, optimizer, criterion, train_dataset, val_dataset, train_data, val_data, dataset_name="Samara")
