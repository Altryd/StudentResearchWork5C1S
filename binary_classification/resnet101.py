import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from utility import load_dataset, train_model, load_dataset_with_train_test_transforms

resnet101 = models.resnet101(pretrained=False, num_classes=2)
resnet101.name = "resnet101"
# num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов
# resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet101.parameters(), lr=0.001, momentum=0.9)
train_transform_v34 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_v34 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform_v101 = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_v101 = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


(train_dataset, val_dataset, train_data,
 val_data) = load_dataset_with_train_test_transforms("../Samara_inverse",
                                                     train_transform=test_transform_v101,
                                                     test_transform=test_transform_v101, batch_size=4)
"""
(train_dataset, val_dataset, train_data,
 val_data) = load_dataset("Samara", transform=test_transform)
"""

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet101.to(device)

train_model(resnet101, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name="Samara_inverse", num_epochs=80)
