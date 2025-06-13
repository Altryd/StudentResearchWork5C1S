import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from utility import load_dataset, train_model, load_dataset_with_train_test_transforms


DATASET_PATH = r"../Samara"
densenet121 = models.densenet121(pretrained=True)
num_classes = 2
densenet121.classifier = nn.Linear(densenet121.classifier.in_features, num_classes)
train_transform = transforms.Compose([
    transforms.Resize((232, 232), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),  # to 0.0 - 1.0
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((232, 232), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(densenet121.parameters(), lr=0.0005, momentum=0.2)
(train_dataset, val_dataset, train_data,
 val_data) = load_dataset_with_train_test_transforms(DATASET_PATH,
                                                     train_transform=train_transform,
                                                     test_transform=test_transform)
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 200
train_model(densenet121, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name=DATASET_PATH, num_epochs=num_epochs)
