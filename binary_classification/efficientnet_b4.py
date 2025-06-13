import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

from utility import load_dataset_with_train_test_transforms, train_model

eff_net = models.efficientnet_b0(pretrained=False)
print(eff_net)
efficientnet_b4 = models.efficientnet_b4(pretrained=False, num_classes=2)
# num_classes = 2  # Укажите количество классов
# efficientnet_b4.fc = nn.Linear(efficientnet_b4.fc.in_features, num_classes)
# num_ftrs = efficientnet_b4.classifier[1].in_features
# efficientnet_b4._fc = torch.nn.Linear(in_features=efficientnet_b4._fc.in_features, out_features=num_classes, bias=True)

# efficientnet_b4.classifier = nn.Sequential(
#            nn.Dropout(p=0.1, inplace=True),
#            nn.Linear(num_ftrs, num_classes),
#        )



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(efficientnet_b4.parameters(), lr=0.001, momentum=0.9)
train_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((380, 380)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

path_to_dataset = r"../CEDAR"
(train_dataset, val_dataset,
 train_data, val_data) = load_dataset_with_train_test_transforms(path_to_dataset,
                                                                 train_transform=train_transform,
                                                                 test_transform=test_transform,
                                                                 batch_size=4)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 220
train_model(efficientnet_b4, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name=path_to_dataset, num_epochs=num_epochs)
