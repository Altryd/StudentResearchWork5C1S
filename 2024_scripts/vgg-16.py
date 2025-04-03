import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

from utility import load_dataset_with_train_test_transforms, train_model

# vit_ = models.vit_b_32(pretrained=False)
vgg16 = models.vgg16(pretrained=False, num_classes=2)
vgg16.name = "vgg16"
num_classes = 2  # Укажите количество классов
# vgg16.fc = nn.Linear(vgg16.fc.in_features, num_classes)
"""
num_ftrs = vgg16.classifier[6].in_features
# vgg16._fc = torch.nn.Linear(in_features=vgg16._fc.in_features, out_features=num_classes, bias=True)

vgg16.classifier = nn.Sequential(
      nn.Linear(in_features=25088, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5, inplace=False),
      nn.Linear(in_features=4096, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5, inplace=False),
      nn.Linear(in_features=num_ftrs, out_features=2, bias=True)
)
"""


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.00001, momentum=0.2)
train_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.48235, 0.45882, 0.40784],
                         std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48235, 0.45882, 0.40784],
                         std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
])

(train_dataset, val_dataset, train_data,
 val_data) = load_dataset_with_train_test_transforms("../Samara_inverse",
                                                     train_transform=train_transform,
                                                     test_transform=test_transform, batch_size=4)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 130
best_acc = 0
min_loss = 10 ** 10
train_model(vgg16, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name="Samara_inverse",
            num_epochs=num_epochs)
