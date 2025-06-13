import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from TransformDataset import TransformDataset
from sklearn.model_selection import train_test_split

from utility import train_model, load_dataset_with_train_test_transforms

swin_b = models.swin_v2_b(pretrained=True)
num_features = swin_b.head.in_features  # TODOOOOOOOOOOO Получаем количество входных признаков
swin_b.head = torch.nn.Linear(num_features, 2)  # Заменяем на новый слой с 2 классами
# swin_b = torch.load("swin_v2_b_Samara_inverse_trained.pth", weights_only=False)
# swin_b.name = "swin_v2_b"


criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(swin_b.parameters(), lr=0.00002, momentum=0.3)
optimizer = torch.optim.Adam(swin_b.parameters(), lr=0.04)
scheduler = ExponentialLR(optimizer, gamma=0.2)
train_transform_v2 = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    # transforms.CenterCrop((256, 256)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),  # to 0.0 - 1.0
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform_v2 = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    # transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_train_transform = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.ToTensor(),  # to 0.0 - 1.0
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform_swin_b = transforms.Compose([
    transforms.Resize((238, 238), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    base_train_transform
])

train_transform_swin_s = transforms.Compose([
    transforms.Resize((246, 246), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    base_train_transform
])

base_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_swin_s = transforms.Compose([
    transforms.Resize((246, 246), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    base_test_transform
])

path_to_dataset = r"../Samara"
"""
(train_dataset, val_dataset,
 train_data, val_data) = load_dataset_with_train_test_transforms(path_to_dataset,
                                                                 train_transform=train_transform_v2,
                                                                 test_transform=test_transform_v2)
"""
(train_dataset, val_dataset,
 train_data, val_data) = load_dataset_with_train_test_transforms(path_to_dataset,
                                                                 train_transform=train_transform_v2,
                                                                 test_transform=test_transform_v2)
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))

train_model(swin_b, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name=path_to_dataset, num_epochs=150, scheduler=scheduler)
