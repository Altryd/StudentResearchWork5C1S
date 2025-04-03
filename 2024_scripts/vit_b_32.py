import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from TransformDataset import TransformDataset
from utility import load_dataset_with_train_test_transforms, train_model

# vit_b_32 = models.vit_b_32(num_classes=2)
vit_b_32 = torch.load("../vit_b_32_CEDAR_best_acc.pth", weights_only=False)  # TODO !!!
# vit_b_32.name = "vit_b_32"
# vit_b_32 = torch.load("vit_b_32_2fc_samara_trained.pth", weights_only=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vit_b_32.parameters(), lr=0.001, momentum=0.3)
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.GaussianNoise(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Split dataset into train and validation
path_to_dataset = r"../CEDAR/signatures"
(train_dataset, val_dataset,
 train_data, val_data) = load_dataset_with_train_test_transforms(path_to_dataset,
                                                                 train_transform=train_transform,
                                                                 test_transform=test_transform,
                                                                 batch_size=8)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


train_model(vit_b_32, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name="CEDAR", num_epochs=100)
