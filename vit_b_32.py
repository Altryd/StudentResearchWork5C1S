import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from TransformDataset import TransformDataset
from utility import load_dataset_with_train_test_transforms, train_model

vit_b_32 = models.vit_b_32(num_classes=2)
# vit_b_32 = torch.load("vit_b_32_2fc_samara_trained.pth", weights_only=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vit_b_32.parameters(), lr=0.0002, momentum=0.3)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: это ваще нужно?
])

#Split dataset into train and validation
path_to_dataset = r"Samara_inverse"
(train_dataset, val_dataset,
 train_data, val_data) = load_dataset_with_train_test_transforms(path_to_dataset,
                                                                 train_transform=models.ViT_B_32_Weights.IMAGENET1K_V1.transforms(),
                                                                 test_transform=models.ViT_B_32_Weights.IMAGENET1K_V1.transforms())

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


train_model(vit_b_32, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
            dataset_name=path_to_dataset, num_epochs=45)