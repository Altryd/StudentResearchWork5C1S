import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from utility import load_dataset_with_train_test_transforms, train_model

# Load the pre-trained ViT model
vit_b_32 = timm.create_model('vit_base_patch16_224', pretrained=True)
vit_b_32.patch_embed.proj = nn.Conv2d(
    in_channels=1,
    out_channels=vit_b_32.patch_embed.proj.out_channels,
    kernel_size=vit_b_32.patch_embed.proj.kernel_size,
    stride=vit_b_32.patch_embed.proj.stride,
    padding=vit_b_32.patch_embed.proj.padding,
    bias=vit_b_32.patch_embed.proj.bias is not None
)

# Adjust the classification head for binary classification
num_features = vit_b_32.head.in_features
vit_b_32.head = nn.Linear(num_features, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_b_32 = vit_b_32.to(device)

# Adjust dropout rates for regularization
vit_b_32.drop_rate = 0.1  # Dropout after attention and MLP layers
vit_b_32.attn_drop_rate = 0.1  # Dropout within attention layers
vit_b_32.drop_path_rate = 0.1  # Stochastic depth

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit_b_32.parameters(), lr=1e-4)

print("Model setup completed successfully!")
device
print("Modified input layer:")
print(vit_b_32.patch_embed.proj)

print(vit_b_32)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vit_b_32.parameters(), lr=0.0002, momentum=0.3)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure images are single-channel
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Validation/Test transformations
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
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
            dataset_name=r"CEDAR", num_epochs=150)
