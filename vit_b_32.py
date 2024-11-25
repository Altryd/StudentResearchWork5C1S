import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

vit_b_32 = models.vit_b_32(num_classes=2)
# vit_b_32 = torch.load("vit_b_32_2fc_trained.pth", weights_only=False)
num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов
# vit_b_32.fc = nn.Linear(vit_b_32.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vit_b_32.parameters(), lr=0.0002, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: это ваще нужно?
])

dataset = datasets.ImageFolder(root=r'CEDAR\signatures', transform=transform)
loader = DataLoader(dataset=dataset, shuffle=True)
print(dataset)
train_ratio = 0.8
train_size = int(len(dataset)*train_ratio)
test_size = len(dataset) - train_size
print(dataset)

train_data, val_data = random_split(dataset, [train_size, test_size])
train_dataset = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_dataset = DataLoader(dataset=val_data, batch_size=32, shuffle=True)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_b_32.to(device)

for epoch in range(num_epochs):
    vit_b_32.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = vit_b_32(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    vit_b_32.eval()
    total_correct = 0
    with torch.no_grad():

        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vit_b_32(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(val_data)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f} ;  Test Accuracy: {(100 * accuracy):.2f}%')
    torch.save(vit_b_32, f'vit_b_32_2fc_epoch{epoch+1}.pth')


torch.save(vit_b_32, 'vit_b_32_2fc_trained.pth')
