import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

resnet101 = models.resnet101(pretrained=True)
num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов
resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet101.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((260, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: это ваще нужно?
])

dataset = datasets.ImageFolder(root=r'Samara', transform=transform)
loader = DataLoader(dataset=dataset, shuffle=True)
print(dataset)
test_ratio = 0.2
# train_size = int(len(dataset)*train_ratio)
# test_size = len(dataset) - train_size
# print(dataset)

#Split dataset into train and validation
train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=test_ratio, stratify=dataset.targets)
train_data = torch.utils.data.Subset(dataset, train_indices)
val_data = torch.utils.data.Subset(dataset, val_indices)

#Create DataLoader
train_dataset = DataLoader(train_data, batch_size=4, shuffle=True)
val_dataset = DataLoader(val_data, batch_size=4, shuffle=True)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 35
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet101.to(device)
best_acc = 0
min_loss = 10 ** 10

for epoch in range(num_epochs):
    resnet101.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = resnet101(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    resnet101.eval()
    total_correct = 0
    with torch.no_grad():

        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet101(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(val_data)
    if accuracy > best_acc:
        torch.save(resnet101, f'resnet101_2fc_samara_best_acc.pth')
        best_acc = accuracy
    elif best_acc == accuracy and running_loss/len(train_data) < min_loss:
        min_loss = running_loss/len(train_data)
        torch.save(resnet101, f'resnet101_2fc_samara_best_acc.pth')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f} ;  Test Accuracy: {(100 * accuracy):.2f}%')

print(f"Best acc: {best_acc} with corresponding loss: {min_loss}")

torch.save(resnet101, 'resnet101_2fc_samara_trained.pth')

"""
Epoch [1/5], Loss: 0.8967 ;  Test Accuracy: 50.45%
Epoch [2/5], Loss: 0.6850 ;  Test Accuracy: 65.34%
Epoch [3/5], Loss: 0.6060 ;  Test Accuracy: 71.26%
Epoch [4/5], Loss: 0.5168 ;  Test Accuracy: 70.45%
Epoch [5/5], Loss: 0.4239 ;  Test Accuracy: 82.02%
for Handwritten signature verification"""
