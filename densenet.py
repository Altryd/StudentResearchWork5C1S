import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

densenet121 = models.densenet121(pretrained=True)
num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов

densenet121.classifier = nn.Linear(densenet121.classifier.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(densenet121.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((260, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: это ваще нужно?
])

dataset = datasets.ImageFolder(root=r'CEDAR\signatures', transform=transform)
loader = DataLoader(dataset=dataset, shuffle=True)
train_ratio = 0.8
train_size = int(len(dataset)*train_ratio)
test_size = len(dataset) - train_size
print(dataset)

train_data, val_data = random_split(dataset, [train_size, test_size])
train_dataset = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_dataset = DataLoader(dataset=val_data, shuffle=True)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
densenet121.to(device)

for epoch in range(num_epochs):
    densenet121.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = densenet121(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    densenet121.eval()
    total_correct = 0
    with torch.no_grad():

        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = densenet121(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(val_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f} ;  Test Accuracy: {(100 * accuracy):.2f}%')
    torch.save(densenet121, f'densenet121_handwritten_2fc_epoch{epoch+1}.pth')


torch.save(densenet121, 'densenet121_handwritten_2fc_trained.pth')
