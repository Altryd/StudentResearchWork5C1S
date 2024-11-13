import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

efficientnet_b4 = models.efficientnet_b4(pretrained=True)
num_classes = 2  # Укажите количество классов
# efficientnet_b4.fc = nn.Linear(efficientnet_b4.fc.in_features, num_classes)
num_ftrs = efficientnet_b4.classifier[1].in_features
# efficientnet_b4._fc = torch.nn.Linear(in_features=efficientnet_b4._fc.in_features, out_features=num_classes, bias=True)

efficientnet_b4.classifier = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(efficientnet_b4.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((260, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: это ваще нужно?
])

dataset = datasets.ImageFolder(root=r'C:\Users\Altryd\CEDAR\signatures', transform=transform)
loader = DataLoader(dataset=dataset, shuffle=True)
print(dataset)
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

for epoch in range(num_epochs):
    efficientnet_b4.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = efficientnet_b4(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    efficientnet_b4.eval()
    total_correct = 0
    with torch.no_grad():
        predictions = []
        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = efficientnet_b4(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            predictions.append(predicted)
            total_correct += (predicted == labels).sum().item()
        print(f"predictions: {predictions}")

    accuracy = total_correct / len(val_data)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f} ;  Test Accuracy: {(100 * accuracy):.2f}%')
    torch.save(efficientnet_b4.state_dict(), f'efficientnet_b4_2fc_epoch{epoch+1}.pth')


torch.save(efficientnet_b4.state_dict(), 'efficientnet_b4_2fc_trained.pth')
