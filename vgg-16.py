import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

vgg16 = models.vgg16(pretrained=True)
num_classes = 2  # Укажите количество классов
# vgg16.fc = nn.Linear(vgg16.fc.in_features, num_classes)
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



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

# train_data, val_data = random_split(dataset, [train_size, test_size])
# train_dataset = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
# val_dataset = DataLoader(dataset=val_data, shuffle=True)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))


num_epochs = 25
best_acc = 0
min_loss = 10 ** 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

for epoch in range(num_epochs):
    vgg16.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = vgg16(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    vgg16.eval()
    total_correct = 0
    with torch.no_grad():
        predictions = []
        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vgg16(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            predictions.append(predicted)
            total_correct += (predicted == labels).sum().item()
        # print(f"predictions: {predictions}")

    accuracy = total_correct / len(val_data)
    if accuracy > best_acc:
        best_acc = accuracy
        min_loss = running_loss / len(train_data)
        torch.save(vgg16, f'vgg16_2fc_samara_epoch_best_acc.pth')
    elif best_acc == accuracy and running_loss/len(train_data) < min_loss:
        min_loss = running_loss/len(train_data)
        torch.save(vgg16, f'vgg16_2fc_samara_epoch_best_acc.pth')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f} ;  Test Accuracy: {(100 * accuracy):.2f}%')
    # torch.save(vgg16, f'vgg16_2fc_epoch{epoch+1}.pth')

print(f"Best acc: {best_acc} with corresponding loss: {min_loss}")
torch.save(vgg16, 'vgg16_2fc_trained.pth')
