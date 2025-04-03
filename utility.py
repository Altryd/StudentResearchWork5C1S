import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from TransformDataset import TransformDataset


def load_dataset_full(dataset_path: str, batch_size=16, transform=None, shuffle=True):
    dataset = datasets.ImageFolder(root=dataset_path)
    train_data = TransformDataset(dataset, transform)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_dataset, train_data


def load_dataset(dataset_path: str, batch_size=4, transform=None, test_ratio=0.2):
    if transform:
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    else:
        dataset = datasets.ImageFolder(root=dataset_path)
    loader = DataLoader(dataset=dataset, shuffle=True)
    print(dataset)
    # train_size = int(len(dataset)*train_ratio)
    # test_size = len(dataset) - train_size
    # print(dataset)

    # Split dataset into train and validation
    train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=test_ratio,
                                                  stratify=dataset.targets)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)

    # Create DataLoader
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset, train_data, val_data


def load_dataset_with_train_test_transforms(dataset_path: str, train_transform,
                                            test_transform,
                                            batch_size=4, test_ratio=0.2, random_state=None,
                                            all_shuffle=True):
    """
    if test_ratio == 1.0:
        dataset = datasets.ImageFolder(root=dataset_path)
        loader = DataLoader(dataset=dataset, shuffle=True)
        train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return loader
    """
    dataset = datasets.ImageFolder(root=dataset_path)
    loader = DataLoader(dataset=dataset, shuffle=all_shuffle)
    print(dataset)
    # train_size = int(len(dataset)*train_ratio)
    # test_size = len(dataset) - train_size
    # print(dataset)

    # Split dataset into train and validation


    train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=test_ratio,
                                                  stratify=dataset.targets, random_state=random_state)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    train_data = TransformDataset(train_data, train_transform)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    val_data = TransformDataset(val_data, test_transform)

    # Create DataLoader
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=all_shuffle)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=all_shuffle)

    return train_dataset, val_dataset, train_data, val_data


def load_dataset_with_train_test_valid_transforms(dataset_path: str, train_transform, validation_transform,
                                            test_transform, batch_size=4, test_val_batch_size=None,
                                                  test_val_ratio=0.2, val_ratio=0.5, random_state=None,
                                            all_shuffle=True):
    """
    if test_ratio == 1.0:
        dataset = datasets.ImageFolder(root=dataset_path)
        loader = DataLoader(dataset=dataset, shuffle=True)
        train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return loader
    """
    if not test_val_batch_size:
        test_val_batch_size = batch_size
    dataset = datasets.ImageFolder(root=dataset_path)
    loader = DataLoader(dataset=dataset, shuffle=all_shuffle)
    print(dataset)
    # train_size = int(len(dataset)*train_ratio)
    # test_size = len(dataset) - train_size
    # print(dataset)

    # Split dataset into train and validation

    train_indices, test_val_indices = train_test_split(list(range(len(dataset.targets))),
                                                       test_size=test_val_ratio,
                                                       stratify=dataset.targets, random_state=random_state)

    test_indices, val_indices = train_test_split(test_val_indices,
                                                 stratify=[dataset.targets[i] for i in test_val_indices],
                                                 test_size=val_ratio,
                                                 random_state=random_state)

    train_data = torch.utils.data.Subset(dataset, train_indices)
    train_data = TransformDataset(train_data, train_transform, imgs=[dataset.imgs[i] for i in train_indices],
                                  targets=[dataset.targets[i] for i in train_indices])

    val_data = torch.utils.data.Subset(dataset, val_indices)
    val_data = TransformDataset(val_data, validation_transform, imgs=[dataset.imgs[i] for i in val_indices],
                                targets=[dataset.targets[i] for i in val_indices])

    test_data = torch.utils.data.Subset(dataset, test_indices)
    test_data = TransformDataset(test_data, test_transform, imgs=[dataset.imgs[i] for i in test_indices],
                                 targets=[dataset.targets[i] for i in test_indices])

    # Create DataLoader
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=all_shuffle)
    if test_val_batch_size == -1:
        test_val_batch_size = len(val_data)
    val_dataset = DataLoader(val_data, batch_size=test_val_batch_size, shuffle=all_shuffle)
    test_dataset = DataLoader(test_data, batch_size=test_val_batch_size, shuffle=all_shuffle)

    return train_dataset, val_dataset, test_dataset, train_data, val_data, test_data



def train_model(model, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
                scheduler=None, dataset_name="", model_name=None, num_epochs=35):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not model_name:
        if hasattr(model, "name"):
            model_name = model.name
        else:
            model_name = model._get_name()
    print(f"model_name={model_name}")
    best_acc = 0
    min_loss = 10 ** 10

    for epoch in range(num_epochs):
        model.train()  # Устанавливаем модель в режим обучения
        running_loss = 0.0
        total_correct_train = 0
        for images, labels in train_dataset:
            images, labels = images.to(device), labels.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)

            _, predicted = torch.max(outputs.data, dim=1)
            total_correct_train += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        model.eval()
        total_correct = 0
        with torch.no_grad():

            for inputs, labels in val_dataset:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(val_data)
        if accuracy > best_acc:
            torch.save(model, f'{model_name}_{dataset_name}_best_acc.pth')
            best_acc = accuracy
            min_loss = running_loss / len(train_data)
        elif best_acc == accuracy and running_loss / len(train_data) < min_loss:
            min_loss = running_loss / len(train_data)
            torch.save(model, f'{model_name}_{dataset_name}_best_acc.pth')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_data):.4f} ;  '
            f'Train Accuracy: {100 * total_correct_train / len(train_data):.2f}% ; '
            f'Test Accuracy: {(100 * accuracy):.2f}%')

    print(f"Best acc: {best_acc} with corresponding loss: {min_loss}")

    torch.save(model, f'{model_name}_{dataset_name}_trained.pth')
    pass

