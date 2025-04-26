import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.models import ViT_B_32_Weights, ViT_B_16_Weights

from used_transforms import *

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


# создание модели
def create_model(model_name="resnet34", embedding_size=128, pretrained_path=None, device="cpu"):
    """
    Создает модель, указанную в model_name. Заменяет количество выходных классов модели на embedding_size
    (для получения эмбеддингов)
    :param model_name:
    :param embedding_size:
    :param pretrained_path:
    :param device:
    :return:
    (model, train_transform, test_transform)
    """
    train_transform = None
    test_transform = None
    dict_of_transforms = {"resnet34": (train_transform_resnet_v34, test_transform_resnet_v34),
                          "resnet101": (train_transform_resnet_v101, test_transform_resnet_v101),
                          "efficientnet_b0": (train_transform_efficient_net_b0, test_transform_efficient_net_b0),
                          "vit_b_32": (train_transform_vit_b_32, test_transform_vit_b_32),
                          "vit_b_16": (train_transform_vit_b_16, test_transform_vit_b_16),
                          "convnext_tiny": (train_transform_convnext_tiny, test_transform_convnext_tiny),
                          "inception_resnet_v2": (train_transform_resnet_inception, test_transform_resnet_inception),
                          "mobilenet_v3_small": (train_transform_mobilenet_v3_small, test_transform_mobilenet_v3_small),
                          "mobilenet_v3_large": (train_transform_mobilenet_v3_small, test_transform_mobilenet_v3_small)}
    if model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        # Заменяем последний слой на слой с нужным размером эмбеддинга
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, embedding_size)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, embedding_size)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        """
        model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=embedding_size,
                                                               bias=True))
        """
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=num_ftrs, out_features=embedding_size, bias=True),
            torch.nn.BatchNorm1d(embedding_size)
        )
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        # TODO спросить: model.heads = nn.Linear(768, out_features=EMBEDDING_SIZE)  # Новая размерность, например, 128
        # Это может быть полезно
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=EMBEDDING_SIZE,
                                                               bias=True))
        """
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        # TODO спросить: model.heads = nn.Linear(768, out_features=EMBEDDING_SIZE)  # Новая размерность, например, 128
        # Это может быть полезно
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=EMBEDDING_SIZE,
                                                               bias=True))
        """
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=True)
        # print(model.classifier)
        num_ftrs = model.classifier[0].normalized_shape[0]
        from utility import LayerNorm2d
        model.classifier = torch.nn.Sequential(LayerNorm2d((num_ftrs,), eps=1e-6, elementwise_affine=True),
                                               torch.nn.Flatten(start_dim=1, end_dim=-1),
                                               torch.nn.Linear(in_features=num_ftrs, out_features=embedding_size,
                                                               bias=True))
    elif model_name == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=True,
                                  num_classes=embedding_size)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        first_num_fltrs = model.classifier[0].in_features
        num_ftrs_second = model.classifier[3].in_features

        model.classifier = torch.nn.Sequential(
            nn.Linear(in_features=first_num_fltrs, out_features=num_ftrs_second, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_ftrs_second, out_features=embedding_size, bias=True),
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        first_num_fltrs = model.classifier[0].in_features
        num_ftrs_second = model.classifier[3].in_features

        model.classifier = torch.nn.Sequential(
            nn.Linear(in_features=first_num_fltrs, out_features=num_ftrs_second, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_ftrs_second, out_features=embedding_size, bias=True),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
        try:
            model_name = model.name
        except:
            model.name = model_name
    else:
        model.name = model_name
    train_transform, test_transform = dict_of_transforms[model_name]
    return model.to(device), train_transform, test_transform


from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score


def calculate_metrics(all_distances, all_labels, threshold):
    """
    Высчитывает метрики по полученным расстояниям между эмбеддингами с использованием labels и threshold.
    :param distances: Расстояния между эмбеддингами
    :param labels: Лейблы (чаще всего 0 и 1)
    :param threshold: Порог принятия решения о принадлежности к классу
    :return:
    Словарь с ключами threshold, roc_auc, average_precision, true_negative, false_positive, false_negative,
    true_positive, accuracy, precision, recall, f1_score
    """
    # Бинаризация предсказаний по порогу
    predictions = (all_distances <= threshold).astype(int)

    # Вычисление метрик
    roc_auc = roc_auc_score(all_labels,
                            -all_distances)  # Используем -distances, так как меньшее расстояние = большее сходство
    ap = average_precision_score(all_labels, -all_distances)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    accuracy = accuracy_score(all_labels, predictions)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(all_labels, predictions)

    return {
        'threshold': threshold,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


# Реализация LayerNorm2d скопирована из torchvision.models.convnext !
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
