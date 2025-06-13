import itertools
import json
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.models import ViT_B_32_Weights, ViT_B_16_Weights

from used_transforms import *

from TransformDataset import TransformDataset


# Функция для выбора триплетов (online triplet mining)
def select_triplets(embeddings, labels, use_hard=False, margin=1.0):
    """
    Выполняет онлайн-майнинг триплетов для выбора триплетов (якорь, позитивный, негативный) из эмбеддингов.

    Алгоритм проходит по всем уникальным меткам в наборе данных. Для каждой метки формируются триплеты,
    состоящие из якоря, позитивного примера (того же класса, что и якорь) и негативного примера (другого класса).
    Триплеты создаются только если для класса есть как минимум два примера в батче. Расстояния между эмбеддингами
    вычисляются с использованием L2-расстояния. Затем триплеты отбираются на основе параметра `use_hard`:
    - Если `use_hard=False`, выбираются semi-hard триплеты, т.е. триплеты с `pos_dist < neg_dist < pos_dist + margin`.
    - Если `use_hard=True`, выбираются hard триплеты, т.е. триплеты с `neg_dist < pos_dist`.

    :param embeddings: (torch.Tensor): Тензор формы (batch_size, embedding_size), содержащий эмбеддинги.
    :param labels: (torch.Tensor): Тензор формы (batch_size,), содержащий метки классов.
    :param use_hard: (bool, опционально): Если True, выбираются hard триплеты.
    :param margin: (float, опционально): Значение отступа для выбора semi-hard триплетов.
    :return:
    list: Список триплетов, где каждый триплет — это кортеж (anchor_idx, pos_idx, neg_idx) с индексами в батче.
    """
    triplets = []
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)
    l2_distance = PairwiseDistance(p=2)

    for pos_class in unique_labels:
        pos_mask = labels == pos_class
        neg_mask = ~pos_mask  # logical inverse
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) < 2:  # Нужны как минимум 2 примера для positive
            continue

        for anchor_idx, pos_idx in itertools.combinations(pos_indices, 2):
            anchor_emb = embeddings[anchor_idx]
            pos_emb = embeddings[pos_idx]
            neg_embs = embeddings[neg_indices]

            # Вычисление расстояний
            pos_dist = l2_distance(anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0)).item()
            neg_dists = l2_distance(anchor_emb.unsqueeze(0).expand_as(neg_embs), neg_embs)

            # Semi-hard: pos_dist < neg_dist < pos_dist + margin
            valid_mask = (pos_dist < neg_dists) & (neg_dists < pos_dist + margin)
            valid_neg_indices = neg_indices[valid_mask.cpu().numpy()]
            if len(valid_neg_indices) > 0:
                neg_idx = np.random.choice(valid_neg_indices)
                triplets.append((anchor_idx, pos_idx, neg_idx))
        if use_hard:
            # Hard: neg_dist < pos_dist
            valid_mask = neg_dists < pos_dist
            valid_neg_indices = neg_indices[valid_mask.cpu().numpy()]
            if len(valid_neg_indices) > 0:
                neg_idx = valid_neg_indices[np.argmin(neg_dists[valid_mask].cpu().detach().numpy())]
                triplets.append((anchor_idx, pos_idx, neg_idx))

    return triplets


def evaluate(model, dataloader, class_to_idx, l2_distance, device):
    model.eval()
    embeddings_dict = {cls: [] for cls in class_to_idx}  # Словарь для эмбеддингов по классам
    labels_dict = {cls: [] for cls in class_to_idx}  # Словарь для хранения индексов классов

    # Вычисляем эмбеддинги для всего датасета
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            for i, label in enumerate(labels.cpu().numpy()):
                try:
                    label_str = dataloader.dataset.base.classes[label]
                except:
                    label_str = dataloader.dataset.base.dataset.classes[label]
                embeddings_dict[label_str].append(embeddings[i])
                labels_dict[label_str].append(i)

    # Преобразуем списки в тензоры
    for cls in embeddings_dict:
        embeddings_dict[cls] = torch.stack(embeddings_dict[cls]) if embeddings_dict[cls] else None

    all_distances = {"pos-pos": [], "gen-forg": [], "gen-imp": []}

    # Считаем расстояния между всеми нужными парами
    for gen_class in class_to_idx:
        if "gen" not in gen_class and "orig" not in gen_class:
            continue  # Пропускаем поддельные классы

        num_class = re.findall(r"\d+", gen_class)[0]  # ID пользователя
        name = gen_class[re.search("\d+", gen_class).end():]  # Имя
        forg_class = f"forg_{num_class}{name}"  # Поддельный класс
        gen_embs = embeddings_dict[gen_class]
        forg_embs = embeddings_dict.get(forg_class, None)
        if forg_embs is None:
            forg_class = f"forged_{num_class}{name}"
            forg_embs = embeddings_dict.get(forg_class, None)

        # Positive-Positive (внутри класса)
        if gen_embs is not None and len(gen_embs) > 1:
            pos_pos_dists = l2_distance(gen_embs.unsqueeze(1), gen_embs.unsqueeze(0))
            pos_pos_dists = pos_pos_dists[np.triu_indices(len(gen_embs), k=1)]  # Верхний треугольник (без диагонали)
            all_distances["pos-pos"].extend(pos_pos_dists.cpu().numpy())

        # Genuine-Forged (настоящая vs поддельная)
        if gen_embs is not None and forg_embs is not None:
            gen_forg_dists = l2_distance(gen_embs.unsqueeze(1), forg_embs.unsqueeze(0)).flatten()
            all_distances["gen-forg"].extend(gen_forg_dists.cpu().numpy())

        # Genuine-Impostor (настоящая vs чужая)
        for other_class in class_to_idx:
            if other_class == gen_class or other_class == forg_class:
                continue  # Пропускаем свой класс и подделки

            other_embs = embeddings_dict[other_class]
            if gen_embs is not None and other_embs is not None:
                gen_imp_dists = l2_distance(gen_embs.unsqueeze(1), other_embs.unsqueeze(0)).flatten()
                all_distances["gen-imp"].extend(gen_imp_dists.cpu().numpy())

    return all_distances


def compute_metrics_from_evaluation(evaluation_results, threshold, logger, epoch=None):
    """
    Собирает расстояния из evaluate() в единый массив и считает метрики.
    :param evaluation_results: Результат evaluate(), содержащий три списка расстояний
    :param threshold: Порог принятия решения
    :param epoch: Номер эпохи для сохранения в метриках (необязательно)
    :return: Словарь с метриками
    """
    # Собираем все расстояния в один массив
    all_distances = (
            np.array(evaluation_results["pos-pos"]).flatten().astype(np.float32).tolist() +
            np.array(evaluation_results["gen-forg"]).flatten().astype(np.float32).tolist() +
            np.array(evaluation_results["gen-imp"]).flatten().astype(np.float32).tolist()
    )

    # Формируем метки: pos-pos (1), остальные (0)
    all_labels = (
            [1] * len(evaluation_results["pos-pos"]) +
            [0] * len(evaluation_results["gen-forg"]) +
            [0] * len(evaluation_results["gen-imp"])
    )

    #logger.info(f"Pairs distribution: pos-pos={len(evaluation_results['pos-pos'])}, "
    #            f"gen-forg={len(evaluation_results['gen-forg'])}, gen-imp={len(evaluation_results['gen-imp'])}")

    # Если списки пусты, возвращаем заглушки, но логируем проблему
    if not all_distances or not all_labels:
        logger.warning("No valid pairs found for metrics computation. Returning placeholder metrics.")
        return {
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0,
            'true_positive': 0,
            'threshold': threshold
        }

    return calculate_metrics(np.array(all_distances), np.array(all_labels), threshold, epoch)



# Функция обучения одной эпохи
def train_epoch(model, dataloader, loss_fn, optimizer, logger, device, MARGIN=1.0, use_hard=True, best_threshold=0.0):
    model.train()
    total_loss = 0
    num_valid_triplets = 0
    all_distances = {"pos-pos": [], "gen-forg": [], "gen-imp": []}
    l2_distance = PairwiseDistance(p=2)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Получение эмбеддингов
        embeddings = model(images)

        # Выбор триплетов
        triplets = select_triplets(embeddings, labels, use_hard=use_hard, margin=MARGIN)
        if not triplets:
            continue

        anchor_idx, pos_idx, neg_idx = zip(*triplets)
        anchor_emb = embeddings[list(anchor_idx)]
        pos_emb = embeddings[list(pos_idx)]
        neg_emb = embeddings[list(neg_idx)]

        # Вычисление потерь
        loss = loss_fn(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(triplets)
        num_valid_triplets += len(triplets)

        if batch_idx % 2 == 0:  # Уменьшил частоту логов для скорости
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Valid triplets: {len(triplets)}")

            # Сбор метрик для графиков
            with torch.no_grad():
                for i, label in enumerate(labels.cpu().numpy()):
                    try:
                        label_str = dataloader.dataset.base.classes[label]
                    except:
                        label_str = dataloader.dataset.base.dataset.classes[label]
                    for j in range(len(labels)):
                        if i != j:
                            try:
                                label_to_compare = dataloader.dataset.base.classes[labels[j].cpu().numpy()]
                            except:
                                label_to_compare = dataloader.dataset.base.dataset.classes[labels[j].cpu().numpy()]
                            dist = l2_distance(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                            if label_str == label_to_compare:
                                all_distances["pos-pos"].append(dist)
                            elif "forg" in label_to_compare:
                                all_distances["gen-forg"].append(dist)
                            else:
                                all_distances["gen-imp"].append(dist)

    # print("num_valid_triplets=", num_valid_triplets)
    if num_valid_triplets == 0:
        if hasattr(train_epoch, 'last_metrics') and train_epoch.last_metrics is not None:
            return 0, 0, train_epoch.last_metrics
        return 0, 0, {
            'f1_score': 0.0,  # Будет заменено в главном цикле последними метриками
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0,
            'true_positive': 0,
            'threshold': best_threshold
        }

    train_metrics = compute_metrics_from_evaluation(all_distances, best_threshold, epoch=None, logger=logger)
    train_epoch.last_metrics = train_metrics
    avg_loss = total_loss / num_valid_triplets if num_valid_triplets > 0 else 0
    return avg_loss, num_valid_triplets, train_metrics


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


from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score, \
    precision_recall_curve


def calculate_metrics(all_distances, all_labels, threshold, epoch=None):
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
    # Micro ROC AUC для бинарной задачи

    # print(all_labels)
    # print(len(all_labels))
    roc_auc = roc_auc_score(all_labels,
                            -all_distances)  # Используем -distances, так как меньшее расстояние = большее сходство
    # average_precision_score вычисляет площадь под кривой precision-recall, что соответствует precision-recall AUC
    ap = average_precision_score(all_labels, -all_distances)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    accuracy = accuracy_score(all_labels, predictions)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(all_labels, predictions)

    if epoch is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, -all_distances)
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f'Precision-Recall AUC = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Epoch {epoch})')
        plt.legend()
        plt.savefig(f'pr_curve_epoch_{epoch}.png')
        plt.close()

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


# Функция для сохранения метрик в JSON
def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


# Функция для загрузки метрик из JSON
def load_metrics(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {
        'train_losses': [],
        'train_f1_scores': [],
        'train_roc_aucs': [],
        'train_pr_aucs': [],
        'val_f1_scores': [],
        'val_roc_aucs': [],
        'val_pr_aucs': []
    }


# Функция для построения и сохранения графиков
def plot_and_save_curves(metrics, current_epoch, dataset_name, save_dir='plots', epoch_label="unknown",
                         model_name="unknown"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(0, current_epoch + 1)

    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics['val_pr_aucs'], label='Validation Loss (approx)')  # Используем PR AUC как прокси
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_curves_{dataset_name}_model_{model_name}_epoch_{epoch_label}.png')
    plt.close()

    # График F1-score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_f1_scores'], label='Train F1-score')
    plt.plot(epochs, metrics['val_f1_scores'], label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Training and Validation F1-score')
    plt.legend()
    plt.savefig(f'{save_dir}/f1_curves_{dataset_name}_model_{model_name}_epoch_{epoch_label}.png')
    plt.close()

    # График ROC AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_roc_aucs'], label='Train ROC AUC')
    plt.plot(epochs, metrics['val_roc_aucs'], label='Validation ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('Training and Validation ROC AUC')
    plt.legend()
    plt.savefig(f'{save_dir}/roc_auc_curves_{dataset_name}_model_{model_name}_epoch_{epoch_label}.png')
    plt.close()

    # График Precision-Recall AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_pr_aucs'], label='Train Precision-Recall AUC')
    plt.plot(epochs, metrics['val_pr_aucs'], label='Validation Precision-Recall AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Precision-Recall AUC')
    plt.title('Training and Validation Precision-Recall AUC')
    plt.legend()
    plt.savefig(f'{save_dir}/pr_auc_curves_{dataset_name}_model_{model_name}_epoch_{epoch_label}.png')
    plt.close()

# Реализация LayerNorm2d скопирована из torchvision.models.convnext !
import torch
from torch import nn, Tensor
from torch.nn import functional as F, PairwiseDistance


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
