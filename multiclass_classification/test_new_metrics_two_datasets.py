# -*- coding: utf-8 -*-
import datetime
import re
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import itertools

from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights

from used_transforms import *  # Предполагается, что это ваши трансформации
from TransformDataset import TransformDataset
from utility import load_dataset_with_train_test_transforms, load_dataset_full, create_model, calculate_metrics
import logging
import time
logger = logging.getLogger(__name__)


# Глобальные константы
EMBEDDING_SIZE = 128
MARGIN = 1.0
EPOCHS = 50
BATCH_SIZE = 50
RANDOM_STATE = 111
dataset_path = "datasets/march_1_full"
second_dataset_path = "datasets/april_6signatures"
# second_dataset_path = None
dataset_name = dataset_path.split("/")[-1]
if second_dataset_path:
    second_dataset_name = second_dataset_path.split("/")[-1]
SAVE_MODEL_EVERY_N_EPOCHS = 20


# Функция для выбора триплетов (online triplet mining)
def select_triplets(embeddings, labels, use_hard=False, margin=MARGIN):
    """
    Выполняет онлайн-майнинг триплетов для выбора триплетов (якорь, позитивный, негативный) из эмбеддингов.

    Алгоритм проходит по всем уникальным меткам в наборе данных. Для каждой метки формируются триплеты,
    состоящие из якоря, позитивного примера (того же класса, что и якорь) и негативного примера (другого класса).
    Триплеты создаются только если для класса есть как минимум два примера в батче. Расстояния между эмбеддингами
    вычисляются с использованием L2-расстояния. Затем триплеты отбираются на основе параметра `use_semihard`:
    - Если `use_semihard=True`, выбираются semi-hard триплеты, т.е. триплеты с `pos_dist < neg_dist < pos_dist + margin`.
    - Если `use_semihard=False`, выбираются hard триплеты, т.е. триплеты с `neg_dist < pos_dist`.

    :param embeddings: (torch.Tensor): Тензор формы (batch_size, embedding_size), содержащий эмбеддинги.
    :param labels: (torch.Tensor): Тензор формы (batch_size,), содержащий метки классов.
    :param use_semihard: (bool, опционально): Если True, выбираются semi-hard триплеты; иначе — hard.
    По умолчанию False.
    :param margin: (float, опционально): Значение отступа для выбора semi-hard триплетов.
    :return:
    list: Список триплетов, где каждый триплет — это кортеж (anchor_idx, pos_idx, neg_idx) с индексами в батче.
    """
    triplets = []
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

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
            # logger.info("The number of triplets is less than 5, using hard triplets too")
            # Hard: neg_dist < pos_dist
            valid_mask = neg_dists < pos_dist
            valid_neg_indices = neg_indices[valid_mask.cpu().numpy()]
            """
            if len(valid_neg_indices) > 0:
                neg_idx = valid_neg_indices[np.argmin(neg_dists[valid_mask].cpu().detach().numpy())]
                triplets.append((anchor_idx, pos_idx, neg_idx))
                # возможно нужно так:
            """
            if len(valid_neg_indices) > 0:
                neg_idx = valid_neg_indices[np.argmin(neg_dists[valid_mask].cpu().detach().numpy())]
                triplets.append((anchor_idx, pos_idx, neg_idx))

    return triplets


# Функция обучения одной эпохи
def train_epoch(model, dataloader, loss_fn, optimizer, use_hard=True):
    model.train()
    total_loss = 0
    num_valid_triplets = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Получение эмбеддингов
        embeddings = model(images)

        # Выбор триплетов
        triplets = select_triplets(embeddings, labels, use_hard=use_hard)
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

    avg_loss = total_loss / num_valid_triplets if num_valid_triplets > 0 else 0
    return avg_loss, num_valid_triplets


def evaluate(model, dataloader, class_to_idx, l2_distance):
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
                labels_dict[label_str].append(i)   # ??????

    # Преобразуем списки в тензоры
    for cls in embeddings_dict:
        embeddings_dict[cls] = torch.stack(embeddings_dict[cls]) if embeddings_dict[cls] else None

    all_distances = {"pos-pos": [], "gen-forg": [], "gen-imp": []}

    # Считаем расстояния между всеми нужными парами
    for gen_class in class_to_idx:
        if "gen" not in gen_class:
            continue  # Пропускаем поддельные классы

        num_class = re.findall(r"\d+", gen_class)[0]  # ID пользователя
        name = gen_class[re.search("\d+", gen_class).end():]  # Имя
        forg_class = f"forg_{num_class}{name}"  # Поддельный класс
        gen_embs = embeddings_dict[gen_class]
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


def compute_metrics_from_evaluation(evaluation_results, threshold):
    """
    Собирает расстояния из evaluate() в единый массив и считает метрики.
    :param evaluation_results: Результат evaluate(), содержащий три списка расстояний
    :param threshold: Порог принятия решения
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

    return calculate_metrics(np.array(all_distances), np.array(all_labels), threshold)


# Основной цикл
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели
model, train_transform, test_transform = create_model(model_name="resnet34",
                                                      embedding_size=EMBEDDING_SIZE,
                                                      pretrained_path="trained_models/resnet34_trained_march_3_and_march_3_test-real_with_all_forged_epoch_50.pth",
                                                      device=device)

# Оптимизатор и потери
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
l2_distance = PairwiseDistance(p=2)

"""
train_dataset, _, _, _ = load_dataset_with_train_test_transforms(
    dataset_path, train_transform=train_transform, test_ratio=0.2,
    test_transform=test_transform, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, all_shuffle=False
)
"""

if second_dataset_path:
    train_dataset, _ = load_dataset_full(
        dataset_path, transform=train_transform, batch_size=BATCH_SIZE, shuffle=False
    )

    test_dataset, _ = load_dataset_full(
        second_dataset_path, transform=test_transform, batch_size=BATCH_SIZE, shuffle=False
    )
else:
    train_dataset, test_dataset, _, _ = load_dataset_with_train_test_transforms(
        dataset_path, train_transform=train_transform, test_ratio=0.2,
        test_transform=test_transform, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, all_shuffle=False
    )

use_hard_negatives = False  # Переключатель для semi-hard/hard negatives
current_date = f"{datetime.datetime.now().day}-{datetime.datetime.now().month}"


"""
if use_semihard_negatives:
    logging.basicConfig(filename=f'logs/semi_{model.name}_dataset_{dataset_name}_{current_date}.log', level=logging.INFO)
else:
    logging.basicConfig(filename=f'logs/{model.name}_dataset_{dataset_name}_{current_date}.log', level=logging.INFO)
"""
logging.basicConfig(filename=f'logs/TESTING_new_metrics_combined_tripl_{model.name}_dataset_{dataset_name}_{current_date}.log',
                    level=logging.INFO)


logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; RANDOM_STATE={RANDOM_STATE}; DATASET={dataset_path}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: combined"
            f"TRANSFORMS:"
            f"Train={train_transform}"
            f"Test={test_transform}")
            # f"Triplets: {'SEMI-HARD' if use_hard_negatives else 'HARD'}")

best_test_metrics = {
    'f1_score': 0.0,
    'roc_auc': 0.0,
    'average_precision': 0.0,
    'epoch': -1,
    'true_negative': 0,
    'false_positive': 0,
    'false_negative': 0,
    'true_positive': 0,
    'threshold': 0.0
}

# Оценка на тестовом наборе
if second_dataset_path:
    # distances, labels = evaluate(model, test_dataset, test_dataset.dataset.base.class_to_idx, l2_distance)
    all_distances = evaluate(model, test_dataset, test_dataset.dataset.base.class_to_idx, l2_distance)
else:
    # distances, labels = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx, l2_distance)
    all_distances = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx, l2_distance)

for key, dists in all_distances.items():
    print(
        f"{key}: mean={np.mean(dists):.4f}, median={np.median(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")

# Вычисление метрик для разных порогов
results = [compute_metrics_from_evaluation(all_distances, t) for t in np.arange(0.1, 800, 0.25)]
best_result = max(results, key=lambda x: x['f1_score'])
if second_dataset_path:
    logger.info(f"First dataset: {dataset_path} ; Second dataset was: {second_dataset_path}")
logger.info(f"Best Threshold: {best_result['threshold']:.1f}, F1: {best_result['f1_score']:.4f}, "
            f"ROC-AUC: {best_result['roc_auc']:.4f}, AP: {best_result['average_precision']:.4f}")
logger.info(f"TN: {best_result['true_negative']}, FP: {best_result['false_positive']}, "
            f"FN: {best_result['false_negative']}, TP: {best_result['true_positive']}"
            f"Accuracy: {best_result['accuracy']} Precision: {best_result['precision']}; "
            f"Recall: {best_result['recall']}")
if best_result['f1_score'] > best_test_metrics['f1_score']:
    best_test_metrics.update({
        'f1_score': best_result['f1_score'],
        'roc_auc': best_result['roc_auc'],
        'average_precision': best_result['average_precision'],
        'epoch': 0,
        'true_negative': best_result['true_negative'],
        'false_positive': best_result['false_positive'],
        'false_negative': best_result['false_negative'],
        'true_positive': best_result['true_positive'],
        'threshold': best_result['threshold']
    })