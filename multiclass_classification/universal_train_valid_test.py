# -*- coding: utf-8 -*-
import gc
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

from used_transforms import *
from TransformDataset import TransformDataset
from utility import (load_dataset_with_train_test_transforms, load_dataset_with_train_test_valid_transforms,
                     create_model, calculate_metrics)
import logging
import time
import datetime

logger = logging.getLogger(__name__)

# Глобальные константы
EMBEDDING_SIZE = 128
MARGIN = 1.0
EPOCHS = 75
BATCH_SIZE = 264
TEST_VAL_BATCH_SIZE = -1
RANDOM_STATE = 111
dataset_path = "datasets/CEDAR_refactored"
dataset_name = dataset_path.split("/")[-1]
SAVE_MODEL_EVERY_N_EPOCHS = 10



# Функция для выбора триплетов (online triplet mining)
def select_triplets(embeddings, labels, use_semihard=False, margin=MARGIN):
    with torch.no_grad():
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

                if use_semihard:
                    # Semi-hard: pos_dist < neg_dist < pos_dist + margin
                    valid_mask = (pos_dist < neg_dists) & (neg_dists < pos_dist + margin)
                    valid_neg_indices = neg_indices[valid_mask.cpu().numpy()]
                    if len(valid_neg_indices) > 0:
                        neg_idx = np.random.choice(valid_neg_indices)
                        triplets.append((anchor_idx, pos_idx, neg_idx))
                else:
                    # Hard: neg_dist < pos_dist
                    valid_mask = neg_dists < pos_dist
                    valid_neg_indices = neg_indices[valid_mask.cpu().numpy()]
                    if len(valid_neg_indices) > 0:
                        neg_idx = valid_neg_indices[np.argmin(neg_dists[valid_mask].cpu().detach().numpy())]
                        triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets


# Функция обучения одной эпохи
def train_epoch(model, dataloader, loss_fn, optimizer, use_semihard=False):
    model.train()
    total_loss = 0
    num_valid_triplets = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Получение эмбеддингов
        embeddings = model(images)

        # Выбор триплетов
        triplets = select_triplets(embeddings, labels, use_semihard=use_semihard)
        print("number of triplets", len(triplets))
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")
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


# Функция оценки на тестовом наборе
def evaluate(model, dataloader, class_to_idx, l2_distance):
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            labels_np = labels.cpu().numpy()

            # Формирование пар
            all_gen = [k for k in class_to_idx.keys() if "gen" in k]
            for gen_class in all_gen:
                num_class = re.findall(r"\d+", gen_class)[0]
                name = gen_class[re.search("\d+", gen_class).end():]
                if "CEDAR" in dataset_name:
                    opp_class = f"forged_{num_class}{name}"
                else:
                    opp_class = f"forg_{num_class}{name}"
                if opp_class not in class_to_idx:
                    raise ValueError(f"Opposite class {opp_class} not found")

                gen_idx = class_to_idx[gen_class]
                forg_idx = class_to_idx[opp_class]

                pos_indices = np.where(labels_np == gen_idx)[0]
                neg_indices = np.where(labels_np == forg_idx)[0]

                # Positive pairs
                for a, b in itertools.combinations(pos_indices, 2):
                    dist = l2_distance(embeddings[a].unsqueeze(0), embeddings[b].unsqueeze(0)).item()
                    all_distances.append(dist)
                    all_labels.append(1)

                # Negative pairs
                for a, b in itertools.product(pos_indices, neg_indices):
                    dist = l2_distance(embeddings[a].unsqueeze(0), embeddings[b].unsqueeze(0)).item()
                    all_distances.append(dist)
                    all_labels.append(0)

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    return all_distances, all_labels


# Основной цикл
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели
model, train_transform, test_transform = create_model(model_name="inception_resnet_v2",
                                                      embedding_size=EMBEDDING_SIZE,
                                                      pretrained_path=None,
                                                      device=device)

# Оптимизатор и потери
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
l2_distance = PairwiseDistance(p=2)

(train_dataset, val_dataset, test_dataset, train_data,
 val_data, test_data) = load_dataset_with_train_test_valid_transforms(dataset_path,
                                                                      train_transform=train_transform,
                                                                      test_transform=test_transform,
                                                                      validation_transform=test_transform,
                                                                      test_val_ratio=0.375, val_ratio=0.5,
                                                                      batch_size=BATCH_SIZE,
                                                                      test_val_batch_size=TEST_VAL_BATCH_SIZE,
                                                                      random_state=RANDOM_STATE,
                                                                      all_shuffle=False)

use_semihard_negatives = True  # Переключатель для semi-hard/hard negatives
current_date = f"{datetime.datetime.now().day}-{datetime.datetime.now().month}"

if use_semihard_negatives:
    logging.basicConfig(filename=f'logs/semi_{model.name}_dataset_{dataset_name}_{current_date}.log',
                        level=logging.INFO)
else:
    logging.basicConfig(filename=f'logs/{model.name}_dataset_{dataset_name}_{current_date}.log', level=logging.INFO)
logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; TEST_VAL_BATCH_SIZE={TEST_VAL_BATCH_SIZE}; "
            f"RANDOM_STATE={RANDOM_STATE}; DATASET={dataset_path}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: {'SEMI-HARD' if use_semihard_negatives else 'HARD'}"
            f"\nTRAIN TEST VALIDATION SPLIT")

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

print(f"Параметры модели: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн")
print(f"Память под веса: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2:.2f} МБ")
for epoch in range(EPOCHS):
    torch.cuda.empty_cache()
    start_time = time.time()
    logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"epoch: {epoch}")
    # Обучение
    avg_loss, num_triplets = train_epoch(model, train_dataset, triplet_loss, optimizer,
                                         use_semihard=use_semihard_negatives)
    logger.info(f"Avg Loss: {avg_loss:.4f}, Valid Triplets: {num_triplets}")

    print("Приступаем к валидации")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")
    # Оценка на valid наборе
    distances, labels = evaluate(model, val_dataset, val_dataset.dataset.base.dataset.class_to_idx, l2_distance)
    print("после evaluate")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    # Вычисление метрик для разных порогов
    results = [calculate_metrics(distances, labels, t) for t in np.arange(0.1, 6.0, 0.5)]
    best_result = max(results, key=lambda x: x['f1_score'])

    logger.info(f"[VALIDATION] "
                f"Best Threshold for epoch {epoch}: {best_result['threshold']:.1f}, F1: {best_result['f1_score']:.4f}, "
                f"ROC-AUC: {best_result['roc_auc']:.4f}, AP: {best_result['average_precision']:.4f}")
    logger.info(f"TN: {best_result['true_negative']}, FP: {best_result['false_positive']}, "
                f"FN: {best_result['false_negative']}, TP: {best_result['true_positive']}")

    best_threshold = best_result['threshold']
    print("Приступаем к тестам")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")
    test_distances, test_labels = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx,
                                           l2_distance)
    print("после evaluate")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    test_result = calculate_metrics(test_distances, test_labels, best_threshold)
    logger.info(f"Test result for epoch {epoch}: F1: {test_result['f1_score']:.4f}, "
                f"ROC-AUC: {test_result['roc_auc']:.4f}, AP: {test_result['average_precision']:.4f}")
    logger.info(f"TN: {test_result['true_negative']}, FP: {test_result['false_positive']}, "
                f"FN: {test_result['false_negative']}, TP: {test_result['true_positive']}")
    if test_result['f1_score'] > best_test_metrics['f1_score']:
        best_test_metrics.update({
            'f1_score': test_result['f1_score'],
            'roc_auc': test_result['roc_auc'],
            'average_precision': test_result['average_precision'],
            'epoch': epoch,
            'true_negative': test_result['true_negative'],
            'false_positive': test_result['false_positive'],
            'false_negative': test_result['false_negative'],
            'true_positive': test_result['true_positive'],
            'threshold': best_threshold
        })
    if epoch % SAVE_MODEL_EVERY_N_EPOCHS == 0 and epoch > 0:
        torch.save(model.state_dict(), f"trained_models/{model.name}_trained_{dataset_name}_epoch_{epoch}.pth")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Прошло времени  (секунды): {elapsed_time}')
    gc.collect()
logger.info(f"BEST METRICS:\n{best_test_metrics}")

# Сохранение модели
torch.save(model.state_dict(), f"trained_models/{model.name}_trained_{dataset_name}_epoch_{EPOCHS}.pth")

