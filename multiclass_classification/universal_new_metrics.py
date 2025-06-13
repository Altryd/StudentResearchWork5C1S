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
from utility import (load_dataset_with_train_test_transforms, load_dataset_full, create_model,
                     calculate_metrics, save_metrics, load_metrics, plot_and_save_curves, select_triplets, evaluate,
                     compute_metrics_from_evaluation, train_epoch)
import logging
import time
import datetime

logger = logging.getLogger(__name__)


# Глобальные константы
EMBEDDING_SIZE = 128
MARGIN = 1.0
START_EPOCH = 0
EPOCHS = 50
BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
RANDOM_STATE = 111
dataset_path = "datasets/march_1_full"
# second_dataset_path = "datasets/march_3_test-real_with_all_forged"
second_dataset_path = None
dataset_name = dataset_path.split("/")[-1]
if second_dataset_path:
    second_dataset_name = second_dataset_path.split("/")[-1]
SAVE_MODEL_EVERY_N_EPOCHS = 20




# Основной цикл
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели
model, train_transform, test_transform = create_model(model_name="vit_b_32",
                                                      embedding_size=EMBEDDING_SIZE,
                                                      pretrained_path=None,
                                                      device=device)

plot_save_curves_dir = f"plots/{model.name}_{dataset_name}"

# Оптимизатор и потери
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
l2_distance = PairwiseDistance(p=2)

# Загрузка данных
if second_dataset_path:
    train_dataset, _ = load_dataset_full(
        dataset_path, transform=train_transform, batch_size=BATCH_SIZE, shuffle=False
    )

    test_dataset, _ = load_dataset_full(
        second_dataset_path, transform=test_transform, batch_size=TEST_BATCH_SIZE, shuffle=False
    )
else:
    train_dataset, test_dataset, _, _ = load_dataset_with_train_test_transforms(
        dataset_path, train_transform=train_transform, test_ratio=0.2,
        test_transform=test_transform, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, all_shuffle=False
    )

use_hard_negatives = False  # Переключатель для semi-hard/hard negatives
current_date = f"{datetime.datetime.now().day}-{datetime.datetime.now().month}"

logging.basicConfig(filename=f'logs/new_metrics_combined_tripl_{model.name}_dataset_{dataset_name}_{current_date}.log',
                    level=logging.INFO)

logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; RANDOM_STATE={RANDOM_STATE}; DATASET={dataset_path}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: combined\n"
            f"TRANSFORMS:\n"
            f"Train={train_transform}\n"
            f"Test={test_transform}\n")

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

# Инициализация метрик
metrics_file = f'metrics/metrics_{model.name}_dataset_{dataset_name}_{current_date}.json'

# Загрузка существующих метрик или создание новых
metrics = load_metrics(metrics_file)

# Инициализация списков для метрик
train_losses = metrics['train_losses']
train_f1_scores = metrics['train_f1_scores']
train_roc_aucs = metrics['train_roc_aucs']
train_pr_aucs = metrics['train_pr_aucs']
val_f1_scores = metrics['val_f1_scores']
val_roc_aucs = metrics['val_roc_aucs']
val_pr_aucs = metrics['val_pr_aucs']

# Храним последние известные метрики
last_train_metrics = {
    'f1_score': train_f1_scores[-1] if train_f1_scores else 0.0,
    'roc_auc': train_roc_aucs[-1] if train_roc_aucs else 0.0,
    'average_precision': train_pr_aucs[-1] if train_pr_aucs else 0.0
}

# Проверка на согласованность метрик и START_EPOCH
if len(train_losses) > START_EPOCH:
    logger.warning(
        f"Loaded metrics contain {len(train_losses)} epochs, but START_EPOCH={START_EPOCH}. Truncating metrics.")
    metrics = {
        'train_losses': train_losses[:START_EPOCH],
        'train_f1_scores': train_f1_scores[:START_EPOCH],
        'train_roc_aucs': train_roc_aucs[:START_EPOCH],
        'train_pr_aucs': train_pr_aucs[:START_EPOCH],
        'val_f1_scores': val_f1_scores[:START_EPOCH],
        'val_roc_aucs': val_roc_aucs[:START_EPOCH],
        'val_pr_aucs': val_pr_aucs[:START_EPOCH]
    }
    train_losses = metrics['train_losses']
    train_f1_scores = metrics['train_f1_scores']
    train_roc_aucs = metrics['train_roc_aucs']
    train_pr_aucs = metrics['train_pr_aucs']
    val_f1_scores = metrics['val_f1_scores']
    val_roc_aucs = metrics['val_roc_aucs']
    val_pr_aucs = metrics['val_pr_aucs']

best_threshold = 0.0

for epoch in range(START_EPOCH, EPOCHS):
    torch.cuda.empty_cache()
    start_time = time.time()
    logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"epoch: {epoch}")
    # Обучение
    avg_loss, num_triplets, train_metrics = train_epoch(model, train_dataset, triplet_loss, optimizer,
                                                        use_hard=use_hard_negatives, best_threshold=best_threshold,
                                                        logger=logger, MARGIN=MARGIN, device=device)
    logger.info(f"Avg Loss: {avg_loss:.4f}, Valid Triplets: {num_triplets}")

    print(f"train_metrics before: {train_metrics}")
    if train_metrics['f1_score'] <= 0.001 and num_triplets == 0:
        logger.info("Using last known train metrics due to zero triplets.")
        train_metrics = last_train_metrics.copy()
    else:
        last_train_metrics = train_metrics.copy()
    print(f"train_metrics after: {train_metrics}")
    # TODO: test that:
    """
    if num_triplets < 10 and use_hard_negatives and MARGIN > 10.0:
        logger.info(f"The number of triplets is {num_triplets}, MARGIN is {MARGIN}. "
                    f"Stopping the training to avoid overfitting")
        break
    """
    if num_triplets < 10 and not use_hard_negatives:
        use_hard_negatives = True
        logger.info(f"The number of triplets is less than 10. Start including hard triplets")
    elif num_triplets < 10:
        MARGIN += 0.5
        logger.info(f"The number of triplets is less than 10. improved Margin to {MARGIN}")
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN)

    # Сохраняем метрики обучения
    train_losses.append(avg_loss)
    train_f1_scores.append(train_metrics['f1_score'])
    train_roc_aucs.append(train_metrics['roc_auc'])
    train_pr_aucs.append(train_metrics['average_precision'])

    print("Приступаем к тестам")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    # Оценка на тестовом наборе
    if second_dataset_path:
        test_distances = evaluate(model, test_dataset, test_dataset.dataset.base.class_to_idx, l2_distance,
                                  device=device)
    else:
        test_distances = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx, l2_distance,
                                  device=device)

    print("TEST")
    for key, dists in test_distances.items():
        print(
            f"{key}: mean={np.mean(dists):.4f}, median={np.median(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")
    print("после evaluate")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    # Вычисление метрик для разных порогов
    results = [compute_metrics_from_evaluation(test_distances, t, epoch=epoch, logger=logger) for t in np.arange(0.1, 30, 0.25)]
    best_result = max(results, key=lambda x: x['f1_score'])
    torch.cuda.empty_cache()

    val_f1_scores.append(best_result['f1_score'])
    val_roc_aucs.append(best_result['roc_auc'])
    val_pr_aucs.append(best_result['average_precision'])

    logger.info(f"[TEST] "
                f"Best Threshold for epoch {epoch}: {best_result['threshold']:.1f}, F1: {best_result['f1_score']:.4f}, "
                f"ROC-AUC: {best_result['roc_auc']:.4f}, AP: {best_result['average_precision']:.4f}")
    logger.info(f"TN: {best_result['true_negative']}, FP: {best_result['false_positive']}, "
                f"FN: {best_result['false_negative']}, TP: {best_result['true_positive']}")

    best_threshold = best_result['threshold']

    if best_result['f1_score'] > best_test_metrics['f1_score']:
        best_test_metrics.update({
            'f1_score': best_result['f1_score'],
            'roc_auc': best_result['roc_auc'],
            'average_precision': best_result['average_precision'],
            'epoch': epoch,
            'true_negative': best_result['true_negative'],
            'false_positive': best_result['false_positive'],
            'false_negative': best_result['false_negative'],
            'true_positive': best_result['true_positive'],
            'threshold': best_threshold
        })

    if (epoch % SAVE_MODEL_EVERY_N_EPOCHS == 0 and epoch > 0) or epoch == EPOCHS - 1:
        # Сохраняем метрики в JSON
        metrics = {
            'train_losses': train_losses,
            'train_f1_scores': train_f1_scores,
            'train_roc_aucs': train_roc_aucs,
            'train_pr_aucs': train_pr_aucs,
            'val_f1_scores': val_f1_scores,
            'val_roc_aucs': val_roc_aucs,
            'val_pr_aucs': val_pr_aucs
        }
        save_metrics(metrics, metrics_file)

        # Создаем и сохраняем графики
        plot_and_save_curves(metrics, epoch, dataset_name, epoch_label=epoch, save_dir=f"{plot_save_curves_dir}")

        # Сохраняем модель
        if second_dataset_path:
            torch.save(model.state_dict(),
                       f"trained_models/{model.name}_trained_{dataset_name}_and_{second_dataset_name}_epoch_{epoch}.pth")
        else:
            torch.save(model.state_dict(),
                       f"trained_models/{model.name}_trained_{dataset_name}_epoch_{epoch}.pth")
        logger.info(f"Saved model and metrics at epoch {epoch}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Прошло времени (секунды): {elapsed_time}')
    gc.collect()

logger.info(f"BEST METRICS:\n{best_test_metrics}")

# Сохранение финальной модели
if second_dataset_path:
    torch.save(model.state_dict(),
               f"trained_models/{model.name}_trained_{dataset_name}_and_{second_dataset_name}_final.pth")
else:
    torch.save(model.state_dict(),
               f"trained_models/{model.name}_trained_{dataset_name}_final.pth")

# Сохранение финальных графиков (без номера эпохи)
metrics = {
    'train_losses': train_losses,
    'train_f1_scores': train_f1_scores,
    'train_roc_aucs': train_roc_aucs,
    'train_pr_aucs': train_pr_aucs,
    'val_f1_scores': val_f1_scores,
    'val_roc_aucs': val_roc_aucs,
    'val_pr_aucs': val_pr_aucs
}
plot_and_save_curves(metrics, EPOCHS - 1, dataset_name,
                     save_dir=f"{plot_save_curves_dir}")  # Финальные графики без epoch_label