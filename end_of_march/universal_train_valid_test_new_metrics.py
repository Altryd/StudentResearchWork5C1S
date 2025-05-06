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
                     create_model, calculate_metrics, save_metrics, load_metrics, plot_and_save_curves)
import logging
import time
import datetime

logger = logging.getLogger(__name__)

# Глобальные константы
EMBEDDING_SIZE = 128  # TODO!!!
MARGIN = 1.0
START_EPOCH = 0
EPOCHS = 50
BATCH_SIZE = 75
TEST_VAL_BATCH_SIZE = 75
RANDOM_STATE = 111
dataset_path = "datasets/CEDAR_refactored"
dataset_name = dataset_path.split("/")[-1]
SAVE_MODEL_EVERY_N_EPOCHS = 10



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
def train_epoch(model, dataloader, loss_fn, optimizer, use_hard=True, best_threshold=0.0):
    model.train()
    total_loss = 0
    num_valid_triplets = 0
    all_distances = {"pos-pos": [], "gen-forg": [], "gen-imp": []}
    # all_labels = []

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
                                # all_labels.append(1)
                            elif "forg" in label_to_compare:
                                all_distances["gen-forg"].append(dist)
                                # all_labels.append(0)
                            else:
                                all_distances["gen-imp"].append(dist)
                                # all_labels.append(0)

            if batch_idx % 2 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Valid triplets: {len(triplets)}")

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

    train_metrics = compute_metrics_from_evaluation(all_distances, best_threshold, epoch=None)
    train_epoch.last_metrics = train_metrics
    avg_loss = total_loss / num_valid_triplets if num_valid_triplets > 0 else 0
    return avg_loss, num_valid_triplets, train_metrics


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


def compute_metrics_from_evaluation(evaluation_results, threshold, epoch=None):
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
    # print(f"compute_metrics_from_evaluation; all_distances: {all_distances}; all labels: {all_labels}")
    #logger.info(f"Pairs distribution: pos-pos={len(evaluation_results['pos-pos'])}, "
    #            f"gen-forg={len(evaluation_results['gen-forg'])}, gen-imp={len(evaluation_results['gen-imp'])}")

    # Если списки пусты, возвращаем заглушки, но логируем проблему
    if not all_distances or not all_labels:
        logger.warning("No valid pairs found for metrics computation. Returning placeholder metrics.")
        return {
            'f1_score': 0.0,  # Будет заменено в train_epoch, если возможно
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0,
            'true_positive': 0,
            'threshold': threshold
        }

    return calculate_metrics(np.array(all_distances), np.array(all_labels), threshold, epoch)


# Основной цикл
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели
model, train_transform, test_transform = create_model(model_name="resnet34",
                                                      embedding_size=EMBEDDING_SIZE,
                                                      pretrained_path=None,
                                                      device=device)

plot_save_curves_dir = f"plots/{model.name}_{dataset_name}"

# Оптимизатор и потери
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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

use_hard_negatives = False  # Переключатель для semi-hard/hard negatives
current_date = f"{datetime.datetime.now().day}-{datetime.datetime.now().month}"

logging.basicConfig(filename=f'logs/new_metrics_combined_tripl_{model.name}_dataset_{dataset_name}_{current_date}.log',
                    level=logging.INFO)


logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; RANDOM_STATE={RANDOM_STATE}; DATASET={dataset_path}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: combined"
            f"TRANSFORMS:"
            f"Train={train_transform}"
            f"Test={test_transform}"
            f"\nTRAIN TEST VALIDATION SPLIT")
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

print(f"Параметры модели: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн")
print(f"Память под веса: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2:.2f} МБ")
# Инициализация списков для хранения метрик
# Инициализация метрик
metrics_file = f'metrics/metrics_{model.name}_dataset_{dataset_name}_{current_date}.json'
metrics = load_metrics(metrics_file)

# Убедимся, что списки метрик имеют правильную длину
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
    logger.warning(f"Loaded metrics contain {len(train_losses)} epochs, but START_EPOCH={START_EPOCH}. Truncating metrics.")
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
                                         use_hard=use_hard_negatives, best_threshold=best_threshold)
    logger.info(f"Avg Loss: {avg_loss:.4f}, Valid Triplets: {num_triplets}")
    print(f"train_metrics before: {train_metrics}")
    if train_metrics['f1_score'] <= 0.001 and num_triplets == 0:
        logger.info("Using last known train metrics due to zero triplets.")
        train_metrics = last_train_metrics.copy()
    else:
        last_train_metrics = train_metrics.copy()
    print(f"train_metrics after: {train_metrics}")
    if num_triplets < 10 and not use_hard_negatives:
        use_hard_negatives = True
        logger.info(f"The number of triplets is less than 10. Start including hard triplets")
    elif num_triplets < 10:
        MARGIN += 0.5
        logger.info(f"The number of triplets is less than 10. improved Margin to {MARGIN}")
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN)  # TODO: ?
    train_losses.append(avg_loss)
    train_f1_scores.append(train_metrics['f1_score'])
    train_roc_aucs.append(train_metrics['roc_auc'])
    train_pr_aucs.append(train_metrics['average_precision'])


    print("Приступаем к валидации")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")
    # Оценка на valid наборе
    distances = evaluate(model, val_dataset, val_dataset.dataset.base.dataset.class_to_idx, l2_distance)
    print("VALIDATION")
    for key, dists in distances.items():
        print(
            f"{key}: mean={np.mean(dists):.4f}, median={np.median(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")
    print("после evaluate")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    # Вычисление метрик для разных порогов
    results = [compute_metrics_from_evaluation(distances, t) for t in np.arange(0.1, 30, 0.25)]
    best_result = max(results, key=lambda x: x['f1_score'])
    torch.cuda.empty_cache()
    val_f1_scores.append(best_result['f1_score'])
    val_roc_aucs.append(best_result['roc_auc'])
    val_pr_aucs.append(best_result['average_precision'])

    logger.info(f"[VALIDATION] "
                f"Best Threshold for epoch {epoch}: {best_result['threshold']:.1f}, F1: {best_result['f1_score']:.4f}, "
                f"ROC-AUC: {best_result['roc_auc']:.4f}, AP: {best_result['average_precision']:.4f}")
    logger.info(f"TN: {best_result['true_negative']}, FP: {best_result['false_positive']}, "
                f"FN: {best_result['false_negative']}, TP: {best_result['true_positive']}")

    best_threshold = best_result['threshold']
    print("Приступаем к тестам")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")
    test_distances = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx,
                                           l2_distance)
    print("TEST")
    for key, dists in test_distances.items():
        print(
            f"{key}: mean={np.mean(dists):.4f}, median={np.median(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")
    print("после evaluate")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GiB")

    test_result = compute_metrics_from_evaluation(test_distances, best_threshold)
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
        plot_and_save_curves(metrics, epoch, dataset_name, epoch_label=epoch, save_dir=f"plots/{model.name}")

        # Сохраняем модель
        torch.save(model.state_dict(), f"trained_models/{model.name}_trained_{dataset_name}_epoch_{epoch}.pth")
        logger.info(f"Saved model and metrics at epoch {epoch}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Прошло времени  (секунды): {elapsed_time}')
    gc.collect()

logger.info(f"BEST METRICS:\n{best_test_metrics}")

# Сохранение финальной модели
torch.save(model.state_dict(), f"trained_models/{model.name}_trained_{dataset_name}_epoch_{EPOCHS}.pth")

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
                     save_dir=f"plots/{model.name}")  # Финальные графики без epoch_label