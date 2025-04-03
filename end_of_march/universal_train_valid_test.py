import datetime
import re

import timm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import itertools

from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights

from used_transforms import *
import torch
import torch.nn as nn
from torch.nn import PairwiseDistance
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader
from TransformDataset import TransformDataset
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)


from utility import load_dataset_with_train_test_transforms, load_dataset_with_train_test_valid_transforms

EMBEDDING_SIZE = 128
MARGIN = 1.0
EPOCHS = 75
BATCH_SIZE = 264
TEST_VAL_BATCH_SIZE = -1
RANDOM_STATE = 111
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_metrics(all_distances, all_labels, threshold):
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


# создание модели
def create_model(model_name="resnet34", embedding_size=EMBEDDING_SIZE, pretrained_path=None, device="cpu"):
    train_transform = None
    test_transform = None
    if model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        # Заменяем последний слой на слой с нужным размером эмбеддинга
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, embedding_size)
        train_transform = train_transform_resnet_v34
        test_transform = test_transform_resnet_v34
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, embedding_size)
        train_transform = train_transform_resnet_v101
        test_transform = test_transform_resnet_v101
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=EMBEDDING_SIZE,
                                                               bias=True))
        train_transform = train_transform_efficient_net_b0
        test_transform = test_transform_efficient_net_b0
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        # TODO: model.heads = nn.Linear(768, out_features=EMBEDDING_SIZE)  # Новая размерность, например, 128
        # Это может быть полезно
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=EMBEDDING_SIZE,
                                                               bias=True))
        """
        train_transform = train_transform_vit_b_32
        test_transform = test_transform_vit_b_32
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        # TODO: model.heads = nn.Linear(768, out_features=EMBEDDING_SIZE)  # Новая размерность, например, 128
        # Это может быть полезно
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(in_features=num_ftrs,
                                                               out_features=EMBEDDING_SIZE,
                                                               bias=True))
        """
        train_transform = train_transform_vit_b_16
        test_transform = test_transform_vit_b_16
    elif model_name == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=True,
                                  num_classes=EMBEDDING_SIZE)
        train_transform = train_transform_resnet_inception
        test_transform = test_transform_resnet_inception
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    else:
        model.name = model_name
    return model.to(device), train_transform, test_transform


# Функция для выбора триплетов (online triplet mining)
def select_triplets(embeddings, labels, use_semihard=False, margin=MARGIN):
    triplets = []
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

    for pos_class in unique_labels:
        pos_indices = np.where(labels == pos_class)[0]
        neg_indices = np.where(labels != pos_class)[0]

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
                    neg_idx = valid_neg_indices[np.argmin(neg_dists[valid_mask].detach().numpy())]
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

        if batch_idx % 10 == 0:  # Уменьшил частоту логов для скорости
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Valid triplets: {len(triplets)}")

    avg_loss = total_loss / num_valid_triplets if num_valid_triplets > 0 else 0
    return avg_loss, num_valid_triplets


def evaluate(model, dataloader, class_to_idx):
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
                opp_class = f"forged_{num_class}{name}"
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


model, train_transform, test_transform = create_model(model_name="resnet34", device=device, embedding_size=EMBEDDING_SIZE, pretrained_path=None,)
#resnet34 = torch.load("resnet34_CEDAR_refactored_march_14-03_margin0.8_hard_neg_emb_128_.pth", weights_only=False,
#                       map_location=torch.device('cpu'))
# resnet34.to(device)
# resnet34.name = "resnet34"
# additional_name = "march_22-03_margin1.0_hard_neg_emb_128_epoch_75"
use_semihard_negatives = False
print(device)
logger.info(device)
# num_classes = 2
# Заменяем последний слой так, чтобы была классификация только для двух классов
# resnet34.fc = nn.Linear(resnet34.fc.in_features, EMBEDDING_SIZE)

# criterion = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
l2_distance = PairwiseDistance(p=2)

dataset_path = "datasets/CEDAR_refactored"
dataset_name = dataset_path.split("/")[-1]

current_date = f"{datetime.datetime.now().day}-{datetime.datetime.now().month}"

if use_semihard_negatives:
    logging.basicConfig(filename=f'logs/semi_{model.name}_dataset_{dataset_name}_{current_date}.log', level=logging.INFO)
else:
    logging.basicConfig(filename=f'logs/{model.name}_dataset_{dataset_name}_{current_date}.log', level=logging.INFO)


logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; RANDOM_STATE={RANDOM_STATE}; DATASET={dataset_path}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: {'SEMI-HARD' if use_semihard_negatives else 'HARD'}")

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

for epoch in range(EPOCHS):
    logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Обучение
    avg_loss, num_triplets = train_epoch(model, train_dataset, triplet_loss, optimizer,
                                         use_semihard=use_semihard_negatives)
    logger.info(f"Avg Loss: {avg_loss:.4f}, Valid Triplets: {num_triplets}")

    # Оценка на valid наборе
    distances, labels = evaluate(model, val_dataset, val_dataset.dataset.base.dataset.class_to_idx)

    # Вычисление метрик для разных порогов
    results = [calculate_metrics(distances, labels, t) for t in np.arange(0.1, 6.0, 0.5)]
    best_result = max(results, key=lambda x: x['f1_score'])
    logger.info(f"Best Threshold for epoch {epoch}: {best_result['threshold']:.1f}, F1: {best_result['f1_score']:.4f}, "
          f"ROC-AUC: {best_result['roc_auc']:.4f}, AP: {best_result['average_precision']:.4f}")
    logger.info(f"TN: {best_result['true_negative']}, FP: {best_result['false_positive']}, "
          f"FN: {best_result['false_negative']}, TP: {best_result['true_positive']}")

    best_threshold = best_result['threshold']
    test_distances, test_labels = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx)
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
    if epoch % 25 == 0:
        torch.save(model.state_dict(), f"{model.name}_trained_CEDAR_epoch_{epoch}.pth")

logger.info(f"BEST METRICS:\n{best_test_metrics}")
# Сохранение модели (опционально)
torch.save(model.state_dict(), f"{model.name}_trained_CEDAR_epoch_{EPOCHS}.pth")

