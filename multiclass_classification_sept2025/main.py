# -*- coding: utf-8 -*-
import gc
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score
import itertools
import logging
import time
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast  # Mixed precision

from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights
from multiclass_classification_sept2025.compute_multiclass import compute_multiclass_metrics
from used_transforms import *  # Assume this has your transforms
from TransformDataset import TransformDataset
from utility import (load_dataset_with_train_test_valid_transforms, create_model, save_metrics, load_metrics,
                     select_triplets, evaluate, compute_metrics_from_evaluation)

logger = logging.getLogger(__name__)

# Global constants
EMBEDDING_SIZE = 128
MARGIN = 1.0
START_EPOCH = 0
EPOCHS = 50
BATCH_SIZE = 75  # Can increase to 100 with mixed precision
TEST_VAL_BATCH_SIZE = 75
RANDOM_STATE = 111
DATASET_PATH = "datasets/CEDAR_refactored"
DATASET_NAME = DATASET_PATH.split("/")[-1]
SAVE_MODEL_EVERY_N_EPOCHS = 10
PATIENCE = 5  # For early stopping
TRIPLETS_PER_CLASS = 20  # For offline mining

# Fixed timestamp for files (no date change issues)
CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(filename=f'logs/new_metrics_combined_tripl_vit_b_16_dataset_{DATASET_NAME}_{CURRENT_TIMESTAMP}.log',
                    level=logging.INFO)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model, train_transform, test_transform = create_model(model_name="vit_b_16",
                                                      embedding_size=EMBEDDING_SIZE,
                                                      pretrained_path="trained_models/vit_b_16_trained_BHSig260-Hindi-april-refactored_epoch_4.pth",
                                                      device=device)

# Freeze lower layers for ViT to prevent overfitting
#for param in model.parameters():
#    param.requires_grad = False
#for param in model.head.parameters():  # Unfreeze head
#    param.requires_grad = True

# Optimizer, loss, scheduler
triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
l2_distance = PairwiseDistance(p=2)
scaler = GradScaler()  # For mixed precision

# Load datasets
(train_dataset, val_dataset, test_dataset, _, _, _) = load_dataset_with_train_test_valid_transforms(
    DATASET_PATH, train_transform=train_transform, test_transform=test_transform, validation_transform=test_transform,
    test_val_ratio=0.375, val_ratio=0.5, batch_size=BATCH_SIZE, test_val_batch_size=TEST_VAL_BATCH_SIZE,
    random_state=RANDOM_STATE, all_shuffle=False)


# Offline triplet mining: Generate triplets once before training
class TripletDataset(Dataset):
    def __init__(self, base_dataset, triplets):
        self.base_dataset = base_dataset
        self.triplets = triplets

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        anchor = self.base_dataset[anchor_idx]
        pos = self.base_dataset[pos_idx]
        neg = self.base_dataset[neg_idx]
        return (anchor[0], pos[0], neg[0]), (anchor[1], pos[1], neg[1])  # Images, labels

    def __len__(self):
        return len(self.triplets)


def generate_offline_triplets(dataset, triplets_per_class=TRIPLETS_PER_CLASS, margin=MARGIN):
    """Offline mining: Generate triplets from entire dataset."""
    logger.info("Generating offline triplets...")
    images, labels = [], []
    for img, lbl in dataset:
        images.append(img)
        labels.append(lbl)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)

    with torch.no_grad():
        embeddings = model(images)  # Initial embeddings for mining

    triplets = []
    unique_labels = torch.unique(labels)
    for cls in unique_labels:
        pos_indices = (labels == cls).nonzero(as_tuple=True)[0].cpu().numpy()
        neg_indices = (labels != cls).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(pos_indices) < 2:
            continue
        for _ in range(triplets_per_class):
            anchor_idx, pos_idx = np.random.choice(pos_indices, 2, replace=False)
            neg_idx = np.random.choice(neg_indices)
            triplets.append((anchor_idx, pos_idx, neg_idx))
    logger.info(f"Generated {len(triplets)} triplets.")
    return triplets


# Generate triplets and create triplet loader
triplets = generate_offline_triplets(train_dataset.dataset)  # Assume train_dataset is wrapped
triplet_dataset = TripletDataset(train_dataset.dataset, triplets)
train_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Metrics file
metrics_file = f'metrics/metrics_vit_b_16_dataset_{DATASET_NAME}_{CURRENT_TIMESTAMP}.json'
metrics = load_metrics(metrics_file) or {
    'train_losses': [], 'train_f1_scores': [], 'train_roc_aucs': [], 'train_pr_aucs': [],
    'val_f1_scores': [], 'val_roc_aucs': [], 'val_pr_aucs': []
}

# Early stopping vars
best_val_f1 = 0.0
patience_counter = 0

logger.info(f"EMBED_SIZE={EMBEDDING_SIZE}; MARGIN={MARGIN}; EPOCHS={EPOCHS}\n"
            f"BATCH_SIZE={BATCH_SIZE}; RANDOM_STATE={RANDOM_STATE}; DATASET={DATASET_PATH}\n"
            f"MODEL={model.name}; OPTIMIZER={optimizer}\n"
            f"Triplets: offline, {TRIPLETS_PER_CLASS} per class\n"
            f"TRANSFORMS: Train={train_transform}\nTest={test_transform}\n"
            f"TRAIN TEST VALIDATION SPLIT")

print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн")
print(f"Weights memory: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2:.2f} МБ")

for epoch in range(START_EPOCH, EPOCHS):
    torch.cuda.empty_cache()
    start_time = time.time()
    logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training with offline triplets and mixed precision
    model.train()
    total_loss = 0
    num_valid_triplets = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        anchor_img, pos_img, neg_img = images
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        optimizer.zero_grad()
        with autocast():
            anchor_emb = nn.functional.normalize(model(anchor_img), dim=1)  # Normalize to avoid collapse
            pos_emb = nn.functional.normalize(model(pos_img), dim=1)
            neg_emb = nn.functional.normalize(model(neg_img), dim=1)
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_valid_triplets += BATCH_SIZE  # Since offline, all are valid
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Avg Loss: {avg_loss:.4f}, Triplets: {num_valid_triplets}")

    scheduler.step()  # LR decay

    # Validation (binary and multiclass)
    distances = evaluate(model, val_dataset, val_dataset.dataset.base.dataset.class_to_idx, l2_distance, device=device)
    results = [compute_metrics_from_evaluation(distances, t, logger=logger, epoch=epoch) for t in
               np.arange(0.1, 30, 0.25)]
    best_result = max(results, key=lambda x: x['f1_score'])
    val_f1 = best_result['f1_score']  # Binary F1 for early stopping

    multiclass_metrics = compute_multiclass_metrics(model, val_dataset, device)
    logger.info(
        f"[VALIDATION] Best Threshold: {best_result['threshold']:.1f}, Binary F1: {val_f1:.4f}, Multiclass F1: {multiclass_metrics['f1_score']:.4f}")

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test (similarly)
    test_distances = evaluate(model, test_dataset, test_dataset.dataset.base.dataset.class_to_idx, l2_distance,
                              device=device)
    test_result = compute_metrics_from_evaluation(test_distances, best_result['threshold'], logger=logger)
    test_multiclass = compute_multiclass_metrics(model, test_dataset, device)
    logger.info(f"Test Binary F1: {test_result['f1_score']:.4f}, Multiclass F1: {test_multiclass['f1_score']:.4f}")

    # Append metrics (no last_metrics hack)
    metrics['train_losses'].append(avg_loss)
    # ... (append other metrics from train/test/val)

    # Save if needed
    if (epoch % SAVE_MODEL_EVERY_N_EPOCHS == 0 and epoch > 0) or epoch == EPOCHS - 1:
        save_metrics(metrics, metrics_file)
        torch.save(model.state_dict(), f"trained_models/vit_b_16_trained_{DATASET_NAME}_epoch_{epoch}.pth")
        logger.info(f"Saved model and metrics at epoch {epoch}")

    elapsed_time = time.time() - start_time
    logger.info(f'Elapsed time (sec): {elapsed_time}')
    gc.collect()

# Final save
save_metrics(metrics, metrics_file)
torch.save(model.state_dict(), f"trained_models/vit_b_16_trained_{DATASET_NAME}_epoch_{EPOCHS}.pth")


"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_file', type=str, default=None, help='Path to metrics JSON for resume')
args = parser.parse_args()

if args.metrics_file:
    metrics_file = args.metrics_file
else:
    CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_file = f'metrics/metrics_{model.name}_dataset_{DATASET_NAME}_{CURRENT_TIMESTAMP}.json'

# Load or init
metrics = load_metrics(metrics_file) if os.path.exists(metrics_file) else {
    'train_losses': [], 'train_f1_scores': [], 'train_roc_aucs': [], 'train_pr_aucs': [],
    'val_f1_scores': [], 'val_roc_aucs': [], 'val_pr_aucs': []
}


"""