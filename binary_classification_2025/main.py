import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from TransformDataset import TransformDataset
import matplotlib.pyplot as plt

# Словарь моделей: input_size, weights, custom transforms (если нужны)
MODELS = {
    'densenet121': {
        'input_size': 224,
        'weights': 'IMAGENET1K_V1',
        'model_fn': models.densenet121,
        'classifier_fn': lambda in_features: nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    },
    'resnet101': {
        'input_size': 224,
        'weights': 'IMAGENET1K_V2',
        'model_fn': models.resnet101,
        'classifier_fn': lambda in_features: nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    },
    'vit_b_16': {
        'input_size': 224,
        'weights': 'IMAGENET1K_V1',
        'model_fn': models.vit_b_16,
        'classifier_fn': lambda in_features: nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
    },
    'efficientnet_b4': {
        'input_size': 380,
        'weights': 'IMAGENET1K_V1',
        'model_fn': models.efficientnet_b4,
        'classifier_fn': lambda in_features: nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    },
    # другие модели TODO
}


def get_transforms(input_size, grayscale=False):
    mean, std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) if grayscale else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    grayscale_transform = transforms.Grayscale(num_output_channels=3) if grayscale else transforms.Identity()

    train_transform = transforms.Compose([
        grayscale_transform,
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(25),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.15),
        transforms.ToTensor(),
        transforms.GaussianNoise(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        grayscale_transform,
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, test_transform


def load_model(model_name):
    config = MODELS[model_name]
    model = config['model_fn'](weights=config['weights'])
    model.classifier = config['classifier_fn'](model.classifier.in_features) if hasattr(model, 'classifier') else None
    if hasattr(model, 'fc'):  # ResNet
        model.fc = config['classifier_fn'](model.fc.in_features)
    elif hasattr(model, 'heads'):  # ViT
        in_features = model.heads.head.in_features
        model.heads = config['classifier_fn'](in_features)
    return model


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def plot_and_save_curves(metrics_file, save_dir, dataset_name, model_name, fold):
    os.makedirs(save_dir, exist_ok=True)
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    epochs = range(1, len(metrics['train_losses']) + 1)

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on {dataset_name} (Fold {fold + 1}) - Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_{dataset_name}_fold{fold + 1}_loss.png'))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['val_accs'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} on {dataset_name} (Fold {fold + 1}) - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_{dataset_name}_fold{fold + 1}_acc.png'))
    plt.close()


def train_model(model, optimizer, criterion, train_dataset, val_dataset, train_data, val_data,
                scheduler=None, dataset_name="", model_name=None, num_epochs=100, patience=20, fold=0,
                resume=False, checkpoint_dir="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not model_name:
        model_name = getattr(model, 'name', model._get_name())
    print(f"model_name={model_name}, fold={fold + 1}")
    best_acc = 0
    min_val_loss = float('inf')
    counter = 0
    metrics = {'train_losses': [], 'val_losses': [], 'val_accs': [], 'train_accs': []}

    checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{dataset_name}_fold{fold + 1}_checkpoint.pth')


    start_epoch = 0
    if resume and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint['metrics']
        min_val_loss = checkpoint['min_val_loss']
        best_acc = checkpoint['best_acc']
        counter = checkpoint['counter']
        print(f"Возобновление фолда {fold + 1} с эпохи {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        total_correct_train = 0
        for images, labels in train_dataset:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_data)

        model.eval()
        total_correct = 0
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                running_val_loss += criterion(outputs, labels).item() * inputs.size(0)

        val_loss = running_val_loss / len(val_data)
        accuracy = total_correct / len(val_data)

        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(accuracy)
        metrics['train_accs'].append(100 * total_correct_train / len(train_data))

        if scheduler:
            scheduler.step(val_loss)

        if accuracy > best_acc or (accuracy == best_acc and val_loss < min_val_loss):
            best_acc = accuracy
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_name}_{dataset_name}_fold{fold + 1}_best_acc.pth')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'min_val_loss': min_val_loss,
            'best_acc': best_acc,
            'counter': counter
        }
        torch.save(checkpoint, checkpoint_file)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {100 * total_correct_train / len(train_data):.2f}%, Test Acc: {100 * accuracy:.2f}%')

    print(f"Best acc: {best_acc:.4f} with val loss: {min_val_loss:.4f}")

    # Сохранение метрик
    metrics_file = f'{model_name}_{dataset_name}_fold{fold + 1}_metrics.json'
    save_metrics(metrics, metrics_file)

    # TODO ?:
    #if os.path.exists(checkpoint_file):
    #    os.remove(checkpoint_file)

    return best_acc, metrics_file


def main(args):
    random_state = 1337
    torch.manual_seed(random_state)  # Reproducibility, Mean acc: 0.9625 ± 0.0322 for 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    train_transform, test_transform = get_transforms(MODELS[args.model]['input_size'], args.grayscale)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    accs = []
    f1_scores = []
    metrics_files = []
    dataset = datasets.ImageFolder(root=args.dataset)

    progress_file = os.path.join(args.checkpoint_dir, f'{args.model}_{args.dataset}_progress.json')
    accs = []
    f1_scores = []
    metrics_files = []
    completed_folds = []

    if args.resume and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        accs = progress['accs']
        f1_scores = progress['f1_scores']
        metrics_files = progress['metrics_files']
        completed_folds = progress['completed_folds']
        print(f"Возобновление: Завершено фолдов: {len(completed_folds)}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.targets)):
        if fold in completed_folds:
            print(f"Fold {fold + 1} уже завершен, пропуск")
            continue
        train_data = torch.utils.data.Subset(dataset, train_idx)
        train_data = TransformDataset(train_data, train_transform)
        val_data = torch.utils.data.Subset(dataset, val_idx)
        val_data = TransformDataset(val_data, test_transform)
        print(f"Fold {fold + 1}:")
        print("Number of training samples:", len(train_data))
        print("Number of validation samples:", len(val_data))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        model = load_model(args.model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        best_acc, metrics_file = train_model(model, optimizer, criterion, train_loader, val_loader,
                                             train_data, val_data, scheduler, dataset_name=args.dataset,
                                             model_name=args.model, num_epochs=args.epochs, patience=args.patience,
                                             fold=fold, resume=args.resume, checkpoint_dir=args.checkpoint_dir)
        accs.append(best_acc)
        metrics_files.append(metrics_file)

        # F1 score
        model.load_state_dict(torch.load(f'{args.model}_{args.dataset}_fold{fold + 1}_best_acc.pth', weights_only=True))
        model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
        f1 = f1_score(true_labels, pred_labels, average='macro')
        f1_scores.append(f1)
        print(f"Fold {fold + 1} F1: {f1:.4f}")

        completed_folds.append(fold)
        progress = {
            'accs': accs,
            'f1_scores': f1_scores,
            'metrics_files': metrics_files,
            'completed_folds': completed_folds
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)

    print(f"Mean acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    # Построение графиков для всех фолдов
    for fold, metrics_file in enumerate(metrics_files):
        plot_and_save_curves(metrics_file, args.save_dir, args.dataset, args.model, fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Модульный скрипт для бинарной классификации подписей")
    parser.add_argument('--model', type=str, default='vit_b_16', choices=list(MODELS.keys()), help='Название модели')
    parser.add_argument('--dataset', type=str, default='BHSig260-Bengali-refactored', help='Путь к датасету')
    parser.add_argument('--epochs', type=int, default=100, help='Число эпох')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')  # TODO: увеличил с 8 до 32
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Patience для early stopping')
    parser.add_argument('--grayscale', action='store_true', help='Использовать grayscale transforms')
    parser.add_argument('--save_dir', type=str, default='plots', help='Директория для сохранения графиков')
    parser.add_argument('--resume', action='store_true', help='Возобновить обучение с чекпоинта')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Директория для чекпоинтов')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
