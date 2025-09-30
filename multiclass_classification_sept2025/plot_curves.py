import matplotlib.pyplot as plt
import json
import os


def plot_and_save_curves(metrics_file, save_dir, dataset_name):
    os.makedirs(save_dir, exist_ok=True)
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    epochs = range(len(metrics['train_losses']))
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_f1_scores'], label='Train F1')
    plt.plot(epochs, metrics['val_f1_scores'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_roc_aucs'], label='Train ROC AUC')
    plt.plot(epochs, metrics['val_roc_aucs'], label='Val ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['train_pr_aucs'], label='Train PR AUC')
    plt.plot(epochs, metrics['val_pr_aucs'], label='Val PR AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_{dataset_name}.png")
    plt.close()


if __name__ == "__main__":
    metrics_file = "metrics/metrics_vit_b_16_dataset_CEDAR_10-9.json"
    save_dir = "plots/vit_b_16_CEDAR"
    plot_and_save_curves(metrics_file, save_dir, "CEDAR")
