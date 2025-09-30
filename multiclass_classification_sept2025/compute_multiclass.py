import numpy as np
from sklearn.metrics import f1_score
import torch


def compute_multiclass_metrics(model, dataloader, device):  # class_to_idx ??
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images, labels_from_dataloader in dataloader:
            images = images.to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            labels.append(labels_from_dataloader.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # Compute prototypes (mean embedding per class)
    prototypes = {cls: embeddings[labels == cls].mean(axis=0) for cls in np.unique(labels)}

    # Predict class for each test embedding
    y_pred = [np.argmin([np.linalg.norm(emb - prototypes[cls]) for cls in prototypes]) for emb in embeddings]

    # Macro F1
    f1 = f1_score(labels, y_pred, average='macro')
    return {'f1_score': f1, 'y_true': labels, 'y_pred': y_pred}

