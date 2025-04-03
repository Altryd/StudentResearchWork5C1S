import torch.utils.data
from torchvision import datasets


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transformations, imgs=None, targets=None):
        super(TransformDataset, self).__init__()
        self.base = base_dataset
        self.transformations = transformations
        self.imgs = imgs
        self.targets = targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transformations(x), y
