import numpy as np
import torch

from torch.utils.data import Dataset


class OrderlessNADEDataset(Dataset):
    def __init__(self, img_data, is_train):
        self.img_data = img_data
        self.is_train = is_train

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        pixels = self.img_data[idx].flatten()
        idxs = np.arange(len(pixels))
        if self.is_train:
            np.random.shuffle(idxs)
            n_cond = np.random.randint(len(pixels))
        else:
            n_cond = idx % len(pixels)

        mask = np.zeros(len(pixels))
        mask[idxs[:n_cond]] = 1

        # See: Eq. (12) in "A Deep and Tractable Density Estimator".
        scale = len(mask) / (len(mask) - n_cond)

        return {
            "pixels": torch.Tensor(pixels),
            "mask": torch.Tensor(mask),
            "scale": torch.full([len(mask)], scale),
        }
