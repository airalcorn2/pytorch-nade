import torch

from torch.utils.data import Dataset


class NADEDataset(Dataset):
    def __init__(self, img_data):
        self.img_data = img_data

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        pixels = self.img_data[idx].flatten()
        return {"pixels": torch.Tensor(pixels)}
