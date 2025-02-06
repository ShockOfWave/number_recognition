import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import get_project_path


class CRNNDataset(Dataset):
    def __init__(self, csv_file, transform=None, char_to_idx=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if char_to_idx is None:
            raise ValueError("Передайте char_to_idx для преобразования меток.")
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_name']
        # Если метка хранится как float, приводим к int и затем к str
        label_str = str(int(float(row['text'])))
        image = Image.open(os.path.join(get_project_path(), "data", "imgs", img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = [self.char_to_idx[ch] for ch in label_str if ch in self.char_to_idx]
        target = torch.tensor(target, dtype=torch.long)
        return image, target, label_str

# Датасет для тестовой выборки
class CRNNTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_name']
        image = Image.open(os.path.join(get_project_path(), "data", "imgs", img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Функция для объединения примеров с переменной длиной меток
def crnn_collate_fn(batch):
    images, targets, label_strs = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = [len(t) for t in targets]
    targets_concat = torch.cat(targets)
    return images, targets_concat, target_lengths, label_strs

# DataModule для Lightning
class CRNNDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, train_transform, val_transform, char_to_idx, batch_size=64, num_workers=16):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.char_to_idx = char_to_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        

    def setup(self, stage=None):
        self.train_dataset = CRNNDataset(self.train_csv, transform=self.train_transform, char_to_idx=self.char_to_idx)
        self.val_dataset = CRNNDataset(self.val_csv, transform=self.val_transform, char_to_idx=self.char_to_idx)
        self.test_dataset = CRNNTestDataset(self.test_csv, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=crnn_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=crnn_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
