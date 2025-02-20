import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import get_project_path


class CRNNDataset(Dataset):
    """
    A class for handling data stored in a CSV file for CRNN models.

    Methods:
    - __init__: Initializes the object with data from a CSV file and optional transformation and character to index mapping.
    - __len__: Returns the length of the data stored in the object.
    - __getitem__: Retrieves the image and target label for the specified index from the dataset.
    """
    def __init__(self, csv_file, transform=None, char_to_idx=None):
        """
    Initializes the object with data from a CSV file and optional transformation and character to index mapping.

    Args:
        self: The object instance.
        csv_file (str): The path to the CSV file containing the data.
        transform (callable, optional): A function to apply transformation to the data. Default is None.
        char_to_idx (dict, optional): A mapping of characters to their corresponding indices. If None, a ValueError is raised.

    Returns:
        None
    """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if char_to_idx is None:
            raise ValueError("Передайте char_to_idx для преобразования меток.")
        self.char_to_idx = char_to_idx

    def __len__(self):
        """
    Returns the length of the data stored in the object.

    Parameters:
    self (object): The object itself.

    Returns:
    int: The length of the data stored in the object.
    """
        return len(self.data)

    def __getitem__(self, idx):
        """
    Retrieves the image and target label for the specified index from the dataset.

    Args:
        self (object): The CRNNTestDataset object.
        idx (int): The index of the data sample to retrieve.

    Returns:
        tuple: A tuple containing the image (PIL Image), target label (torch.Tensor), and label string (str).
    """
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
    """
    A class representing a dataset for testing CRNN models.

    Methods:
    - __init__: Initializes the object with the provided CSV file.
    - __len__: Returns the length of the data stored in the object.
    - __getitem__: Retrieves an item from the dataset at the specified index.
    """
    def __init__(self, csv_file, transform=None):
        """
        Initializes the object with the provided CSV file.

        Args:
            self: The object instance.
            csv_file (str): The path to the CSV file to be used.

        Returns:
            None
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the data stored in the object.

        Parameters:
            self (object): The object instance.

        Returns:
            None
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
    Retrieves an item from the dataset at the specified index.

    Args:
        self: The object instance.
        idx (int): The index of the item to retrieve.

    Returns:
        tuple: A tuple containing the image data and the image path.
    """
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
    """
    A class for managing data loading and processing for CRNN models.

    Methods:
    - __init__:
        Initializes the dataset with the provided CSV files and transformations.

        Args:
            self: The object instance.
            train_csv (str): Path to the training CSV file.
            val_csv (str): Path to the validation CSV file.
            test_csv (str): Path to the test CSV file.
            train_transform (Any): Transformation to apply to training data.
            val_transform (Any): Transformation to apply to validation data.
            char_to_idx (Dict[str, int]): Mapping of characters to their corresponding indices.

        Returns:
            None

    - setup:
        Initializes the datasets for validation and testing.

        Parameters:
            self: The object instance.

        Returns:
            None

    - train_dataloader:
        Creates and returns a DataLoader for training data.

        Parameters:
            self: The object instance.

        Returns:
            None

    - val_dataloader:
        Creates a validation data loader for the dataset.

        Parameters:
            self: The object instance.

        Returns:
            None

    - test_dataloader:
        Test the data loader functionality.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None: This method does not return anything.
    """
    def __init__(self, train_csv, val_csv, test_csv, train_transform, val_transform, char_to_idx, batch_size=64, num_workers=16):
        """
        Initializes the dataset with the provided CSV files and transformations.

        Args:
            self: The object instance.
            train_csv (str): Path to the training CSV file.
            val_csv (str): Path to the validation CSV file.
            test_csv (str): Path to the test CSV file.
            train_transform (Any): Transformation to apply to training data.
            val_transform (Any): Transformation to apply to validation data.
            char_to_idx (Dict[str, int]): Mapping of characters to their corresponding indices.

        Returns:
            None
        """
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
        """
        Initializes the datasets for validation and testing.

        Parameters:
            self: The object instance.

        Returns:
            None
        """
        self.train_dataset = CRNNDataset(self.train_csv, transform=self.train_transform, char_to_idx=self.char_to_idx)
        self.val_dataset = CRNNDataset(self.val_csv, transform=self.val_transform, char_to_idx=self.char_to_idx)
        self.test_dataset = CRNNTestDataset(self.test_csv, transform=self.val_transform)

    def train_dataloader(self):
        """
        Creates and returns a DataLoader for training data.

        Parameters:
        - self: The object instance.

        Returns:
        None
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=crnn_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Creates a validation data loader for the dataset.

        Parameters:
            self: The object instance.

        Returns:
            None
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=crnn_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Test the data loader functionality.

        Parameters:
        self (object): The instance of the class.

        Returns:
        None: This method does not return anything.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
