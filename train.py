import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.dataset import CRNNDataModule
from src.models.lightning_module import CRNNLightning
from src.utils.decoding import decode_predictions_beam
from src.utils.paths import get_project_path


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    seed_everything(seed=42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # Гиперпараметры
    imgH, imgW = 128, 128
    nc = 3  # RGB
    nh = 256
    characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    char_to_idx = {ch: i + 1 for i, ch in enumerate(characters)}
    idx_to_char = {i + 1: ch for i, ch in enumerate(characters)}
    nclass = len(characters) + 1  # + blank
    learning_rate = 1e-4
    batch_size = 32
    num_workers = 4
    num_epochs = 100

    # Преобразования
    # Для обучения добавляем аугментацию: случайное горизонтальное отражение и небольшое вращение.
    train_transform = transforms.Compose(
        [
            transforms.Resize((imgH, imgW)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # Для валидации и теста – стандартное масштабирование и нормализация.
    val_transform = transforms.Compose(
        [transforms.Resize((imgH, imgW)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_csv = os.path.join(get_project_path(), "data", "train.csv")
    val_csv = os.path.join(get_project_path(), "data", "val.csv")
    test_csv = os.path.join(get_project_path(), "data", "test.csv")

    early_stopping = EarlyStopping(
        monitor="val_loss",  # метрика, по которой будет работать ранняя остановка
        min_delta=0.0001,
        patience=15,  # количество эпох без улучшения, после которых обучение остановится
        verbose=True,
        mode="min",  # так как мы минимизируем loss
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # контрольная метрика
        dirpath="checkpoints",  # папка, куда будут сохраняться модели
        filename="best-checkpoint",  # имя файла
        save_top_k=1,  # сохраняем только лучшую модель
        mode="min",  # минимальное значение loss
        verbose=True,
    )

    # Создаем DataModule
    data_module = CRNNDataModule(
        train_csv,
        val_csv,
        test_csv,
        train_transform,
        val_transform,
        char_to_idx,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Создаем Lightning-модель
    model = CRNNLightning(
        imgH=imgH, nc=nc, nclass=nclass, nh=nh, learning_rate=learning_rate, blank=0, idx_to_char=idx_to_char
    )

    if args.use_wandb:
        wandb_logger = WandbLogger(project="number_recognition_project")
        logger = wandb_logger
    else:
        logger = None

    # Инициализируем Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    # Обучение и тестирование (валидация) модели
    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    model = CRNNLightning.load_from_checkpoint(best_model_path, idx_to_char=idx_to_char)

    trainer.test(model, datamodule=data_module)

    # --- Инференс на тестовом наборе ---
    test_loader = data_module.test_dataloader()
    all_img_paths = []
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, img_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            # Декодируем предсказания с помощью beam search
            decoded_preds = decode_predictions_beam(outputs, idx_to_char, beam_width=10, blank=0)
            all_img_paths.extend(img_paths)
            all_preds.extend(decoded_preds)
    # Загружаем тестовый CSV, добавляем предсказания и сохраняем новый файл
    test_df = pd.read_csv(test_csv)
    pred_dict = dict(zip(all_img_paths, all_preds))
    test_df["predicted_price"] = test_df["img_name"].map(pred_dict)
    output_csv = "test_with_predictions.csv"
    test_df.to_csv(output_csv, index=False)
    print(f"Test predictions saved in '{output_csv}'.")


if __name__ == "__main__":
    main()
