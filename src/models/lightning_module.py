import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.crnn import CRNN
from src.utils.decoding import decode_predictions_beam
from src.utils.metrics import levenshtein_distance


class CRNNLightning(pl.LightningModule):
    """
    A class representing a CRNN (Convolutional Recurrent Neural Network) model implemented using PyTorch Lightning.

    Methods:
        __init__: Initializes the CRNN model with the specified parameters including image height, number of channels, number of classes, number of hidden units in LSTM, learning rate, blank token index, and a dictionary for decoding.
        forward: Performs a forward pass through the network.
        training_step: Performs a single training step and returns a dictionary containing the loss and various training metrics.
        validation_step: Computes the validation loss and metrics for the given batch and returns a dictionary containing the validation loss and various validation metrics.
        test_step: Performs a test step using the given batch and batch index.
        configure_optimizers: Configures the optimizers for the model training.
    """
    def __init__(self, imgH, nc, nclass, nh, learning_rate=1e-4, blank=0, idx_to_char=None):
        """
        imgH: высота изображения (например, 128)
        nc: число каналов (3 для RGB)
        nclass: число классов (цифры + blank)
        nh: число скрытых единиц LSTM
        learning_rate: скорость обучения
        blank: индекс blank-токена
        idx_to_char: словарь для декодирования (индекс -> символ)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["idx_to_char"])
        self.model = CRNN(imgH, nc, nclass, nh)
        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True)
        self.learning_rate = learning_rate
        self.blank = blank
        if idx_to_char is None:
            raise ValueError("Необходимо передать idx_to_char для декодирования.")
        self.idx_to_char = idx_to_char

    def forward(self, x):
        """
    Perform a forward pass through the network.

    Args:
        self: The object instance.
        x: Input data of type TBD.

    Returns:
        None: This method does not return any value.
    """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
    Perform a single training step.

    Args:
        self: The object itself.
        batch: The input batch data.
        batch_idx: The index of the batch.

    Returns:
        dict: A dictionary containing the loss and various training metrics including image accuracy,
        precision, recall, and F1 score for both image and digit levels.
    """
        images, targets_concat, target_lengths, label_strs = batch
        outputs = self(images)
        T, batch_size, _ = outputs.size()
        input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=self.device)
        loss = self.criterion(
            outputs.log_softmax(2),
            targets_concat.to(self.device),
            input_lengths,
            torch.tensor(target_lengths, device=self.device),
        )

        # Декодирование и вычисление метрик (image-level и digit-level)
        outputs_log = outputs.log_softmax(2)
        decoded_preds = decode_predictions_beam(outputs_log, self.idx_to_char, beam_width=10, blank=self.blank)

        correct_images = 0
        total_images = 0
        total_correct_digits = 0
        total_predicted_digits = 0
        total_actual_digits = 0
        for pred_str, true_str in zip(decoded_preds, label_strs):
            total_images += 1
            if pred_str == true_str:
                correct_images += 1
            total_predicted_digits += len(pred_str)
            total_actual_digits += len(true_str)
            d = levenshtein_distance(true_str, pred_str)
            correct_digits = max(0, len(true_str) - d)
            total_correct_digits += correct_digits

        train_image_acc = correct_images / total_images if total_images > 0 else 0
        # Для бинарной задачи "полностью правильное число" precision, recall, f1 равны accuracy
        train_image_prec = train_image_acc
        train_image_rec = train_image_acc
        train_image_f1 = train_image_acc

        train_digit_prec = total_correct_digits / total_predicted_digits if total_predicted_digits > 0 else 0
        train_digit_rec = total_correct_digits / total_actual_digits if total_actual_digits > 0 else 0
        train_digit_f1 = (
            2 * train_digit_prec * train_digit_rec / (train_digit_prec + train_digit_rec)
            if (train_digit_prec + train_digit_rec) > 0
            else 0
        )

        # Логирование метрик
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_image_acc", train_image_acc, prog_bar=True)
        self.log("train_image_prec", train_image_prec, prog_bar=True)
        self.log("train_image_rec", train_image_rec, prog_bar=True)
        self.log("train_image_f1", train_image_f1, prog_bar=True)
        self.log("train_digit_prec", train_digit_prec, prog_bar=True)
        self.log("train_digit_rec", train_digit_rec, prog_bar=True)
        self.log("train_digit_f1", train_digit_f1, prog_bar=True)

        return {
            "loss": loss,
            "train_image_acc": train_image_acc,
            "train_image_prec": train_image_prec,
            "train_image_rec": train_image_rec,
            "train_image_f1": train_image_f1,
            "train_digit_prec": train_digit_prec,
            "train_digit_rec": train_digit_rec,
            "train_digit_f1": train_digit_f1,
        }

    def validation_step(self, batch, batch_idx):
        """
        Computes the validation loss and metrics for the given batch.

        Args:
            self: The object itself.
            batch: The input batch for validation.
            batch_idx: The index of the batch.

        Returns:
            dict: A dictionary containing the validation loss and various validation metrics.
                - "val_loss": The validation loss.
                - "val_image_acc": The image accuracy for validation.
                - "val_image_prec": The image precision for validation.
                - "val_image_rec": The image recall for validation.
                - "val_image_f1": The image F1 score for validation.
                - "val_digit_prec": The digit precision for validation.
                - "val_digit_rec": The digit recall for validation.
                - "val_digit_f1": The digit F1 score for validation.
        """
        images, targets_concat, target_lengths, label_strs = batch
        outputs = self(images)
        outputs = outputs.log_softmax(2)
        T, batch_size, _ = outputs.size()
        input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=self.device)
        # Вычисляем валидационный loss с использованием CTCLoss
        val_loss = self.criterion(
            outputs, targets_concat.to(self.device), input_lengths, torch.tensor(target_lengths, device=self.device)
        )
        self.log("val_loss", val_loss, prog_bar=True)

        # Далее вычисляем метрики (image-level и digit-level) как раньше:
        decoded_preds = decode_predictions_beam(outputs, self.idx_to_char, beam_width=10, blank=self.blank)
        correct_images = 0
        total_images = 0
        total_correct_digits = 0
        total_predicted_digits = 0
        total_actual_digits = 0
        for pred_str, true_str in zip(decoded_preds, label_strs):
            total_images += 1
            if pred_str == true_str:
                correct_images += 1
            total_predicted_digits += len(pred_str)
            total_actual_digits += len(true_str)
            d = levenshtein_distance(true_str, pred_str)
            correct_digits = max(0, len(true_str) - d)
            total_correct_digits += correct_digits

        val_image_acc = correct_images / total_images if total_images > 0 else 0
        val_image_prec = val_image_acc  # для задачи полного совпадения
        val_image_rec = val_image_acc
        val_image_f1 = val_image_acc
        val_digit_prec = total_correct_digits / total_predicted_digits if total_predicted_digits > 0 else 0
        val_digit_rec = total_correct_digits / total_actual_digits if total_actual_digits > 0 else 0
        val_digit_f1 = (
            2 * val_digit_prec * val_digit_rec / (val_digit_prec + val_digit_rec)
            if (val_digit_prec + val_digit_rec) > 0
            else 0
        )

        self.log("val_image_acc", val_image_acc, prog_bar=True)
        self.log("val_image_prec", val_image_prec, prog_bar=True)
        self.log("val_image_rec", val_image_rec, prog_bar=True)
        self.log("val_image_f1", val_image_f1, prog_bar=True)
        self.log("val_digit_prec", val_digit_prec, prog_bar=True)
        self.log("val_digit_rec", val_digit_rec, prog_bar=True)
        self.log("val_digit_f1", val_digit_f1, prog_bar=True)

        return {
            "val_loss": val_loss,
            "val_image_acc": val_image_acc,
            "val_image_prec": val_image_prec,
            "val_image_rec": val_image_rec,
            "val_image_f1": val_image_f1,
            "val_digit_prec": val_digit_prec,
            "val_digit_rec": val_digit_rec,
            "val_digit_f1": val_digit_f1,
        }

    def test_step(self, batch, batch_idx):
        """
        Perform a test step using the given batch and batch index.

        Args:
            self: The object instance.
            batch: The input batch for testing.
            batch_idx: The index of the batch.

        Returns:
            None
        """
        images, img_paths = batch
        outputs = self(images)
        outputs = outputs.log_softmax(2)
        decoded_preds = decode_predictions_beam(outputs, self.idx_to_char, beam_width=10, blank=self.blank)
        return {"img_paths": img_paths, "preds": decoded_preds}

    def configure_optimizers(self):
        """
    Configures the optimizers for the model training.

    Args:
        self: The instance of the class.

    Returns:
        None. The method configures the optimizers for the model training without returning any value.
    """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # планировщик будет уменьшать lr, если val_loss не улучшается
                "interval": "epoch",
                "frequency": 1,
            },
        }
