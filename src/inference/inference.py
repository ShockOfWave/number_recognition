import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models.lightning_module import CRNNLightning
from src.utils.decoding import decode_predictions_beam


class CRNNInference:
    """
    A class for performing inference using a CRNN model.

    Methods:
    - __init__: Initializes the inference class.

        Args:
            checkpoint_path (str): Path to the Lightning model checkpoint.
            img_size (tuple): Image size (imgH, imgW). Default is (128, 128).
            beam_width (int): Beam search decoder width. Default is 10.
            device (torch.device): Inference device. If None, automatically selected.

    - predict_image: Performs prediction for an image at the specified path.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Predicted string (number).

    - predict: Performs prediction for a PIL.Image object.

        Args:
            image (PIL.Image.Image): Image for prediction.

        Returns:
            str: Predicted string (number).
    """
    def __init__(self, checkpoint_path, img_size=(128, 128), beam_width=10, device=None):
        """
        Инициализирует класс инференса.

        Args:
            checkpoint_path (str): Путь к чекпоинту модели Lightning.
            img_size (tuple): Размер изображения (imgH, imgW). По умолчанию (128, 128).
            beam_width (int): Ширина beam search декодера. По умолчанию 10.
            device (torch.device): Устройство для инференса. Если None, выбирается автоматически.
        """
        self.checkpoint_path = checkpoint_path
        self.beam_width = beam_width
        self.imgH, self.imgW = img_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Настраиваем преобразование для входных изображений (аналогично валидационному transform)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.imgH, self.imgW)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Определяем алфавит (цифры) и словари для декодирования
        self.characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.characters)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(self.characters)}

        # Загружаем модель из чекпоинта (обратите внимание, что load_from_checkpoint вызывается на классе)
        self.model = CRNNLightning.load_from_checkpoint(self.checkpoint_path, idx_to_char=self.idx_to_char)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path: str) -> str:
        """
        Выполняет предсказание для изображения по указанному пути.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            str: Предсказанная строка (номер).
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict(image)

    def predict(self, image: Image.Image) -> str:
        """
        Выполняет предсказание для объекта PIL.Image.

        Args:
            image (PIL.Image.Image): Изображение для предсказания.

        Returns:
            str: Предсказанная строка (номер).
        """
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)  # добавляем измерение батча
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)  # размер: (T, batch, nclass)
            outputs = outputs.log_softmax(2)
            decoded_preds = decode_predictions_beam(outputs, self.idx_to_char, beam_width=self.beam_width, blank=0)
        return decoded_preds[0]
