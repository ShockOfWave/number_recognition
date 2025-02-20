import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """
    A class representing a Bidirectional LSTM model.

    Class Methods:
    - __init__: 
        Initializes a BidirectionalLSTM object.

        Args:
            self: The object instance.
            nIn (int): Number of input features.
            nHidden (int): Number of hidden units in the LSTM layer.
            nOut (int): Number of output features.

        Returns:
            None

    - forward: 
        Perform forward pass through the model.

        Args:
            self: The object instance.
            input (torch.Tensor): The input tensor of shape (T, b, input_size).

        Returns:
            torch.Tensor: The output tensor after forward pass with shape (T, b, output_size).
    """
    def __init__(self, nIn, nHidden, nOut):
        """
    Initializes a BidirectionalLSTM object.

    Args:
        self: The object instance.
        nIn (int): Number of input features.
        nHidden (int): Number of hidden units in the LSTM layer.
        nOut (int): Number of output features.

    Returns:
        None
    """
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        """
        Perform forward pass through the model.

        Args:
            self: The object instance.
            input (torch.Tensor): The input tensor of shape (T, b, input_size).

        Returns:
            torch.Tensor: The output tensor after forward pass with shape (T, b, output_size).
        """
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition.

    Methods:
    - __init__: Initialize the CRNN model with the specified parameters.
        imgH: Height of the input image (e.g., 128)
        nc: Number of image channels (3 for RGB)
        nclass: Number of classes (digits + blank)
        nh: Number of hidden units in LSTM

    - forward: Perform a forward pass through the model.
        Args:
            self: The object instance.
            x (Tensor): Input tensor of shape (T, B, n_features).

        Returns:
            None
    """
    def __init__(self, imgH, nc, nclass, nh):
        """
        imgH: высота изображения (например, 128)
        nc: число каналов (3 для RGB)
        nclass: число классов (цифры + blank)
        nh: число скрытых единиц LSTM
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH должно быть кратно 16"
        self.cnn = nn.Sequential(
            # Увеличиваем число фильтров
            nn.Conv2d(nc, 128, 3, 1, 1),  # было: Conv2d(nc, 64, 3, 1, 1)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),  # было: Conv2d(64, 128, 3, 1, 1)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),  # было: Conv2d(128, 256, 3, 1, 1)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),  # осталось без изменений (но теперь 512 каналов)
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 1024, 3, 1, 1),  # было: Conv2d(256, 512, 3, 1, 1)
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, 1, 1),  # было: Conv2d(512, 512, 3, 1, 1)
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(1024, 1024, 2, 1, 0),  # остается с 1024 каналами
            nn.ReLU(True),
        )
        # Изменяем входное число каналов для рекуррентной части с 512 на 1024
        self.rnn = BidirectionalLSTM(1024, nh, nclass)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            self: The object instance.
            x (Tensor): Input tensor of shape (T, B, n_features).

        Returns:
            None
        """
        conv = self.cnn(x)
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))  # (B, 1024, 1, W)
        conv = conv.squeeze(2)  # (B, 1024, W)
        conv = conv.permute(2, 0, 1)  # (W, B, 1024)
        output = self.rnn(conv)  # (T, B, nclass)
        return output
