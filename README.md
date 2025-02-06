# Распознавание Номеров с помощью CRNN

![GitHub](https://img.shields.io/github/license/ShockOfWave/number_recognition)
![GitHub last commit](https://img.shields.io/github/last-commit/ShockOfWave/number_recognition)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ShockOfWave/number_recognition)
![contributors](https://img.shields.io/github/contributors/ShockOfWave/number_recognition) 
![codesize](https://img.shields.io/github/languages/code-size/ShockOfWave/number_recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/ShockOfWave/number_recognition)
![GitHub top language](https://img.shields.io/github/languages/top/ShockOfWave/number_recognition)
![GitHub language count](https://img.shields.io/github/languages/count/ShockOfWave/number_recognition)

Этот проект реализует CRNN (сверточно-рекуррентную нейронную сеть) для распознавания чисел (цифр) по изображениям. Модель построена на основе PyTorch и организована с использованием PyTorch Lightning для удобства обучения и оценки. Кроме того, проект предоставляет модуль инференса с классом `CRNNInference` и FastAPI-эндпоинтом, а также поддержку Docker для развертывания.

## Содержание

- [Структура проекта](#структура-проекта)
- [Особенности](#особенности)
- [Требования](#требования)
- [Установка](#установка)
- [Обучение](#обучение)
- [Инференс](#инференс)
  - [Использование класса для инференса](#использование-класса-для-инференса)
  - [FastAPI-эндпоинт](#fastapi-эндпоинт)
- [Docker](#docker)
- [Линтинг и качество кода](#лингтинг-и-качество-кода)
- [Контакты](#контакты)
- [Лицензия](#лицензия)

## Подробности по реализации проекта находятся в [документации](docs/README.md).

## Структура проекта

```
number_recognition/
├── Dockerfile
├── app.py                 # FastAPI-эндпоинт
├── train.py               # Скрипт для обучения (запуск: python main.py)
├── inference.py           # Скрипт для инференса из командной строки (опционально)
├── README.md
├── requirements.txt
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py     # Датасеты и DataModule для обучения, валидации и теста
    ├── models/
    │   ├── __init__.py
    │   ├── crnn.py        # Архитектура CRNN и Bidirectional LSTM
    │   └── lightning_module.py  # Обёртка модели CRNN в LightningModule
    ├── inference/
    │   ├── __init__.py
    │   └── inference.py   # Класс CRNNInference для инференса
    └── utils/
        ├── __init__.py
        ├── decoding.py    # Функции beam search декодирования
        └── metrics.py     # Вспомогательные функции (например, расстояние Левенштейна)
```

## Особенности

- **Архитектура модели:**  
  Мощная CRNN с увеличенным числом фильтров в сверточных слоях для распознавания цифр.

- **Обучение:**  
  Обучение модели через PyTorch Lightning с:
  - Пользовательскими шагами обучения и валидации.
  - Логированием метрик: 
    - **На уровне изображения**: точность, precision, recall, F1-score для определения, правильно ли распознано число целиком.
    - **На уровне цифр**: precision, recall, F1-score для распознавания отдельных цифр.
  - Вычислением лосса на обучении и валидации.
  - Планировщиком ReduceLROnPlateau для адаптивного уменьшения скорости обучения.
  - Регуляризацией с использованием weight decay.
  - Ранней остановкой (Early Stopping) и сохранением лучших весов (ModelCheckpoint).

- **Инференс:**  
  Класс `CRNNInference` позволяет легко выполнять предсказание для отдельного изображения. Также реализован FastAPI-эндпоинт для получения предсказаний через REST API.

- **Docker:**  
  Проект содержит Dockerfile для сборки и развертывания контейнера с приложением.

- **Качество кода:**  
  Рекомендуется использовать линтеры (flake8, black, isort, mypy) для проверки стиля и качества кода.

## Требования

Установите необходимые зависимости:

```bash
pip install -r requirements.txt
```

Для разработки можно установить дополнительно:

```
flake8
black
isort
```

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/ShockOfWave/number_recognition.git
   cd number_recognition
   ```

2. **(Опционально) Создайте виртуальное окружение и активируйте его:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
   ```

3. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Скачайте данные и лучшую модель:**

   ```bash
   sh get_data_and_best_model.sh
   ```

## Обучение

Запустите скрипт для обучения:

```bash
python train.py --use_wandb
```

- Параметр `--use_wandb` включает логирование в Weights & Biases (если настроено).
- Скрипт `train.py` настраивает аугментации, обучающие параметры, коллбэки (Early Stopping и ModelCheckpoint) и сохраняет лучшую модель в папке `checkpoints/`.

## Инференс

### Использование класса для инференса

Вы можете импортировать класс `CRNNInference` и использовать его для получения предсказания для отдельного изображения:

```python
from src.inference.inference import CRNNInference

inference_engine = CRNNInference(checkpoint_path="checkpoints/best-checkpoint.ckpt")
prediction = inference_engine.predict_image("test.jpg")
print("Распознанное число:", prediction)
```

### FastAPI-эндпоинт

Запустите FastAPI-эндпоинт (файл `app.py`):

```bash
python app.py
```

Пример запроса через curl:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"
```

Ответ будет в формате JSON, например:

```json
{
  "filename": "test.jpg",
  "prediction": "12345"
}
```

## Docker

Чтобы собрать и запустить контейнер с приложением:

1. **Соберите Docker-образ:**

   ```bash
   docker build -t crnn-app .
   ```

2. **Запустите контейнер:**

   ```bash
   docker run -p 8000:8000 crnn-app
   ```

Приложение будет доступно по адресу [http://localhost:8000](http://localhost:8000).

## Линтинг и качество кода

Для проверки качества кода выполните следующие команды:

- **flake8:**

  ```bash
  flake8 src/ main.py
  ```

- **black (форматирование кода):**

  ```bash
  black src/ main.py
  ```

- **isort (сортировка импортов):**

  ```bash
  isort .
  ```


## Запуск через Docker

Для сборки и запуска Docker-контейнера выполните:

```bash
docker build -t crnn-app .
docker run -p 8000:8000 crnn-app
```

## Контакты

Если у вас возникнут вопросы, пожалуйста, свяжитесь с автором проекта.

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности смотрите в файле [LICENSE](LICENSE).