# Используем официальный образ Python (например, 3.10-slim)
FROM python:3.10-slim

# Отключаем создание .pyc файлов и буферизацию вывода
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости (например, для работы с изображениями)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей в рабочую директорию
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости из requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт 8000 для доступа к приложению
EXPOSE 8000

# Команда для запуска FastAPI-приложения через uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
