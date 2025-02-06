import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.inference.inference import CRNNInference
from src.utils.paths import get_project_path

app = FastAPI(title="CRNN Inference API")

# Инициализируем инференс-движок один раз при старте сервера.
CHECKPOINT_PATH = os.path.join(get_project_path(), "checkpoints", "best-checkpoint.ckpt")
inference_engine = CRNNInference(checkpoint_path=CHECKPOINT_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Проверяем, что загруженный файл имеет поддерживаемый формат
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        # Читаем содержимое файла в память
        contents = await file.read()
        image_stream = BytesIO(contents)
        # Открываем изображение через PIL
        image = Image.open(image_stream).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # Получаем предсказание (номер) с помощью нашего класса инференса
    prediction = inference_engine.predict(image)
    return JSONResponse(content={"filename": file.filename, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
