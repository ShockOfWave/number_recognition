import argparse

from src.inference.inference import CRNNInference


def main():
    parser = argparse.ArgumentParser(description="Inference for CRNN model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (e.g., test.jpg)")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best-checkpoint.ckpt", help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    # Инициализируем класс инференса
    inference_engine = CRNNInference(checkpoint_path=args.checkpoint)

    # Выполняем предсказание для указанного изображения
    prediction = inference_engine.predict_image(args.image)
    print("Predicted number:", prediction)


if __name__ == "__main__":
    main()
