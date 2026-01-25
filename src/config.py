from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
