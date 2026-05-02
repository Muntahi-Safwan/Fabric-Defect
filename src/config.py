from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# YOLO Detection Settings
TRAINED_MODEL_PATH = MODEL_DIR / "trained" / "best.pt"
TRAINING_RUN_PATH = BASE_DIR / "runs" / "fabric-defect-yolov8n" / "weights" / "best.pt"
YOLO_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
PIXELS_PER_MM = 10.0  # Default calibration: 10 pixels = 1mm (adjust based on your camera setup)

# Defect class names (must match data.yaml)
DEFECT_CLASSES = [
    "foreign yarn",
    "hole", 
    "missing yarn",
    "slub",
    "spot",
    "thick yarn"
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
