"""
Fabric Defect Detection - Image Detection Script

Usage:
    python detect_image.py image.jpeg
    python detect_image.py image.jpeg --output results/
    python detect_image.py image.jpeg --model path/to/model.pt --conf 0.3
    python detect_image.py image.jpeg --pixels-per-mm 15.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    BASE_DIR,
    TRAINED_MODEL_PATH,
    TRAINING_RUN_PATH,
    OUTPUT_DIR,
    YOLO_IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    PIXELS_PER_MM,
    DEFECT_CLASSES
)


# Colors for each defect class (BGR format)
DEFECT_COLORS = {
    "foreign yarn": (255, 0, 0),      # Blue
    "hole": (0, 0, 255),              # Red
    "missing yarn": (0, 165, 255),    # Orange
    "slub": (0, 255, 255),            # Yellow
    "spot": (255, 0, 255),            # Magenta
    "thick yarn": (0, 255, 0),        # Green
}


def find_model_path(custom_path: str = None) -> Path:
    """
    Find the trained model path with fallback logic.
    
    Priority:
    1. Custom path provided via --model argument
    2. Default path: models/trained/best.pt
    3. Fallback: runs/fabric-defect-yolov8n/weights/best.pt
    """
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Model not found at specified path: {custom_path}")
    
    # Try default trained model path
    if TRAINED_MODEL_PATH.exists():
        return TRAINED_MODEL_PATH
    
    # Fallback to training run output
    if TRAINING_RUN_PATH.exists():
        print(f"[INFO] Using model from training run: {TRAINING_RUN_PATH}")
        return TRAINING_RUN_PATH
    
    raise FileNotFoundError(
        f"No trained model found.\n"
        f"  Expected at: {TRAINED_MODEL_PATH}\n"
        f"  Or fallback: {TRAINING_RUN_PATH}\n"
        f"Please train a model first using: python src/training/train_yolo.py"
    )


def calculate_dimensions(box, pixels_per_mm: float) -> tuple:
    """
    Calculate defect dimensions in millimeters.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        pixels_per_mm: Calibration ratio (pixels per millimeter)
    
    Returns:
        Tuple of (width_mm, height_mm)
    """
    x1, y1, x2, y2 = box
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    
    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm
    
    return width_mm, height_mm


def annotate_image(image: np.ndarray, results, pixels_per_mm: float) -> np.ndarray:
    """
    Draw bounding boxes with defect info on the image.
    
    Args:
        image: Original image (BGR format)
        results: YOLO prediction results
        pixels_per_mm: Calibration ratio for size calculation
    
    Returns:
        Annotated image with bounding boxes and labels
    """
    annotated = image.copy()
    
    for result in results:
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
            
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class and confidence
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            # Get class name
            if class_id < len(DEFECT_CLASSES):
                class_name = DEFECT_CLASSES[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Get color for this defect type
            color = DEFECT_COLORS.get(class_name, (255, 255, 255))
            
            # Calculate dimensions in mm
            width_mm, height_mm = calculate_dimensions(
                [x1, y1, x2, y2], pixels_per_mm
            )
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            size_label = f"W:{width_mm:.1f}mm H:{height_mm:.1f}mm"
            
            # Calculate text sizes for background rectangles
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (size_w, size_h), _ = cv2.getTextSize(size_label, font, font_scale, thickness)
            
            # Draw label background and text (above box)
            label_y = max(y1 - 10, label_h + 10)
            cv2.rectangle(
                annotated,
                (x1, label_y - label_h - 5),
                (x1 + label_w + 5, label_y + 5),
                color,
                -1
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, label_y),
                font, font_scale, (255, 255, 255), thickness
            )
            
            # Draw size label background and text (below the class label)
            size_y = label_y + size_h + 15
            cv2.rectangle(
                annotated,
                (x1, size_y - size_h - 5),
                (x1 + size_w + 5, size_y + 5),
                color,
                -1
            )
            cv2.putText(
                annotated, size_label,
                (x1 + 2, size_y),
                font, font_scale, (255, 255, 255), thickness
            )
    
    return annotated


def print_detection_summary(results, pixels_per_mm: float):
    """Print a summary of detected defects to console."""
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    total_defects = 0
    defect_counts = {name: 0 for name in DEFECT_CLASSES}
    
    for result in results:
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        for i, box in enumerate(boxes):
            total_defects += 1
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            if class_id < len(DEFECT_CLASSES):
                class_name = DEFECT_CLASSES[class_id]
            else:
                class_name = f"class_{class_id}"
            
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
            
            width_mm, height_mm = calculate_dimensions(
                [x1, y1, x2, y2], pixels_per_mm
            )
            
            print(f"\nDefect #{total_defects}:")
            print(f"  Type: {class_name}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Size: {width_mm:.1f}mm x {height_mm:.1f}mm")
    
    print("\n" + "-" * 60)
    print(f"SUMMARY: {total_defects} defect(s) detected")
    
    if total_defects > 0:
        print("\nDefect counts by type:")
        for name, count in defect_counts.items():
            if count > 0:
                print(f"  - {name}: {count}")
    
    print("=" * 60 + "\n")


def detect_defects(
    image_path: str,
    model_path: str = None,
    output_path: str = None,
    confidence: float = CONFIDENCE_THRESHOLD,
    pixels_per_mm: float = PIXELS_PER_MM,
    show: bool = False
) -> Path:
    """
    Main detection function.
    
    Args:
        image_path: Path to input image
        model_path: Optional custom model path
        output_path: Optional output directory
        confidence: Confidence threshold for detections
        pixels_per_mm: Calibration ratio for size calculation
        show: Whether to display the result image
    
    Returns:
        Path to the saved output image
    """
    # Validate input image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"[INFO] Processing: {image_path}")
    
    # Find and load model
    model_file = find_model_path(model_path)
    print(f"[INFO] Loading model: {model_file}")
    model = YOLO(str(model_file))
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"[INFO] Confidence threshold: {confidence}")
    print(f"[INFO] Pixels per mm: {pixels_per_mm}")
    
    # Run inference
    print("[INFO] Running detection...")
    results = model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=confidence,
        verbose=False
    )
    
    # Print detection summary
    print_detection_summary(results, pixels_per_mm)
    
    # Annotate image
    annotated = annotate_image(image, results, pixels_per_mm)
    
    # Determine output path
    if output_path:
        output_dir = Path(output_path)
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    
    # Save annotated image
    cv2.imwrite(str(output_file), annotated)
    print(f"[INFO] Result saved to: {output_file}")
    
    # Show image if requested
    if show:
        window_name = f"Defect Detection - {image_path.name}"
        cv2.imshow(window_name, annotated)
        print("[INFO] Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return output_file


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect fabric defects in an image using trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_image.py fabric_sample.jpg
  python detect_image.py fabric_sample.jpg --show
  python detect_image.py fabric_sample.jpg --output results/
  python detect_image.py fabric_sample.jpg --model custom_model.pt
  python detect_image.py fabric_sample.jpg --conf 0.5 --pixels-per-mm 12.0
        """
    )
    
    parser.add_argument(
        "image",
        help="Path to the input image file"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Path to custom YOLO model (default: models/trained/best.pt)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for annotated image (default: output/)"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detections (default: {CONFIDENCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--pixels-per-mm", "-p",
        type=float,
        default=PIXELS_PER_MM,
        help=f"Pixels per millimeter for size calculation (default: {PIXELS_PER_MM})"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the annotated image in a window"
    )
    
    args = parser.parse_args()
    
    try:
        detect_defects(
            image_path=args.image,
            model_path=args.model,
            output_path=args.output,
            confidence=args.conf,
            pixels_per_mm=args.pixels_per_mm,
            show=args.show
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
