from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="fabric-defect-yolov8n",

        # --- Augmentation Settings ---
        fliplr=0.5,    # 50% chance to flip Left-Right (Great for fabric)
        flipud=0.5,    # 50% chance to flip Up-Down (Great for fabric)
        degrees=10.0,  # Rotate the image slightly (+/- 10 degrees)
        hsv_h=0.015,   # Change color hue slightly (Simulates different dye lots)
        hsv_s=0.7,     # Change saturation (Simulates lighting intensity)
        hsv_v=0.4,     # Change value/brightness (Simulates shadows)
    )


if __name__ == "__main__":
    main()
