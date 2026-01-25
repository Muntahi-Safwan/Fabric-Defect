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
    )


if __name__ == "__main__":
    main()
