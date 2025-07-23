from ultralytics import YOLO

# Load base model
model = YOLO("models/yolov8n.pt")

# Train on custom data
model.train(
    data="emergency-dataset/data.yaml",
    epochs=50,
    imgsz=640
)


