# controllers/detection.py
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")  # Make sure this file exists

def detect_emergency_vehicles(frame):
    results = model.predict(source=frame, verbose=False)
    labels = results[0].names
    detected = [labels[int(cls)] for cls in results[0].boxes.cls]
    print("🔍 Detected classes:", detected)

    for item in detected:
        if any(keyword in item.lower() for keyword in ["ambulance", "police", "fire", "truck"]):
            return True
    return False


