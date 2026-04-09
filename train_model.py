from ultralytics import YOLO

# Load pretrained classification model
model = YOLO("yolov8n-cls.pt")

# Train model
model.train(
    data="dataset",
    epochs=12,
    imgsz=128,
    batch=256,
    workers=8,
    device="cpu",
    pretrained=True
)