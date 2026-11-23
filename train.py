from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.pt")

train_results = model.train(
    data="dataset/target.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    workers=0,
    patience=10,
)
