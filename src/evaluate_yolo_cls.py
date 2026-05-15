from ultralytics import YOLO

MODEL_PATH = "models/yolo_drowsiness_cls.pt"
DATA_DIR = "data"

model = YOLO(MODEL_PATH)

metrics = model.val(data=DATA_DIR, split="test")

print(metrics)