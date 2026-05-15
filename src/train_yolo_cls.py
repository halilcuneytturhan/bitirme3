from ultralytics import YOLO
import os
import shutil

DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Küçük ve hızlı classification modeli
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATA_DIR,
    epochs=30,
    imgsz=224,
    batch=16,
    patience=5,
    project="runs",
    name="yolo_cls_drowsiness"
)

# En iyi modeli models klasörüne kopyala
best_model_path = "runs/classify/runs/yolo_cls_drowsiness/weights/best.pt"
target_path = os.path.join(MODEL_DIR, "yolo_drowsiness_cls.pt")

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, target_path)
    print(f"Model kaydedildi: {target_path}")
else:
    print("best.pt bulunamadı. runs klasörünü kontrol et.")