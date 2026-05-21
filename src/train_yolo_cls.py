import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# AYARLAR
# ==================================================

DATA_PATH = "data"
BASE_MODEL_PATH = "yolov8n-cls.pt"

PROJECT_NAME = "runs"
RUN_NAME = "classify/yolo_cls_drowsiness_final"

OUTPUT_DIR = "outputs/metrics"

EPOCHS = 30
IMG_SIZE = 224
BATCH_SIZE = 16


# ==================================================
# KLASÖR OLUŞTUR
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# 1. YOLO CLASSIFICATION MODELİNİ EĞİT
# ==================================================

model = YOLO(BASE_MODEL_PATH)

train_results = model.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_NAME,
    patience=5,
    name=RUN_NAME,
    exist_ok=True
)

print("\nEğitim tamamlandı.")


# ==================================================
# 2. EĞİTİLEN BEST MODELİ YÜKLE
# ==================================================


save_dir = train_results.save_dir
best_model_path = os.path.join(str(save_dir), "weights", "best.pt")

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Best model bulunamadı: {best_model_path}")

model = YOLO(best_model_path)

print("Best model yüklendi:", best_model_path)


# ==================================================
# 3. ACCURACY - LOSS GRAFİĞİ OLUŞTUR
# ==================================================

results_csv_path = os.path.join(str(save_dir), "results.csv")

if os.path.exists(results_csv_path):
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(10, 6))

    if "metrics/accuracy_top1" in df.columns:
        plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Top-1 Accuracy")

    if "metrics/accuracy_top5" in df.columns:
        plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Top-5 Accuracy")

    if "train/loss" in df.columns:
        plt.plot(df["epoch"], df["train/loss"], label="Train Loss")

    if "val/loss" in df.columns:
        plt.plot(df["epoch"], df["val/loss"], label="Validation Loss")

    plt.title("Model Accuracy ve Loss Grafiği")
    plt.xlabel("Epoch")
    plt.ylabel("Değer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_loss_path = os.path.join(OUTPUT_DIR, "accuracy_loss_graph.png")
    plt.savefig(accuracy_loss_path, dpi=300)
    plt.close()

    print("Accuracy/Loss grafiği kaydedildi:", accuracy_loss_path)

else:
    print("results.csv bulunamadı. Accuracy/Loss grafiği oluşturulamadı.")


# ==================================================
# 4. TEST VERİLERİ ÜZERİNDE TAHMİN YAP
# ==================================================

test_dir = os.path.join(DATA_PATH, "test")

if not os.path.exists(test_dir):
    raise FileNotFoundError("data/test klasörü bulunamadı. Confusion matrix için test klasörü gerekli.")

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(test_dir, "*", ext)))

if len(image_paths) == 0:
    raise FileNotFoundError("data/test içinde görsel bulunamadı.")

y_true = []
y_pred = []

class_names = list(model.names.values())

print("\nSınıflar:", class_names)
print("Test görsel sayısı:", len(image_paths))


for img_path in image_paths:
    true_label = os.path.basename(os.path.dirname(img_path))

    results = model.predict(img_path, verbose=False)
    result = results[0]

    pred_class_id = int(result.probs.top1)
    pred_label = model.names[pred_class_id]

    y_true.append(true_label)
    y_pred.append(pred_label)


# ==================================================
# 5. PERFORMANS METRİKLERİ
# ==================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n========== PERFORMANS METRİKLERİ ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# TXT olarak kaydet
metrics_txt_path = os.path.join(OUTPUT_DIR, "performance_metrics.txt")

with open(metrics_txt_path, "w", encoding="utf-8") as f:
    f.write("Sürücü Uyku ve Yorgunluk Tespiti - Performans Metrikleri\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, zero_division=0))

print("Performans metrikleri TXT olarak kaydedildi:", metrics_txt_path)


# CSV olarak kaydet
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics_csv_path = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print("Performans metrikleri CSV olarak kaydedildi:", metrics_csv_path)


# ==================================================
# 6. PERFORMANS METRİKLERİ GRAFİĞİ
# ==================================================

plt.figure(figsize=(8, 6))
plt.bar(metrics_df["Metric"], metrics_df["Value"])
plt.ylim(0, 1)
plt.title("Model Performans Metrikleri")
plt.xlabel("Metrik")
plt.ylabel("Skor")

for i, value in enumerate(metrics_df["Value"]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=11)

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

performance_graph_path = os.path.join(OUTPUT_DIR, "performance_metrics.png")
plt.savefig(performance_graph_path, dpi=300)
plt.close()

print("Performans metrikleri grafiği kaydedildi:", performance_graph_path)


# ==================================================
# 7. CONFUSION MATRIX OLUŞTUR
# ==================================================

labels = sorted(list(set(y_true + y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = range(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

confusion_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()

print("Confusion matrix kaydedildi:", confusion_matrix_path)


# ==================================================
# 8. ULTRALYTICS'İN OLUŞTURDUĞU GRAFİKLERİ DE KOPYALA
# ==================================================

ultralytics_files = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png"
]

for file_name in ultralytics_files:
    source_path = os.path.join(str(save_dir), file_name)

    if os.path.exists(source_path):
        target_path = os.path.join(OUTPUT_DIR, f"ultralytics_{file_name}")
        shutil.copy(source_path, target_path)
        print(f"{file_name} kopyalandı:", target_path)


print("\nTüm grafikler ve metrikler başarıyla oluşturuldu.")
print("Çıktı klasörü:", OUTPUT_DIR)