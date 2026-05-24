import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# AYARLAR
# =========================

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

OUTPUT_DIR = "outputs/yawn_cnn_metrics"
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yawn_cnn.keras")

IMG_SIZE = 144
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5

CLASS_NAMES = ["no_yawn", "yawn"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# DATA GENERATOR
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="binary",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="binary",
    shuffle=False
)


print("Sınıf indeksleri:", train_generator.class_indices)


# =========================
# CNN MODELİ
# =========================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_OUTPUT_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# =========================
# EĞİTİM
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("Eğitim tamamlandı.")
print("Model kaydedildi:", MODEL_OUTPUT_PATH)


# =========================
# ACCURACY - LOSS GRAFİĞİ
# =========================

plt.figure(figsize=(10, 6))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Yawn CNN Accuracy ve Loss Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Değer")
plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_loss_path = os.path.join(OUTPUT_DIR, "accuracy_loss_graph.png")
plt.savefig(accuracy_loss_path, dpi=300)
plt.close()

print("Accuracy/Loss grafiği kaydedildi:", accuracy_loss_path)


# =========================
# TEST ÜZERİNDE TAHMİN
# =========================

test_generator.reset()

pred_probs = model.predict(test_generator)
y_pred = (pred_probs > 0.5).astype(int).reshape(-1)

y_true = test_generator.classes

label_names = CLASS_NAMES


# =========================
# PERFORMANS METRİKLERİ
# =========================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n========== YAWN CNN PERFORMANS METRİKLERİ ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=label_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# =========================
# METRİKLERİ TXT KAYDET
# =========================

metrics_txt_path = os.path.join(OUTPUT_DIR, "performance_metrics.txt")

with open(metrics_txt_path, "w", encoding="utf-8") as f:
    f.write("Esneme Tespiti CNN Modeli - Performans Metrikleri\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Performans metrikleri TXT olarak kaydedildi:", metrics_txt_path)


# =========================
# METRİKLERİ CSV KAYDET
# =========================

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics_csv_path = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print("Performans metrikleri CSV olarak kaydedildi:", metrics_csv_path)


# =========================
# PERFORMANS METRİKLERİ GRAFİĞİ
# =========================

plt.figure(figsize=(8, 6))
plt.bar(metrics_df["Metric"], metrics_df["Value"])
plt.ylim(0, 1)

plt.title("Yawn CNN Performans Metrikleri")
plt.xlabel("Metrik")
plt.ylabel("Skor")

for i, value in enumerate(metrics_df["Value"]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

performance_graph_path = os.path.join(OUTPUT_DIR, "performance_metrics.png")
plt.savefig(performance_graph_path, dpi=300)
plt.close()

print("Performans metrikleri grafiği kaydedildi:", performance_graph_path)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Yawn CNN Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(label_names))
plt.xticks(tick_marks, label_names)
plt.yticks(tick_marks, label_names)

plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

for i in range(len(label_names)):
    for j in range(len(label_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

confusion_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()

print("Confusion matrix kaydedildi:", confusion_matrix_path)


print("\nTüm metrikler başarıyla oluşturuldu.")
print("Çıktı klasörü:", OUTPUT_DIR)