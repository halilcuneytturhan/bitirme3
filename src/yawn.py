import os
import tensorflow as tf
import visualkeras
from tensorflow.keras.models import load_model
from PIL import ImageFont


# =========================
# DOSYA YOLLARI
# =========================
MODEL_PATH = r"C:\Users\cavus\Desktop\bitirme3\outputs\yawn_cnn_metrics\yawn_cnn.keras"
OUTPUT_DIR = "outputs/model_visualization"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yawn_cnn_architecture.png")


# =========================
# KLASÖR OLUŞTUR
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# MODELİ YÜKLE
# =========================
model = load_model(MODEL_PATH)

print("CNN modeli yüklendi:", MODEL_PATH)
print("\nModel Özeti:")
model.summary()


# =========================
# OUTPUT_SHAPE DÜZELTME
# =========================
# VisualKeras bazı Keras sürümlerinde layer.output_shape bekliyor.
# O yüzden eksikse manuel olarak ekliyoruz.
for layer in model.layers:
    try:
        if not hasattr(layer, "output_shape"):
            layer.output_shape = tf.keras.backend.int_shape(layer.output)
    except Exception as e:
        print(f"{layer.name} için output_shape alınamadı: {e}")


# =========================
# FONT AYARI
# =========================
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = None


# =========================
# GÖRSELLEŞTİRME
# =========================
img = visualkeras.layered_view(
    model,
    legend=True,
    font=font,
    spacing=30,
    scale_xy=1.2,
    scale_z=1.0,
    max_z=120,
    draw_volume=True,
    to_file=OUTPUT_PATH
)
print("\nModel mimarisi kaydedildi:")
print(OUTPUT_PATH)