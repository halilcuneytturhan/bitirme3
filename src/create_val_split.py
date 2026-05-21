import os
import random
import shutil
from pathlib import Path


# =========================
# AYARLAR
# =========================

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# Train içinden yüzde kaç val ayrılacak?
VAL_RATIO = 0.20

# Aynı sonucu almak için sabit değer
RANDOM_SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =========================
# KONTROLLER
# =========================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Train klasörü bulunamadı: {TRAIN_DIR}")

if VAL_DIR.exists():
    print(f"Uyarı: {VAL_DIR} zaten var.")
    print("Tekrar bölme yapmamak için işlem durduruldu.")
    print("Yeniden oluşturmak istiyorsan data/val klasörünü silip tekrar çalıştır.")
    exit()


random.seed(RANDOM_SEED)
VAL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# TRAIN'DEN VAL AYIR
# =========================

class_dirs = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]

if len(class_dirs) == 0:
    raise FileNotFoundError("data/train içinde sınıf klasörü bulunamadı.")

print("Sınıflar:")
for class_dir in class_dirs:
    print("-", class_dir.name)

print("\nVal ayırma işlemi başladı...\n")

for class_dir in class_dirs:
    class_name = class_dir.name
    val_class_dir = VAL_DIR / class_name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)

    # Çok az görsel varsa en az 1 tane ayır
    if len(images) > 0 and val_count == 0:
        val_count = 1

    val_images = images[:val_count]

    for img_path in val_images:
        target_path = val_class_dir / img_path.name
        shutil.move(str(img_path), str(target_path))

    print(f"{class_name}: {len(images)} görüntü vardı, {val_count} tanesi val klasörüne taşındı.")

print("\nİşlem tamamlandı.")
print("Yeni yapı:")
print("data/train -> eğitim")
print("data/val   -> doğrulama")
print("data/test  -> final test")