import cv2
import time
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

#BUNU CALISTIRACAKSIN
YOLO_MODEL_PATH = "models/yolo_drowsiness_cls.pt"
EYE_MODEL_PATH = "models/eye_cnn.keras"

IMG_SIZE = 144
CLOSED_LIMIT = 2.0

yolo_model = YOLO(YOLO_MODEL_PATH)
eye_model = load_model(EYE_MODEL_PATH)

print("YOLO modeli yüklendi:", YOLO_MODEL_PATH)
print("Eye CNN modeli yüklendi:", EYE_MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

closed_start = None


def predict_eye(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return None, 0.0

    img = cv2.resize(eye_crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = eye_model.predict(img, verbose=0)[0][0]

    # Class indices: {'Closed': 0, 'Open': 1}
    if pred < 0.5:
        return "Closed", 1.0 - float(pred)
    else:
        return "Open", float(pred)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    frame = cv2.flip(frame, 1)

    # =========================
    # YOLO ile esneme tahmini
    # =========================
    results = yolo_model(frame, verbose=False)
    result = results[0]

    yolo_class_id = int(result.probs.top1)
    yolo_conf = float(result.probs.top1conf)
    yolo_label = result.names[yolo_class_id]

    cv2.putText(
        frame,
        f"YOLO: {yolo_label} ({yolo_conf:.2f})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if yolo_label == "yawn":
        cv2.putText(
            frame,
            "YORGUNLUK: ESNEME ALGILANDI!",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

    # =========================
    # Yüz bulma
    # =========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(120, 120)
    )

    eye_labels = []

    if len(faces) > 0:
        # En büyük yüzü seç
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Yüzün üst kısmından göz bölgelerini yaklaşık kırp
        eye_y1 = y + int(h * 0.22)
        eye_y2 = y + int(h * 0.50)

        left_x1 = x + int(w * 0.08)
        left_x2 = x + int(w * 0.48)

        right_x1 = x + int(w * 0.52)
        right_x2 = x + int(w * 0.92)

        left_eye = frame[eye_y1:eye_y2, left_x1:left_x2]
        right_eye = frame[eye_y1:eye_y2, right_x1:right_x2]

        left_label, left_conf = predict_eye(left_eye)
        right_label, right_conf = predict_eye(right_eye)

        if left_label:
            eye_labels.append(left_label)
            cv2.rectangle(frame, (left_x1, eye_y1), (left_x2, eye_y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"L: {left_label} {left_conf:.2f}",
                (left_x1, eye_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if right_label:
            eye_labels.append(right_label)
            cv2.rectangle(frame, (right_x1, eye_y1), (right_x2, eye_y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"R: {right_label} {right_conf:.2f}",
                (right_x1, eye_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # İki göz de Closed ise kapalı kabul et
        if eye_labels.count("Closed") >= 2:
            if closed_start is None:
                closed_start = time.time()

            elapsed = time.time() - closed_start

            cv2.putText(
                frame,
                f"Goz kapali sure: {elapsed:.1f} sn",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if elapsed >= CLOSED_LIMIT:
                cv2.putText(
                    frame,
                    "UYARI: SURUCU UYKULU!",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )
        else:
            closed_start = None

    else:
        cv2.putText(
            frame,
            "Yuz algilanamadi",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        closed_start = None

    cv2.imshow("BITIRME_3 - YOLO + Haar Eye Crop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()