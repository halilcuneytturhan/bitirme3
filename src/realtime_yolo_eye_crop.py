import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model


# =========================
# Model yolları
# =========================

YOLO_MODEL_PATH = "models/yolo_drowsiness_cls.pt"
EYE_MODEL_PATH = "models/eye_cnn.keras"

IMG_SIZE = 144

# eye_cnn class sırası:
# {'Closed': 0, 'Open': 1}
EYE_CLASS_NAMES = ["Closed", "Open"]

CLOSED_LIMIT = 2.0


# =========================
# Modelleri yükle
# =========================

yolo_model = YOLO(YOLO_MODEL_PATH)
eye_model = load_model(EYE_MODEL_PATH)

print("YOLO modeli yüklendi:", YOLO_MODEL_PATH)
print("Eye CNN modeli yüklendi:", EYE_MODEL_PATH)


# =========================
# MediaPipe Face Mesh
# =========================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Göz landmark indexleri
LEFT_EYE_IDX = [33, 133, 159, 145, 153, 154, 155, 246]
RIGHT_EYE_IDX = [362, 263, 386, 374, 380, 381, 382, 466]


def crop_eye(frame, landmarks, eye_indices, margin=20):
    h, w, _ = frame.shape

    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, w)
    y2 = min(max(ys) + margin, h)

    eye_crop = frame[y1:y2, x1:x2]

    return eye_crop, (x1, y1, x2, y2)


def predict_eye(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return None, 0.0

    img = cv2.resize(eye_crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = eye_model.predict(img, verbose=0)[0][0]

    # Binary sigmoid:
    # 0'a yakın Closed, 1'e yakın Open
    if pred < 0.5:
        label = "Closed"
        conf = 1.0 - float(pred)
    else:
        label = "Open"
        conf = float(pred)

    return label, conf


# =========================
# Kamera
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı. 0 yerine 1 veya 2 deneyin.")
    exit()

closed_start = None


while True:
    ret, frame = cap.read()

    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    frame = cv2.flip(frame, 1)

    # =========================
    # YOLO ile yawn tahmini
    # =========================

    yolo_results = yolo_model(frame, verbose=False)
    yolo_result = yolo_results[0]

    yolo_class_id = int(yolo_result.probs.top1)
    yolo_conf = float(yolo_result.probs.top1conf)
    yolo_label = yolo_result.names[yolo_class_id]

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
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

    # =========================
    # MediaPipe ile göz crop
    # =========================

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)

    eye_labels = []

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        left_eye_crop, left_box = crop_eye(frame, landmarks, LEFT_EYE_IDX)
        right_eye_crop, right_box = crop_eye(frame, landmarks, RIGHT_EYE_IDX)

        left_label, left_conf = predict_eye(left_eye_crop)
        right_label, right_conf = predict_eye(right_eye_crop)

        if left_label:
            eye_labels.append(left_label)
            x1, y1, x2, y2 = left_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"L: {left_label} {left_conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if right_label:
            eye_labels.append(right_label)
            x1, y1, x2, y2 = right_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"R: {right_label} {right_conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # İki gözden biri Closed ise kapalı kabul et
        if "Closed" in eye_labels:
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

    cv2.imshow("BITIRME_3 - YOLO + Eye Crop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()