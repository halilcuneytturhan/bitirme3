import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "models/yolo_drowsiness_cls.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı. 0 yerine 1 veya 2 deneyin.")
    exit()

closed_start = None
CLOSED_LIMIT = 2.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    results = model(frame, verbose=False)
    result = results[0]

    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = result.names[class_id]

    cv2.putText(
        frame,
        f"Tahmin: {label} ({confidence:.2f})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    if label == "Closed":
        if closed_start is None:
            closed_start = time.time()

        elapsed = time.time() - closed_start

        cv2.putText(
            frame,
            f"Goz kapali: {elapsed:.1f} sn",
            (30, 80),
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

    if label == "yawn":
        cv2.putText(
            frame,
            "YORGUNLUK: ESNEME ALGILANDI!",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2
        )

    cv2.imshow("YOLO Classification - Surucu Durumu", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()