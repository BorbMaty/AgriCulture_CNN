import cv2
import torch

# ==== 🔧 Beállítások ====
MODEL_PATH = '/home/lucy/Desktop/brassai_projekt/yolov5/runs/train/Agriculture_CNN/weights/best.pt'
CONF_THRESHOLD = 0.4
VIDEO_SOURCE = 0  # 0 = alapértelmezett kamera, vagy pl. "video.mp4"

# ==== 🚀 Modell betöltése ====
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = CONF_THRESHOLD

# ==== 🎥 Kamera vagy videófájl megnyitása ====
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Nem sikerült megnyitni a kamerát vagy videófájlt.")
    exit()

# ==== 🪟 Egy ablak, újraméretezhető ====
cv2.namedWindow("Detekció", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# ==== 🔁 Feldolgozás ====
while True:
    ret, frame = cap.read()
    if not ret:
        print("📭 Nincs több képkocka vagy hiba történt.")
        break

    # 🔍 Detektálás
    results = model(frame)

    # 🖼️ Kirajzolt képkép
    annotated_frame = results.render()[0]
    annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

    # 🪞 Megjelenítés egyetlen ablakban
    try:
        cv2.imshow("Detekció", frame)
    except cv2.error:
        print("🚪 Ablak bezárva. Kilépés.")
        break

    # ❌ Kilépés ha ablak bezáródott vagy 'x' gomb
    if cv2.getWindowProperty("Detekció", cv2.WND_PROP_VISIBLE) < 1:
        print("🧼 Ablak kézzel bezárva.")
        break
    if cv2.waitKey(1) & 0xFF == ord('x'):
        print("🧻 'x' megnyomva.")
        break

# ==== 🧹 Erőforrások felszabadítása ====
cap.release()
cv2.destroyAllWindows()
