# INT8 quantized YOLOv11n model — exported from FP32 best.pt using OpenVINO NNCF calibration.
# Model size reduced from 5.4MB (FP32) to 3.2MB (INT8), ~2x faster inference on CPU.
# Trained on BDD100K dataset, 7 classes: car, traffic sign, traffic light, person, truck, bus, cyclist.
# Tested on real Pakistan road conditions — Islamabad/Rawalpindi.



import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_int8_openvino_model"
IMAGE_FOLDER = "."          # current folder — same folder as images
CONF_THRESH  = 0.5
IMGSZ        = 640

CLASS_NAMES = {
    0: "car", 1: "traffic sign", 2: "traffic light",
    3: "person", 4: "truck", 5: "bus", 6: "cyclist",
}

CLASS_COLORS = {
    0: (255, 0,   0  ),   # car           — blue
    1: (0,   255, 255),   # traffic sign  — yellow
    2: (0,   255, 0  ),   # traffic light — green
    3: (255, 0,   255),   # person        — magenta
    4: (0,   165, 255),   # truck         — orange
    5: (128, 0,   128),   # bus           — purple
    6: (0,   0,   255),   # cyclist       — red
}

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print(f"Loading INT8 model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ── PROCESS ALL IMAGES IN FOLDER ──────────────────────────────────────────────
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
image_files = [
    f for f in Path(IMAGE_FOLDER).iterdir()
    if f.suffix.lower() in image_extensions
]

print(f"Found {len(image_files)} images\n")

for img_path in sorted(image_files):
    print(f"Processing: {img_path.name}")

    results = model.predict(
        source  = str(img_path),
        imgsz   = IMGSZ,
        conf    = CONF_THRESH,
        iou     = 0.45,
        verbose = False,
    )

    img    = cv2.imread(str(img_path))
    result = results[0]

    for box in result.boxes:
        cls_id       = int(box.cls[0])
        conf         = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name     = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color        = CLASS_COLORS.get(cls_id, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label        = f"{cls_name} {conf:.2f}"
        (tw, th), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save result in same folder with _result suffix
    output_path = img_path.parent / f"{img_path.stem}_result{img_path.suffix}"
    cv2.imwrite(str(output_path), img)

    print(f"  Detected: {len(result.boxes)} objects → saved: {output_path.name}")

print("\nDone!")
