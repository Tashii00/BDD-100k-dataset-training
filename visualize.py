import json
import random
import os
import cv2

with open(r"archive (8)\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json", "r") as f:
    data = json.load(f)

# Search in both 10k and 100k folders
IMAGE_FOLDERS = [
    r"archive (8)\bdd100k\bdd100k\images\100k\train",
    r"archive (8)\bdd100k\bdd100k\images\100k\val",
    r"archive (8)\bdd100k\bdd100k\images\10k\train",
    r"archive (8)\bdd100k\bdd100k\images\10k\val",
]

OUTPUT_PATH = "visualized_samples"
os.makedirs(OUTPUT_PATH, exist_ok=True)

COLORS = {
    "car":           (0, 255, 0),
    "lane":          (255, 255, 0),
    "traffic sign":  (0, 0, 255),
    "traffic light": (255, 0, 255),
    "drivable area": (0, 165, 255),
    "person":        (255, 255, 255),
    "truck":         (0, 255, 255),
    "bus":           (128, 0, 255),
    "bike":          (255, 128, 0),
    "rider":         (0, 128, 255),
    "motor":         (128, 255, 0),
    "train":         (0, 0, 128),
}

def find_image(img_name):
    for folder in IMAGE_FOLDERS:
        path = os.path.join(folder, img_name)
        if os.path.exists(path):
            return path
    return None

# Pick 20 random samples and try to find 10 valid ones
random.shuffle(data)
saved = 0

for item in data:
    if saved >= 10:
        break

    img_name = item["name"]
    img_path = find_image(img_name)

    if not img_path:
        continue

    img = cv2.imread(img_path)

    if "labels" in item and item["labels"]:
        for label in item["labels"]:
            cat = label["category"]
            color = COLORS.get(cat, (255, 255, 255))
            if "box2d" in label:
                box = label["box2d"]
                x1, y1 = int(box["x1"]), int(box["y1"])
                x2, y2 = int(box["x2"]), int(box["y2"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, cat, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = os.path.join(OUTPUT_PATH, img_name)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
    saved += 1

print(f"\nDone! Saved {saved} images to '{OUTPUT_PATH}' folder.")