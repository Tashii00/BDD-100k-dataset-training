import json
import os
import shutil
from pathlib import Path

# ── CLASS MAPPING ─────────────────────────────────────────────────────────────
CLASSES = {
    "car": 0, "traffic sign": 1, "traffic light": 2,
    "person": 3, "truck": 4, "bus": 5,
    "bike": 6, "rider": 6,
}

IMAGE_W, IMAGE_H = 1280, 720

BASE = "/home/aliraza/BDD-100k-dataset-training/BDD-100K-DATASET-TRAINING/archive (8)"

IMAGE_FOLDERS = {
    "train": [
        f"{BASE}/bdd100k/bdd100k/images/100k/train",
        f"{BASE}/bdd100k/bdd100k/images/100k/train/trainA",
        f"{BASE}/bdd100k/bdd100k/images/100k/train/trainB",
        f"{BASE}/bdd100k/bdd100k/images/10k/train",
    ],
    "val": [
        f"{BASE}/bdd100k/bdd100k/images/100k/val",
        f"{BASE}/bdd100k/bdd100k/images/10k/val",
    ]
}

LABEL_FOLDER = f"{BASE}/bdd100k_labels_release/bdd100k/labels"

def find_image(img_name, split):
    for folder in IMAGE_FOLDERS[split]:
        path = os.path.join(folder, img_name)
        if os.path.exists(path):
            return path
    return None

# Clean previous dataset
if Path("dataset").exists():
    shutil.rmtree("dataset")
    print("Cleaned old dataset folder")

for split in ["train", "val"]:
    label_path = f"{LABEL_FOLDER}/bdd100k_labels_images_{split}_cleaned.json"

    out_labels = Path(f"dataset/labels/{split}")
    out_images = Path(f"dataset/images/{split}")
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    with open(label_path, "r") as f:
        data = json.load(f)

    saved, skipped = 0, 0

    for item in data:
        img_name = item["name"]
        img_path = find_image(img_name, split)

        if not img_path:
            skipped += 1
            continue

        label_lines = []
        if "labels" in item and item["labels"]:
            for label in item["labels"]:
                cat = label["category"]
                if cat not in CLASSES or "box2d" not in label:
                    continue
                box = label["box2d"]
                x_center = ((box["x1"] + box["x2"]) / 2) / IMAGE_W
                y_center = ((box["y1"] + box["y2"]) / 2) / IMAGE_H
                width    = (box["x2"] - box["x1"]) / IMAGE_W
                height   = (box["y2"] - box["y1"]) / IMAGE_H
                label_lines.append(f"{CLASSES[cat]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_file = out_labels / (Path(img_name).stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))

        shutil.copy2(img_path, out_images / img_name)
        saved += 1

        if saved % 5000 == 0:
            print(f"[{split}] Processed {saved:,} images...")

    print(f"\n[{split}] Done! Saved: {saved:,} | Skipped: {skipped:,}")

with open("data.yaml", "w") as f:
    f.write("""path: ./dataset
train: images/train
val: images/val

nc: 7
names:
  0: car
  1: traffic sign
  2: traffic light
  3: person
  4: truck
  5: bus
  6: cyclist
""")

print("\ndata.yaml saved!")
print("Dataset ready for YOLO training!")