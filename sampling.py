"""
sampling.py — Professional Augmentation Pipeline
=================================================
5 distinct augmentation techniques (not just brightness copies):
  0. Original        — clean baseline
  1. Horizontal flip — mirror scene geometry
  2. Perspective warp — simulate different camera angle / tilt
  3. Motion blur      — camera shake or fast-moving subject
  4. Rain simulation  — adverse weather condition
  5. CLAHE            — low-light / tunnel / underexposed scene

Only bike + rider are augmented (truly rare).
Bus is NOT augmented — 8,993 images is sufficient.
"""

import json
import os
import shutil
import random
import math
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)
np.random.seed(42)

# ── CLASS MAPPING ─────────────────────────────────────────────────────────────
CLASSES = {
    "car": 0, "traffic sign": 1, "traffic light": 2,
    "person": 3, "truck": 4, "bus": 5,
    "bike": 6, "rider": 6,
}
IMAGE_W, IMAGE_H = 1280, 720

# Rare classes to oversample (use original category names from JSON)
RARE_CLASSES = {"rider", "bike", "bus"}
COMMON_LIMIT = 3_500  # max images per dominant class
RARE_LIMIT   = 1_500  # max ORIGINAL images per rare class before augmentation
                       # 1,500 × 5 augmentations = 7,500 per class max

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

CLASS_NAMES = {
    0: "car", 1: "traffic sign", 2: "traffic light",
    3: "person", 4: "truck", 5: "bus", 6: "cyclist",
}

# ── IMAGE FINDERS ─────────────────────────────────────────────────────────────
def find_image(img_name, split):
    for folder in IMAGE_FOLDERS[split]:
        path = os.path.join(folder, img_name)
        if os.path.exists(path):
            return path
    return None

# ── AUGMENTATION FUNCTIONS ────────────────────────────────────────────────────

def aug_flip(img):
    """Horizontal flip — mirror the scene"""
    return cv2.flip(img, 1)

def aug_perspective(img):
    """
    Perspective warp — simulates camera tilt or slight viewpoint change.
    Moves corners randomly by up to 5% of image dimension.
    """
    h, w = img.shape[:2]
    margin = 0.05
    dx = int(w * margin)
    dy = int(h * margin)
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst = np.float32([
        [random.randint(0,dx),       random.randint(0,dy)      ],
        [w-1-random.randint(0,dx),   random.randint(0,dy)      ],
        [w-1-random.randint(0,dx),   h-1-random.randint(0,dy)  ],
        [random.randint(0,dx),       h-1-random.randint(0,dy)  ],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), M

def aug_motion_blur(img):
    """
    Motion blur — simulates camera shake or fast-moving object.
    Random direction: horizontal, diagonal left-right, diagonal right-left.
    """
    kernel_size = random.choice([11, 15, 21])
    kernel      = np.zeros((kernel_size, kernel_size))
    direction   = random.choice(["horizontal", "diagonal_lr", "diagonal_rl"])

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "diagonal_lr":
        np.fill_diagonal(kernel, 1.0)
    else:
        kernel = np.fliplr(np.eye(kernel_size))

    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def aug_rain(img):
    """
    Rain simulation — adds streaks + slight blur + darkening.
    InSight must work outdoors in all weather conditions.
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Darken — overcast sky
    img = cv2.convertScaleAbs(img, alpha=0.75, beta=0)

    # Rain streaks
    num_streaks = random.randint(300, 600)
    for _ in range(num_streaks):
        x1     = random.randint(0, w)
        y1     = random.randint(0, h)
        length = random.randint(10, 25)
        angle  = random.uniform(-20, -10)
        x2     = int(x1 + length * math.sin(math.radians(angle)))
        y2     = int(y1 + length * math.cos(math.radians(angle)))
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)

    # Wet lens blur
    return cv2.GaussianBlur(img, (3, 3), 0)

def aug_clahe(img):
    """
    CLAHE — low-light / tunnel / underexposed scene.
    Enhances local contrast per region — simulates going from sunlight
    into shade, underpass, or indoor corridor (common for blind users).
    """
    img   = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ── LABEL TRANSFORMS ──────────────────────────────────────────────────────────

def labels_flip(lines):
    result = []
    for line in lines:
        p = line.strip().split()
        if not p: continue
        cls, x, y, w, h = p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])
        x = 1.0 - x
        result.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return result

def labels_perspective(lines, M, img_w=1280, img_h=720):
    result = []
    for line in lines:
        p = line.strip().split()
        if not p: continue
        cls, x, y, w, h = p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])
        px, py = x * img_w, y * img_h
        pt  = np.array([[[px, py]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, M)[0][0]
        nx  = max(0.01, min(0.99, dst[0] / img_w))
        ny  = max(0.01, min(0.99, dst[1] / img_h))
        result.append(f"{cls} {nx:.6f} {ny:.6f} {w:.6f} {h:.6f}")
    return result

def labels_unchanged(lines):
    """Motion blur, rain, CLAHE don't move objects"""
    return [l for l in lines if l.strip()]

# ── CONVERT JSON ITEM → YOLO LINES ───────────────────────────────────────────

def item_to_label_lines(item):
    lines = []
    if "labels" in item and item["labels"]:
        for label in item["labels"]:
            cat = label["category"]
            if cat not in CLASSES or "box2d" not in label:
                continue
            box = label["box2d"]
            xc  = ((box["x1"] + box["x2"]) / 2) / IMAGE_W
            yc  = ((box["y1"] + box["y2"]) / 2) / IMAGE_H
            w   = (box["x2"] - box["x1"]) / IMAGE_W
            h   = (box["y2"] - box["y1"]) / IMAGE_H
            lines.append(f"{CLASSES[cat]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines

# ── APPLY AUGMENTATION ────────────────────────────────────────────────────────

def apply_aug(img, base_lines, aug_name):
    if aug_name == "flip":
        return aug_flip(img), labels_flip(base_lines)

    elif aug_name == "perspective":
        aug_img, M = aug_perspective(img)
        return aug_img, labels_perspective(base_lines, M)

    elif aug_name == "motion_blur":
        return aug_motion_blur(img), labels_unchanged(base_lines)

    elif aug_name == "rain":
        return aug_rain(img), labels_unchanged(base_lines)

    elif aug_name == "clahe":
        return aug_clahe(img), labels_unchanged(base_lines)

    return img, base_lines

# ── MAIN ──────────────────────────────────────────────────────────────────────

if Path("dataset_balanced").exists():
    shutil.rmtree("dataset_balanced")
    print("Cleaned old dataset_balanced/")

label_path = f"{BASE}/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train_cleaned.json"
with open(label_path) as f:
    data = json.load(f)

# Group images by class
rare_images   = defaultdict(list)
common_images = defaultdict(list)

for item in data:
    if "labels" not in item or not item["labels"]:
        continue
    cats     = set(l["category"] for l in item["labels"] if l["category"] in CLASSES)
    has_rare = cats & RARE_CLASSES
    if has_rare:
        for cat in has_rare:
            rare_images[cat].append(item)
    else:
        primary = max(cats, key=lambda c: sum(1 for l in item["labels"] if l["category"] == c)) if cats else None
        if primary:
            common_images[primary].append(item)

print("\nCommon-only image counts:")
for cat, items in sorted(common_images.items()):
    print(f"  {cat:<20} {len(items):>6,} images")
print(f"  {'TOTAL':<20} {sum(len(v) for v in common_images.values()):>6,}")

print("\nRare class counts (before aug):")
for cls in RARE_CLASSES:
    print(f"  {cls:<20} {len(rare_images[cls]):>6,} → ×6 (orig+5 aug) = {min(len(rare_images[cls]),RARE_LIMIT)*6:,}")

out_labels = Path("dataset_balanced/labels/train")
out_images = Path("dataset_balanced/images/train")
out_labels.mkdir(parents=True, exist_ok=True)
out_images.mkdir(parents=True, exist_ok=True)

saved         = 0
class_counter = Counter()
rare_seen     = set()

AUG_NAMES = ["flip", "perspective", "motion_blur", "rain", "clahe"]

print("\n--- Augmenting rare classes (bike + rider) ---")
for cat in RARE_CLASSES:
    random.shuffle(rare_images[cat])
    for item in rare_images[cat][:RARE_LIMIT]:
        img_name = item["name"]
        if img_name in rare_seen:
            continue
        rare_seen.add(img_name)

        img_path = find_image(img_name, "train")
        if not img_path:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_lines = item_to_label_lines(item)
        stem       = Path(img_name).stem
        ext        = Path(img_name).suffix

        # Save original
        shutil.copy2(img_path, out_images / img_name)
        with open(out_labels / f"{stem}.txt", "w") as f:
            f.write("\n".join(base_lines))
        for line in base_lines:
            class_counter[int(line.split()[0])] += 1
        saved += 1

        # Save 5 augmented versions — each a different technique
        for aug_name in AUG_NAMES:
            aug_img, aug_lines = apply_aug(img.copy(), base_lines, aug_name)
            cv2.imwrite(str(out_images / f"{stem}_{aug_name}{ext}"), aug_img)
            with open(out_labels / f"{stem}_{aug_name}.txt", "w") as f:
                f.write("\n".join(aug_lines))
            for line in aug_lines:
                class_counter[int(line.split()[0])] += 1
            saved += 1

        if saved % 1000 == 0:
            print(f"  [rare] {saved:,} saved...")

print(f"\nAfter rare augmentation: {saved:,} images")

# Fill with common images
print("\n--- Adding common class images ---")
all_common = []
for cat, items in common_images.items():
    random.shuffle(items)
    all_common.extend(items[:COMMON_LIMIT])
random.shuffle(all_common)

seen_common = set()
for item in all_common:
    if saved >= TOTAL_TARGET:
        break
    img_name = item["name"]
    if img_name in seen_common or img_name in rare_seen:
        continue
    seen_common.add(img_name)

    img_path = find_image(img_name, "train")
    if not img_path:
        continue

    label_lines = item_to_label_lines(item)
    shutil.copy2(img_path, out_images / img_name)
    with open(out_labels / f"{Path(img_name).stem}.txt", "w") as f:
        f.write("\n".join(label_lines))
    for line in label_lines:
        class_counter[int(line.split()[0])] += 1
    saved += 1

# ── TRAIN REPORT ──────────────────────────────────────────────────────────────
print(f"\n{'='*42}")
print(f"  TRAIN FINAL: {saved:,} images")
print(f"{'='*42}")
print(f"  {'Class':<25} {'Boxes':>10}")
print(f"  {'-'*37}")
for cid in range(7):
    print(f"  {CLASS_NAMES[cid]:<25} {class_counter.get(cid,0):>10,}")

# ── VAL (always clean, no augmentation) ───────────────────────────────────────
print("\n--- Processing val (clean, no augmentation) ---")
val_label_path = f"{BASE}/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val_cleaned.json"
with open(val_label_path) as f:
    val_data = json.load(f)

out_val_labels = Path("dataset_balanced/labels/val")
out_val_images = Path("dataset_balanced/images/val")
out_val_labels.mkdir(parents=True, exist_ok=True)
out_val_images.mkdir(parents=True, exist_ok=True)

random.shuffle(val_data)
val_saved   = 0
val_counter = Counter()

for item in val_data:
    if val_saved >= 2_000:
        break
    img_name = item["name"]
    img_path = find_image(img_name, "val")
    if not img_path:
        continue
    label_lines = item_to_label_lines(item)
    shutil.copy2(img_path, out_val_images / img_name)
    with open(out_val_labels / (Path(img_name).stem + ".txt"), "w") as f:
        f.write("\n".join(label_lines))
    for line in label_lines:
        val_counter[int(line.split()[0])] += 1
    val_saved += 1

print(f"\n  VAL: {val_saved:,} images")
print(f"  {'Class':<25} {'Boxes':>10}")
print(f"  {'-'*37}")
for cid in range(7):
    print(f"  {CLASS_NAMES[cid]:<25} {val_counter.get(cid,0):>10,}")

# ── YAML ──────────────────────────────────────────────────────────────────────
with open("data_balanced.yaml", "w") as f:
    f.write("""path: ./dataset_balanced
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

print("\ndata_balanced.yaml saved!")
print("Dataset ready!")