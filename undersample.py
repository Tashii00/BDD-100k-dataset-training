"""
undersample.py — Image-Level Undersampling
==========================================
NO box removal — whole images removed.

Target: bring car, traffic sign, traffic light box counts
down to ~125,000 (same as person class).

Only removes images that contain ONLY dominant classes
(car, traffic sign, traffic light) with NO protected classes
(person, truck, bus, cyclist).
"""

import random
import shutil
from pathlib import Path
from collections import Counter

SRC_IMGS   = Path("dataset_balanced/images/train")
SRC_LABELS = Path("dataset_balanced/labels/train")
OUT_IMGS   = Path("dataset_balanced/images/train_final")
OUT_LABELS = Path("dataset_balanced/labels/train_final")

CLASS_NAMES = {
    0: "car", 1: "traffic sign", 2: "traffic light", 3: "person",
    4: "truck", 5: "bus", 6: "cyclist",
}

DOMINANT_CLASSES  = {0, 1, 2}
PROTECTED_CLASSES = {3, 4, 5, 6}
TARGET_BOX        = 125_000
RANDOM_SEED       = 42

def main():
    random.seed(RANDOM_SEED)

    if OUT_IMGS.exists():
        shutil.rmtree(OUT_IMGS)
    if OUT_LABELS.exists():
        shutil.rmtree(OUT_LABELS)

    label_files = sorted(SRC_LABELS.glob("*.txt"))
    print(f"Label files found: {len(label_files):,}\n")

    all_labels    = {}
    image_classes = {}

    for lf in label_files:
        lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
        all_labels[lf.stem]    = lines
        image_classes[lf.stem] = set(int(l.split()[0]) for l in lines if l.strip())

    before = Counter()
    for lines in all_labels.values():
        for line in lines:
            before[int(line.split()[0])] += 1

    print("  Before undersampling:")
    for cid in range(7):
        print(f"  {CLASS_NAMES[cid]:<22} {before.get(cid,0):>10,}")

    protected_images = []
    dominant_only    = []

    for stem, classes in image_classes.items():
        if classes & PROTECTED_CLASSES:
            protected_images.append(stem)
        elif classes & DOMINANT_CLASSES:
            dominant_only.append(stem)
        else:
            protected_images.append(stem)

    print(f"\n  Protected images (always keep): {len(protected_images):,}")
    print(f"  Dominant-only candidates:       {len(dominant_only):,}")

    protected_counts = Counter()
    for stem in protected_images:
        for line in all_labels[stem]:
            protected_counts[int(line.split()[0])] += 1

    print(f"\n  Dominant boxes from protected images:")
    for cid in DOMINANT_CLASSES:
        gap = TARGET_BOX - protected_counts.get(cid, 0)
        print(f"  {CLASS_NAMES[cid]:<22} {protected_counts.get(cid,0):>10,}  (need {max(0,gap):,} more)")

    random.shuffle(dominant_only)
    running       = Counter(protected_counts)
    kept_dominant = []

    for stem in dominant_only:
        img_counts = Counter(int(l.split()[0]) for l in all_labels[stem] if l.strip())
        needs_more = any(
            running.get(cid, 0) < TARGET_BOX
            for cid in DOMINANT_CLASSES
            if cid in img_counts
        )
        if needs_more:
            kept_dominant.append(stem)
            for cid, cnt in img_counts.items():
                running[cid] += cnt

    removed     = len(dominant_only) - len(kept_dominant)
    final_stems = set(protected_images) | set(kept_dominant)

    print(f"\n  Dominant-only kept:    {len(kept_dominant):,}")
    print(f"  Dominant-only removed: {removed:,}")
    print(f"  Final total images:    {len(final_stems):,}")

    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    final_counts = Counter()
    written = 0

    for lf in label_files:
        stem = lf.stem
        if stem not in final_stems:
            continue
        lines = all_labels[stem]
        (OUT_LABELS / lf.name).write_text("\n".join(lines))
        for line in lines:
            final_counts[int(line.split()[0])] += 1
        for ext in [".jpg", ".jpeg", ".png"]:
            src_img = SRC_IMGS / (stem + ext)
            if src_img.exists():
                shutil.copy2(src_img, OUT_IMGS / src_img.name)
                break
        written += 1

    print(f"\n{'='*60}")
    print(f"  FINAL CLASS DISTRIBUTION (train_final)")
    print(f"{'='*60}")
    print(f"  {'Class':<22} {'Before':>10}  {'After':>10}  {'Change':>8}")
    print("  " + "-"*56)
    max_cnt = max(final_counts.values(), default=1)
    for cid in range(7):
        name = CLASS_NAMES[cid]
        bef  = before.get(cid, 0)
        aft  = final_counts.get(cid, 0)
        diff = aft - bef
        bar  = "█" * min(25, int(25 * aft / max_cnt))
        print(f"  {name:<22} {bef:>10,}  {aft:>10,}  {diff:>+8,}  {bar}")

    print(f"\n  Images written: {written:,}")

    Path("data_final.yaml").write_text(
        "path: ./dataset_balanced\n"
        "train: images/train_final\n"
        "val: images/val\n\n"
        "nc: 7\n"
        "names:\n" +
        "".join(f"  {cid}: {name}\n" for cid, name in CLASS_NAMES.items())
    )
    print("  data_final.yaml saved")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()