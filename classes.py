import json
from collections import Counter

for split in ["train", "val"]:
    path = rf"archive (8)\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_{split}_cleaned.json"
    
    with open(path, "r") as f:
        data = json.load(f)

    class_counter = Counter()
    for item in data:
        if "labels" in item and item["labels"]:
            for label in item["labels"]:
                class_counter[label["category"]] += 1

    print(f"\n[{split}] Total images: {len(data):,}")
    print(f"Total bounding boxes: {sum(class_counter.values()):,}")
    print(f"\n{'Class':<20} {'Bounding Boxes':>15}")
    print("-" * 37)
    for cls, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"{cls:<20} {count:>15,}")