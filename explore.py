import json
from collections import Counter

# Load the training labels
with open(r"archive (8)\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json", "r") as f:
    data = json.load(f)

# Count classes and bounding boxes
class_counter = Counter()
images_per_class = Counter()

for item in data:
    if "labels" in item and item["labels"]:
        seen_in_image = set()
        for label in item["labels"]:
            cat = label["category"]
            class_counter[cat] += 1  # total bounding boxes
            seen_in_image.add(cat)
        for cat in seen_in_image:
            images_per_class[cat] += 1  # images containing this class

print(f"Total images: {len(data):,}")
print(f"Number of unique classes: {len(class_counter)}")
print(f"\n{'Class':<25} {'Bounding Boxes':>15} {'Images Containing':>18} {'Avg BBox/Image':>15}")
print("-" * 75)
for cls, count in sorted(class_counter.items(), key=lambda x: -x[1]):
    imgs = images_per_class[cls]
    avg = count / imgs if imgs > 0 else 0
    print(f"{cls:<25} {count:>15,} {imgs:>18,} {avg:>15.2f}")

print("-" * 75)
print(f"{'TOTAL':<25} {sum(class_counter.values()):>15,}")