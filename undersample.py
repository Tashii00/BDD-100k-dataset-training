import random
import shutil
from pathlib import Path
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────

SRC_IMGS   = Path("dataset_balanced/images/train")
SRC_LABELS = Path("dataset_balanced/labels/train")

OUT_IMGS   = Path("dataset_balanced/images/train_final")
OUT_LABELS = Path("dataset_balanced/labels/train_final")

CLASS_NAMES = {
    0: "car", 1: "traffic sign", 2: "traffic light", 3: "person",
    4: "truck", 5: "bus", 6: "cyclist",
}

RARE_CLASSES    = {4, 5, 6}       # truck, bus, cyclist — never remove these images
CAR_TARGET      = 167_000         # realistic floor — car is in nearly every scene
CAR_DOMINANT_THRESHOLD = 0.70     # image flagged if car boxes > 70% of all boxes
MIN_BOXES_TO_FLAG      = 3        # only flag images with at least this many boxes
RANDOM_SEED     = 42

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    label_files = sorted(SRC_LABELS.glob("*.txt"))
    print(f"Label files found: {len(label_files):,}\n")

    # ── Step 1: Load all labels ───────────────────────────────────────────────
    all_labels = {}   # stem → list of lines
    for lf in label_files:
        lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
        all_labels[lf.stem] = lines

    # ── Step 2: Separate rare-containing vs car-dominant images ──────────────
    protected  = []   # contains any rare class → always keep
    flaggable  = []   # car-dominant, no rare class → candidate for removal
    neutral    = []   # mixed but not car-dominant → always keep

    for stem, lines in all_labels.items():
        if not lines:
            neutral.append(stem)
            continue

        class_ids = [int(l.split()[0]) for l in lines]
        counts    = Counter(class_ids)
        total     = len(lines)
        car_frac  = counts.get(0, 0) / total
        has_rare  = any(cid in RARE_CLASSES for cid in counts)

        if has_rare:
            protected.append(stem)
        elif total >= MIN_BOXES_TO_FLAG and car_frac >= CAR_DOMINANT_THRESHOLD:
            flaggable.append(stem)
        else:
            neutral.append(stem)

    print(f"  Protected  (rare class present):  {len(protected):,}  → never removed")
    print(f"  Neutral    (mixed, no rare):       {len(neutral):,}   → always kept")
    print(f"  Flaggable  (car-dominant):         {len(flaggable):,}  → candidates for removal\n")

    # ── Step 3: Count car boxes in current set ───────────────────────────────
    def count_cars(stems):
        total = 0
        for stem in stems:
            for line in all_labels.get(stem, []):
                if int(line.split()[0]) == 0:
                    total += 1
        return total

    car_in_protected = count_cars(protected)
    car_in_neutral   = count_cars(neutral)
    car_in_flaggable = count_cars(flaggable)
    total_car        = car_in_protected + car_in_neutral + car_in_flaggable

    print(f"  Car boxes in protected images:  {car_in_protected:,}")
    print(f"  Car boxes in neutral images:    {car_in_neutral:,}")
    print(f"  Car boxes in flaggable images:  {car_in_flaggable:,}")
    print(f"  Total car boxes:                {total_car:,}")
    print(f"  Target car boxes:               {CAR_TARGET:,}\n")

    # ── Step 4: Check if removal is even possible ────────────────────────────
    min_possible = car_in_protected + car_in_neutral
    if min_possible > CAR_TARGET:
        print(f"  WARNING: Even removing ALL flaggable images gives {min_possible:,} car boxes.")
        print(f"  Target {CAR_TARGET:,} is unreachable. Removing all flaggable images.\n")
        kept_flaggable = []
    elif total_car <= CAR_TARGET:
        print(f"  Already at or below target — keeping all flaggable images.\n")
        kept_flaggable = flaggable
    else:
        # How many car boxes to shed from flaggable pool
        need_to_shed = total_car - CAR_TARGET

        # Sort flaggable by car box count descending — remove car-heaviest first
        flaggable_with_counts = [
            (stem, sum(1 for l in all_labels[stem] if int(l.split()[0]) == 0))
            for stem in flaggable
        ]
        flaggable_with_counts.sort(key=lambda x: x[1], reverse=True)

        shed = 0
        removed = set()
        for stem, car_count in flaggable_with_counts:
            if shed >= need_to_shed:
                break
            removed.add(stem)
            shed += car_count

        kept_flaggable = [s for s in flaggable if s not in removed]
        print(f"  Removing {len(removed):,} car-dominant images (shed {shed:,} car boxes)")
        print(f"  Keeping  {len(kept_flaggable):,} flaggable images\n")

    # ── Step 5: Write output ──────────────────────────────────────────────────
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    final_stems  = protected + neutral + kept_flaggable
    final_counts = Counter()
    imgs_written = 0

    for stem in final_stems:
        lines = all_labels.get(stem, [])

        # Find image file
        src_img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = SRC_IMGS / (stem + ext)
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            continue

        shutil.copy2(src_img, OUT_IMGS / src_img.name)
        (OUT_LABELS / (stem + ".txt")).write_text("\n".join(lines))

        for line in lines:
            final_counts[int(line.split()[0])] += 1
        imgs_written += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"  FINAL CLASS DISTRIBUTION (train_final)")
    print(f"{'='*55}")
    cyclist_cnt = final_counts.get(6, 1)
    for cid in range(7):
        name  = CLASS_NAMES[cid]
        cnt   = final_counts.get(cid, 0)
        ratio = f"{cnt/cyclist_cnt:.1f}x"
        bar   = "█" * min(30, int(30 * cnt / max(final_counts.values(), default=1)))
        print(f"  {cid} {name:<20} {cnt:>10,}  {ratio:>10}  {bar}")

    print(f"\n  Images written : {imgs_written:,}")
    print(f"  Labels written : {imgs_written:,}")

    # ── Save yaml ─────────────────────────────────────────────────────────────
    Path("data_final.yaml").write_text(
        "path: ./dataset_balanced\n"
        "train: images/train_final\n"
        "val: images/val\n\n"
        "nc: 7\n"
        "names:\n" +
        "".join(f"  {cid}: {name}\n" for cid, name in CLASS_NAMES.items())
    )
    print(f"  data_final.yaml saved")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()