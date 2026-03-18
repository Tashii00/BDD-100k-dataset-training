import os
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

# Increased targets — previous values were too aggressive
# Car/traffic sign/traffic light/person were underrepresented → low recall
BOX_TARGETS = {
    0: 150_000,  # car          — was 80k, caused 20k background FN
    1: 100_000,  # traffic sign — was 70k
    2: 100_000,  # traffic light — was 60k, only 8% recall!
    3: 100_000,  # person       — was 60k
    4: None,     # truck        → keep all
    5: None,     # bus          → keep all
    6: None,     # cyclist      → keep all
}

RANDOM_SEED = 42

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    # Clean old output
    if OUT_IMGS.exists():
        shutil.rmtree(OUT_IMGS)
    if OUT_LABELS.exists():
        shutil.rmtree(OUT_LABELS)

    label_files = sorted(SRC_LABELS.glob("*.txt"))
    print(f"Label files found: {len(label_files):,}\n")

    # ── Step 1: Load all labels + collect box refs per class ─────────────────
    all_labels = {}
    box_refs   = {cid: [] for cid in BOX_TARGETS}

    for lf in label_files:
        lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
        all_labels[lf.stem] = lines
        for i, line in enumerate(lines):
            cid = int(line.split()[0])
            if cid in box_refs:
                box_refs[cid].append((lf.stem, i))

    # ── Step 2: Print counts + decide what to remove ─────────────────────────
    print(f"  {'Class':<20} {'Found':>10}  {'Target':>10}  {'Remove':>10}")
    print("  " + "-" * 54)

    to_remove = set()

    for cid in range(7):
        found  = len(box_refs.get(cid, []))
        target = BOX_TARGETS.get(cid)

        if target is None or found <= target:
            print(f"  {CLASS_NAMES[cid]:<20} {found:>10,}  {'keep all':>10}  {'0':>10}")
        else:
            remove = found - target
            refs   = box_refs[cid]
            random.shuffle(refs)
            for stem, line_idx in refs[target:]:
                to_remove.add((stem, line_idx))
            print(f"  {CLASS_NAMES[cid]:<20} {found:>10,}  {target:>10,}  {remove:>10,}")

    print(f"\n  Total annotations to remove: {len(to_remove):,}\n")

    # ── Step 3: Write output ──────────────────────────────────────────────────
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    final_counts = Counter()
    imgs_written = 0

    for lf in label_files:
        stem  = lf.stem
        lines = all_labels[stem]

        new_lines = []
        for i, line in enumerate(lines):
            if (stem, i) in to_remove:
                continue
            new_lines.append(line)
            final_counts[int(line.split()[0])] += 1

        (OUT_LABELS / lf.name).write_text("\n".join(new_lines))

        for ext in [".jpg", ".jpeg", ".png"]:
            src_img = SRC_IMGS / (stem + ext)
            if src_img.exists():
                shutil.copy2(src_img, OUT_IMGS / src_img.name)
                break

        imgs_written += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"  FINAL CLASS DISTRIBUTION (train_final)")
    print(f"{'='*55}")
    max_cnt = max(final_counts.values(), default=1)
    for cid in range(7):
        name  = CLASS_NAMES[cid]
        cnt   = final_counts.get(cid, 0)
        bar   = "█" * min(30, int(30 * cnt / max_cnt))
        print(f"  {cid} {name:<20} {cnt:>10,}  {bar}")

    print(f"\n  Images written : {imgs_written:,}")

    # ── Update data_final.yaml ────────────────────────────────────────────────
    yaml_path = Path("data_final.yaml")
    yaml_path.write_text(
        f"path: ./dataset_balanced\n"
        f"train: images/train_final\n"
        f"val: images/val\n\n"
        f"nc: 7\n"
        f"names:\n" +
        "".join(f"  {cid}: {name}\n" for cid, name in CLASS_NAMES.items())
    )
    print(f"  data_final.yaml saved")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()