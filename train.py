if __name__ == "__main__":

    import torch
    import shutil
    from torch import nn
    from pathlib import Path
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.loss import v8DetectionLoss

    # ── Box counts from undersample.py output (train_final) ───────────────────
    box_counts = torch.tensor([
        166_997,  # 0 car
         84_574,  # 1 traffic sign
         78_605,  # 2 traffic light
         58_978,  # 3 person
         10_304,  # 4 truck
          7_795,  # 5 bus
         18_625,  # 6 cyclist
    ], dtype=torch.float32)

    CLASS_WEIGHTS = torch.clamp(box_counts.max() / box_counts, max=10.0)

    print("\nClass weights:")
    names = ["car", "traffic sign", "traffic light", "person", "truck", "bus", "cyclist"]
    for name, w in zip(names, CLASS_WEIGHTS):
        print(f"  {name:<15}: {w:.2f}x")

    # ── Weighted loss ──────────────────────────────────────────────────────────
    class WeightedDetectionLoss(v8DetectionLoss):
        def __init__(self, model, tal_topk=10):
            super().__init__(model, tal_topk=tal_topk)
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=CLASS_WEIGHTS.to(self.device),
                reduction="none",
            )

    class WeightedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        def get_validator(self):
            return super().get_validator()

        def criterion(self, preds, batch):
            if not hasattr(self, "weighted_loss"):
                self.weighted_loss = WeightedDetectionLoss(self.model)
            return self.weighted_loss(preds, batch)

    DATA_YAML = "data_final.yaml"
    PROJECT   = "runs/train"

    # ── Common training params ─────────────────────────────────────────────────
    COMMON = dict(
        trainer      = WeightedTrainer,
        data         = DATA_YAML,
        fraction     = 0.3,        # 30% data for quick check
        imgsz        = 832,
        batch        = 32,
        optimizer    = "AdamW",
        momentum     = 0.937,
        weight_decay = 0.0005,
        cos_lr       = True,
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        fliplr       = 0.5,
        mosaic       = 1.0,
        mixup        = 0.15,
        copy_paste   = 0.3,
        project      = PROJECT,
        save         = True,
        plots        = True,
        val          = True,
        workers      = 0,          # Windows local GPU
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1 — Backbone frozen, head only trains
    # AANKHEIN lock → DIMAGH seekhta hai 7 classes
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*55)
    print("  STAGE 1 — Backbone frozen (10 layers)")
    print("  Truck/bus/cyclist learning class boundaries")
    print("="*55 + "\n")

    model = YOLO("yolo11n.pt")

    model.train(
        **COMMON,
        epochs          = 8,           # short — just warm up the head
        freeze          = 10,          # freeze first 10 layers (backbone)
        lr0             = 0.001,       # higher LR okay — only head training
        lrf             = 0.1,
        warmup_epochs   = 1,
        warmup_momentum = 0.8,
        close_mosaic    = 3,
        save_period     = 5,
        name            = "bdd100k_v4_stage1",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 2 — Full fine-tune, low LR
    # Sab unlock → dheere dheere BDD100K ke scenes seekhna
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*55)
    print("  STAGE 2 — Full fine-tune (all layers unlocked)")
    print("  Low LR — car cannot dominate truck/bus anymore")
    print("="*55 + "\n")

    stage1_weights = f"runs/detect/{PROJECT}/bdd100k_v4_stage1/weights/best.pt"
    model = YOLO(stage1_weights)

    model.train(
        **COMMON,
        epochs          = 12,          # remaining epochs
        freeze          = 0,           # unfreeze everything
        lr0             = 0.0001,      # low LR — gentle fine-tune
        lrf             = 0.01,
        warmup_epochs   = 0,           # no warmup — already trained
        warmup_momentum = 0.8,
        close_mosaic    = 5,
        save_period     = 5,
        name            = "bdd100k_v4_stage2",
    )

    stage2_best = Path(results_s2.save_dir) / "weights" / "best.pt"
    shutil.copy(stage2_best, "insight_final.pt")

    print("\n" + "="*55)
    print("  Two-stage quick check complete!")
    print(f"  Best model: {PROJECT}/bdd100k_v4_stage2/weights/best.pt")
    print("="*55)