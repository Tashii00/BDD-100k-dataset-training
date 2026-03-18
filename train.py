if __name__ == "__main__":

    import torch
    import shutil
    from torch import nn
    from pathlib import Path
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.loss import v8DetectionLoss

    box_counts = torch.tensor([
         150_000,  # 0 car
         100_000,  # 1 traffic sign
         100_000,  # 2 traffic light
         100_000,  # 3 person
          24_601,  # 4 truck
          27_894,  # 5 bus
          60_840,  # 6 cyclist
    ], dtype=torch.float32)

    CLASS_WEIGHTS = box_counts.max() / box_counts

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

    # ── STAGE 1: frozen backbone ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  STAGE 1 — Frozen backbone (30 epochs)")
    print("="*55)

    model = YOLO("yolo11s.pt")

    results_s1 = model.train(
        trainer         = WeightedTrainer,
        data            = DATA_YAML,
        epochs          = 30,
        freeze          = 10,
        imgsz           = 1024,
        batch           = 32,
        cache           = 'disk',

        optimizer       = "AdamW",
        lr0             = 0.0005,      # lower than before — more stable warmup
        lrf             = 0.1,
        momentum        = 0.937,
        weight_decay    = 0.0005,
        cos_lr          = True,

        warmup_epochs   = 5,           # longer warmup — gentler start
        warmup_momentum = 0.8,

        box             = 7.5,
        cls             = 0.5,
        dfl             = 1.5,

        hsv_h           = 0.015,
        hsv_s           = 0.5,
        hsv_v           = 0.3,
        fliplr          = 0.5,
        mosaic          = 0.8,
        mixup           = 0.05,
        copy_paste      = 0.1,
        close_mosaic    = 5,

        patience        = 15,
        project         = PROJECT,
        name            = "insight3_stage1",
        save            = True,
        save_period     = 5,
        plots           = True,
        val             = True,
        workers         = 8,
        amp             = True,
    )

    stage1_best = Path(results_s1.save_dir) / "weights" / "best.pt"
    print(f"\n  Stage 1 complete → {stage1_best}")

    # ── STAGE 2: full fine-tune ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("  STAGE 2 — Full fine-tune (150 epochs)")
    print("="*55)

    model2 = YOLO(str(stage1_best))

    results_s2 = model2.train(
        trainer         = WeightedTrainer,
        data            = DATA_YAML,
        epochs          = 150,         # was 100
        freeze          = 0,
        imgsz           = 1024,
        batch           = 32,
        cache           = 'disk',

        optimizer       = "AdamW",
        lr0             = 0.0002,      # was 0.0005 — lower for better convergence
        lrf             = 0.01,
        momentum        = 0.937,
        weight_decay    = 0.0005,
        cos_lr          = True,

        warmup_epochs   = 3,
        warmup_momentum = 0.8,

        box             = 7.5,
        cls             = 0.5,
        dfl             = 1.5,

        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        fliplr          = 0.5,
        mosaic          = 1.0,
        mixup           = 0.15,
        copy_paste      = 0.3,
        close_mosaic    = 10,          # last 10 epochs clean

        patience        = 50,          # was 30 — more patience
        project         = PROJECT,
        name            = "insight3_stage2",
        save            = True,
        save_period     = 5,
        plots           = True,
        val             = True,
        workers         = 8,
        amp             = True,
    )

    stage2_best = Path(results_s2.save_dir) / "weights" / "best.pt"
    shutil.copy(stage2_best, "insight_final.pt")

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print(f"  Stage 1 : {stage1_best}")
    print(f"  Stage 2 : {stage2_best}")
    print("  Final   : insight_final.pt")
    print("="*55)