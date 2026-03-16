if __name__ == "__main__":

    from ultralytics import YOLO

    model = YOLO("runs/detect/runs/train/bdd100k_v3/weights/best.pt")

    metrics = model.val(
        data    = "data_final.yaml",
        imgsz   = 832,     # matches training imgsz
        batch   = 16,
        workers = 0,
        conf    = 0.15,
        verbose = True,
    )