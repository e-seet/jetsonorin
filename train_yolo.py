import os
import argparse
from ultralytics import YOLO

def train_model(
    model_name,
    dataset_yaml,
    epochs=80,
    batch_size=-1,
    imgsz=512,
    workers=4,
    cache=False,
    device="auto",
    patience=50,
):
    """
    Trains a YOLO model locally.
    Default parameters are set low to prevent Out-Of-Memory (OOM) errors on the Jetson Orin Nano (8GB shared memory).
    """
    print(f"--- Starting local training for {model_name} ---")
    
    # Map friendly names to actual ultralytics base models
    model_map = {
        "yolov8": "yolov8n.pt",
        "yolov9": "yolov9t.pt", # Switched from 9c to 9t (Tiny) for much lower memory usage
        "yolo11": "yolo11n.pt",
        "yolo12": "yolo12n.pt"
    }
    
    base_model = model_map.get(model_name)
    if not base_model:
        print(f"Error: Unknown model {model_name}. Supported models: {list(model_map.keys())}")
        return

    # Initialize model
    model = YOLO(base_model)
    
    # Train the model
    # On Jetson, keep workers low; on Colab L4 you can safely increase.
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        # cache='disk',      # Do not cache images to RAM
        overlap_mask=False, # Save memory during mask processing (mostly for segmentation, but good for stability)
        mosaic=0.5,
        mixup=0.15,
        close_mosaic=10,   # Disable mosaic at the end to save a heavy augmentation step
        amp=False,        # Disable automatic mixed precision
        plots=False,      # 🚀 prevents matplotlib loading
        project="fire_detection_runs_v2",
        name=f"{model_name}_fire",
        exist_ok=True,
        workers=workers,
        cache=cache,
        device=device,
        patience=patience,
    )
    
    print(f"--- Training complete for {model_name} ---")
    print(f"Best weights saved to: fire_detection_runs_v2/{model_name}_fire/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO models locally on Jetson for Fire Detection.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to train: yolov8, yolov9, yolo11, yolo12")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset data.yaml file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (keep low <= 2 for Jetson 8GB)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--cache", type=bool, default=False, help="cache dataset in RAM")
    parser.add_argument("--device", type=str, default="auto", help="Device to use, e.g. '0', 'cpu', 'auto'")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset yaml file '{args.data}' not found!")
        exit(1)
        
    for m in args.models:
        train_model(
            m,
            args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            cache=args.cache,
            device=args.device,
            patience=args.patience,
        )
