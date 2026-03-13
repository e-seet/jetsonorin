from ultralytics import YOLO
import argparse
import os

def export_to_tensorrt(model_path):
    """
    Exports a PyTorch YOLO model to TensorRT format optimized for Jetson.
    Crucially, it uses fp16=True for maximum speed.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading '{model_path}'...")
    model = YOLO(model_path)
    
    # Exporting for Jetson Orin with TensorRT and FP16 half-precision
    # workspace=4 maxes out memory allocation for the conversion process
    print(f"Starting TensorRT export for {model_path} (This may take 10-20 minutes on Jetson)...")
    try:
        exported_path = model.export(
            format='engine', 
            device='0',      # target GPU 0
            half=True,       # fp16 optimization - Critical for Jetson 30 FPS
            workspace=4,     # max workspace limits in GB
            simplify=True    # simplify the onnx graph before TensorRT conversion
        )
        print(f"Successfully exported TensorRT engine to: {exported_path}")
    except Exception as e:
        print(f"Failed to export {model_path}:\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch models to TensorRT.")
    parser.add_argument("--models", nargs="+", required=True, help="List of .pt files to compile (e.g. yolov9c_fire.pt yolo11n_fire.pt)")
    args = parser.parse_args()

    for m in args.models:
        export_to_tensorrt(m)
