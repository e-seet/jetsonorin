#!/bin/bash
# start_training.sh
# Runs YOLO training sequentially for v9, 11, and 12 to avoid overwhelming Jetson memory.

# Ensure dependencies are installed
pip3 install polars seaborn matplotlib pandas requests tqdm pyyaml psutil --no-deps 2>/dev/null

echo "======================================"
echo "Starting Sequential Training on Jetson"
echo "======================================"

echo "Starting YOLOv8..."
python3 train_yolo.py --models yolov8 --data datasets/fire_smoke/data.yaml --epochs 10 --batch 1
echo "YOLOv8 training complete. Resting 30s..."
sleep 30
echo "--------------------------------------"

echo "Starting YOLOv9..."
python3 train_yolo.py --models yolov9 --data datasets/fire_smoke/data.yaml --epochs 10 --batch 1
echo "YOLOv9 training complete. Resting 30s..."
sleep 60
echo "--------------------------------------"

# echo "Starting YOLO11..."
# python3 train_yolo.py --models yolo11 --data datasets/fire_smoke/data.yaml --epochs 10 --batch 2
# echo "YOLO11 training complete. Resting 30s..."
# sleep 30
# echo "--------------------------------------"

echo "Starting YOLO12..."
python3 train_yolo.py --models yolo12 --data datasets/fire_smoke/data.yaml --epochs 10 --batch 2
echo "YOLO12 training complete."
echo "--------------------------------------"

echo "All training finished successfully!"
