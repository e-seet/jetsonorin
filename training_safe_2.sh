#!/bin/bash
# start_training.sh

pip3 install polars seaborn matplotlib pandas requests tqdm pyyaml psutil --no-deps 2>/dev/null

echo "======================================"
echo "Starting Sequential Training on Jetson"
echo "======================================"

# Reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

run_training () {
    MODEL=$1
    echo "Starting $MODEL..."

    python3 train_yolo.py --models $MODEL --data datasets/fire_smoke/data.yaml --epochs 80 --batch 2
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "$MODEL training completed successfully."
    else
        echo "$MODEL crashed (exit code $STATUS). Continuing..."
    fi

    echo "Cleaning CUDA memory..."
    python3 - <<EOF
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
EOF

    echo "Resetting Jetson GPU clocks..."
    jetson_clocks 2>/dev/null

    sleep 60
    echo "--------------------------------------"
}
sleep 30
# run_training yolov8
run_training yolov9
# run_training yolo12

echo "All training attempts finished."