import os
import cv2
import yaml
from pathlib import Path

def preprocess_and_clean(dataset_yaml_path):
    """
    Reads a YOLO dataset YAML file, iterates through the train/val/test directories,
    and removes any corrupt images or annotations that are out of bounds.
    """
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: Could not find {dataset_yaml_path}")
        return

    with open(dataset_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_path = Path(dataset_yaml_path).parent

    # Check 'train', 'val', 'test'
    splits = []
    if 'train' in data: splits.append(base_path / data['train'])
    if 'val' in data: splits.append(base_path / data['val'])
    if 'test' in data: splits.append(base_path / data['test'])

    corrupt_images = 0
    missing_labels = 0

    print("Starting dataset validation and cleaning...")

    for split_dir in splits:
        if not split_dir.exists():
            continue
            
        # Assuming typical YOLO structure: images and labels directories are adjacent, or images are in the fold
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        # Traverse recursively to find images
        for img_path in split_dir.rglob('*'):
            if img_path.suffix.lower() not in valid_extensions:
                continue

            # 1. Check if image is corrupt
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("Image could not be read")
            except Exception as e:
                print(f"Removing corrupt image: {img_path}")
                os.remove(img_path)
                corrupt_images += 1
                
                # Try to remove corresponding label
                label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                if label_path.exists():
                    os.remove(label_path)
                continue

            # 2. Check if label exists
            # YOLO labels usually mirror the images directory structure
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            
            # If label folder isn't parallel to images folder, try same folder
            if not label_path.exists():
                 label_path = img_path.parent / f"{img_path.stem}.txt"

            if not label_path.exists():
                print(f"Warning: Missing label for {img_path}. Removing image.")
                os.remove(img_path)
                missing_labels += 1
                continue
                
            # 3. Validating Bounding Box formats (0-1 normalized)
            with open(label_path, 'r') as lf:
                lines = lf.readlines()
                
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x, y, w, h = map(float, parts)
                    # Check if normalized coordinates are within [0, 1]
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        valid_lines.append(line)
                    else:
                        print(f"Removing invalid bounding box in {label_path}: {line.strip()}")
            
            if len(valid_lines) != len(lines):
                if len(valid_lines) == 0:
                     # Delete image and label if no valid boxes remain
                     os.remove(img_path)
                     os.remove(label_path)
                else:
                    # Write corrected boxes
                    with open(label_path, 'w') as lf:
                        lf.writelines(valid_lines)

    print("\n--- Cleaning Complete ---")
    print(f"Corrupt or unreadable images removed: {corrupt_images}")
    print(f"Images missing labels removed: {missing_labels}")
    print("Dataset is now clean and ready for training!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify and clean a YOLO dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset's data.yaml file.")
    args = parser.parse_args()
    
    preprocess_and_clean(args.data)
