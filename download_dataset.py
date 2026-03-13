import os
import shutil
import kagglehub

def load_env():
    """Loads environment variables from .env file"""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"Loaded {key} from .env")
    else:
        print("Warning: .env file not found. Ensure KAGGLE_API_TOKEN is set.")

def download_kaggle_dataset():
    """
    Downloads the Fire/Smoke detection dataset from Kaggle using kagglehub.
    It authenticates using the KAGGLE_API_TOKEN loaded from .env.
    """
    load_env()
    
    print("Downloading SayedGamal99 Smoke Fire Detection dataset from Kaggle...")
    
    try:
        # kagglehub uses KAGGLE_API_TOKEN automatically if it's in the environment
        dataset_path = kagglehub.dataset_download('sayedgamal99/smoke-fire-detection-yolo')
        print(f"Dataset successfully downloaded to: {dataset_path}")
        
        # Move it to a local folder for easier use
        local_dir = os.path.join(os.getcwd(), "datasets", "fire_smoke")
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
            
        print(f"Copying dataset from {dataset_path} to {local_dir}...")
        shutil.copytree(dataset_path, local_dir)
        print(f"Dataset is ready at {local_dir}.")
        print("You can now run preprocess_data.py to clean it, and train_yolo.py to start training.")
        
    except Exception as e:
        print(f"Failed to download dataset via kagglehub: {e}")

if __name__ == "__main__":
    download_kaggle_dataset()
