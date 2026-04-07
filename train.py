# train.py
# Fine-tune YOLOv8 on the Parking-AMU50 Illegal Parking dataset from Roboflow.
# This trains a custom model specifically for detecting illegally parked vehicles,
# which performs better than the generic COCO-pretrained model.
#
# Usage:
#   1. Set your Roboflow API key below (or pass via command line)
#   2. Run: python train.py
#   3. The best model will be saved and automatically used by the detector

import os
import argparse

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

# ⚠️ PASTE YOUR ROBOFLOW API KEY HERE ⚠️
ROBOFLOW_API_KEY = "KigzvhM0KE8doTUTMi8x"

# Training hyperparameters
YOLO_MODEL       = "yolov8n.pt"    # Base model to fine-tune (nano for speed)
EPOCHS           = 50              # Number of training epochs
IMAGE_SIZE       = 640             # Input image size
BATCH_SIZE       = 16              # Batch size (reduce if GPU memory is limited)
PATIENCE         = 10              # Early stopping patience


def download_dataset(api_key):
    """
    Download the Parking-AMU50 Illegal Parking dataset from Roboflow.

    Args:
        api_key (str): Your Roboflow API key.

    Returns:
        str: Path to the dataset directory.
    """
    print("[Train] Downloading dataset from Roboflow...")
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("parking-amu50").project("illegal-parking")
    dataset = project.version(1).download("yolov8")

    print(f"[Train] Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_model(dataset_path, epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE):
    """
    Fine-tune YOLOv8 on the downloaded dataset.

    Args:
        dataset_path (str): Path to the downloaded dataset folder.
        epochs       (int): Number of training epochs.
        imgsz        (int): Input image size.
        batch        (int): Batch size.

    Returns:
        str: Path to the best trained model weights.
    """
    from ultralytics import YOLO

    # The data.yaml file is in the dataset root
    data_yaml = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"[Train] ERROR: data.yaml not found at {data_yaml}")
        print("[Train] Checking for alternative locations...")
        # Sometimes Roboflow nests it differently
        for root, dirs, files in os.walk(dataset_path):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                print(f"[Train] Found data.yaml at: {data_yaml}")
                break
        else:
            print("[Train] FATAL: Could not find data.yaml anywhere in the dataset.")
            return None

    print(f"[Train] Using dataset config: {data_yaml}")
    print(f"[Train] Base model: {YOLO_MODEL}")
    print(f"[Train] Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")
    print("=" * 60)

    # Load base model and fine-tune
    model = YOLO(YOLO_MODEL)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=PATIENCE,
        project="runs",
        name="illegal_parking",
        exist_ok=True,
        verbose=True,
    )

    # Path to best weights — Ultralytics may nest the save path differently
    possible_paths = [
        os.path.join("runs", "illegal_parking", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "illegal_parking", "weights", "best.pt"),
        os.path.join("runs", "detect", "illegal_parking", "weights", "best.pt"),
    ]

    best_weights = None
    for p in possible_paths:
        if os.path.exists(p):
            best_weights = p
            break

    # Fallback: search recursively
    if best_weights is None:
        for root, dirs, files in os.walk("runs"):
            if "best.pt" in files:
                best_weights = os.path.join(root, "best.pt")
                break

    if best_weights:
        # Copy the best model to the project root for easy access
        import shutil
        dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_parking.pt")
        shutil.copy2(best_weights, dest)
        print("=" * 60)
        print(f"[Train] ✅ Training complete!")
        print(f"[Train] Best model found at: {best_weights}")
        print(f"[Train] Copied to: {dest}")
        print(f"[Train] The detector will automatically use this model.")
        return dest
    else:
        print("[Train] WARNING: best.pt not found anywhere. Check training logs above.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Illegal Parking Dataset")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (overrides the one in the script)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE,
                        help=f"Image size (default: {IMAGE_SIZE})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use if already downloaded)")
    args = parser.parse_args()

    api_key = args.api_key or ROBOFLOW_API_KEY

    if api_key == "YOUR_API_KEY":
        print("=" * 60)
        print("[Train] ERROR: Please set your Roboflow API key!")
        print("[Train] Either:")
        print("  1. Edit train.py and set ROBOFLOW_API_KEY")
        print("  2. Run: python train.py --api-key YOUR_KEY")
        print("")
        print("[Train] Get your key at: https://app.roboflow.com/settings/api")
        print("=" * 60)
        return

    # Step 1: Download dataset
    if args.skip_download:
        # Look for existing dataset
        dataset_path = os.path.join(os.path.dirname(__file__), "Illegal-Parking-1")
        if not os.path.exists(dataset_path):
            print(f"[Train] Dataset not found at {dataset_path}. Downloading...")
            dataset_path = download_dataset(api_key)
    else:
        dataset_path = download_dataset(api_key)

    # Step 2: Train
    best_model = train_model(dataset_path, args.epochs, args.imgsz, args.batch)

    if best_model:
        print("")
        print("🎉 All done! Restart the dashboard (python main.py) to use")
        print("   the custom-trained model for better parking detection.")


if __name__ == "__main__":
    main()
