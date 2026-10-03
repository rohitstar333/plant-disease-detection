"""
=============================================================================
                    DATASET PREPARATION & VALIDATION MODULE
=============================================================================
This module verifies dataset structure, checks image integrity, computes
class balance, and constructs deterministic train/validation/test splits.
=============================================================================
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Fixed Random Seed for Reproducibility
SEED = 42

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_FILE = DATASET_DIR / "splits.json"

# Explicit mapping from directory folder names to clean display names
CLASS_MAPPING = {
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight"
}

EXPECTED_FOLDERS = list(CLASS_MAPPING.keys())
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def verify_dataset_structure(dataset_path: Path = DATASET_DIR) -> Dict[str, Path]:
    """Verifies that the dataset directory exists and contains all 5 required class subfolders."""
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found at: {dataset_path}\n"
            f"Please place the 5 PlantVillage class subfolders inside 'dataset/'\n"
            f"or run 'python scripts/create_sample_dataset.py' for pipeline testing."
        )

    found_classes = {}
    missing_folders = []

    for folder_name in EXPECTED_FOLDERS:
        folder_path = dataset_path / folder_name
        if folder_path.exists() and folder_path.is_dir():
            found_classes[folder_name] = folder_path
        else:
            missing_folders.append(folder_name)

    if missing_folders:
        raise FileNotFoundError(
            f"Dataset incomplete! Missing {len(missing_folders)} class folder(s):\n"
            f" - " + "\n - ".join(missing_folders) + "\n\n"
            f"Expected 5 folders: {EXPECTED_FOLDERS}"
        )

    return found_classes


def scan_and_clean_images(class_folders: Dict[str, Path]) -> Dict[str, List[Path]]:
    """Scans all class folders, identifies valid images, and filters out corrupted/unreadable files."""
    valid_images_by_class: Dict[str, List[Path]] = {}
    total_corrupted = 0
    total_scanned = 0

    print("[INFO] Scanning images for corruption and validity...")

    for folder_name, folder_path in class_folders.items():
        display_name = CLASS_MAPPING[folder_name]
        valid_paths = []

        all_files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

        for img_path in all_files:
            total_scanned += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.load()
                valid_paths.append(img_path)
            except Exception as err:
                print(f"  [WARNING] Corrupted image detected and skipped: {img_path.name} ({err})")
                total_corrupted += 1

        valid_images_by_class[folder_name] = valid_paths
        print(f"  [OK] {display_name} ({folder_name}): {len(valid_paths)} valid images")

    print(f"\nScan Summary: {total_scanned} scanned, {total_corrupted} corrupted/skipped.")
    return valid_images_by_class


def create_dataset_splits(
    valid_images: Dict[str, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
    output_file: Path = SPLITS_FILE
) -> Dict:
    """Creates stratified train, validation, and test splits with fixed random seed."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    random.seed(seed)

    splits = {
        "seed": seed,
        "class_mapping": CLASS_MAPPING,
        "class_order": EXPECTED_FOLDERS,
        "train": [],
        "val": [],
        "test": [],
        "summary": {}
    }

    for folder_name in EXPECTED_FOLDERS:
        paths = list(valid_images[folder_name])
        random.shuffle(paths)

        n_total = len(paths)
        if n_total < 3:
            raise ValueError(f"Class '{folder_name}' has only {n_total} image(s). Need at least 3 for splits.")

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        for p in train_paths:
            splits["train"].append({"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"), "folder": folder_name})
        for p in val_paths:
            splits["val"].append({"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"), "folder": folder_name})
        for p in test_paths:
            splits["test"].append({"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"), "folder": folder_name})

        display_name = CLASS_MAPPING[folder_name]
        splits["summary"][display_name] = {
            "total": n_total,
            "train": n_train,
            "val": n_val,
            "test": n_test
        }

    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    print(f"\n[OK] Dataset splits saved successfully to: {output_file}")
    return splits


def prepare_dataset_pipeline() -> Dict:
    """Main pipeline execution for dataset verification and splitting."""
    print("=========================================================")
    print("           PLANT DISEASE DATASET PREPARATION            ")
    print("=========================================================")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Random Seed : {SEED}")
    print("Ratios      : 70% Train / 15% Validation / 15% Test\n")

    class_folders = verify_dataset_structure(DATASET_DIR)
    valid_images = scan_and_clean_images(class_folders)
    splits = create_dataset_splits(valid_images)

    print("\n---------------------------------------------------------")
    print("CLASS DISTRIBUTION SUMMARY:")
    print("---------------------------------------------------------")
    print(f"{'Class Name':<25} | {'Total':<7} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-" * 62)
    for cls_display, counts in splits["summary"].items():
        print(f"{cls_display:<25} | {counts['total']:<7} | {counts['train']:<7} | {counts['val']:<7} | {counts['test']:<7}")
    print("-" * 62)

    return splits


if __name__ == "__main__":
    prepare_dataset_pipeline()
