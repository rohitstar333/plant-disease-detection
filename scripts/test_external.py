"""
=============================================================================
             EXTERNAL REAL-WORLD IMAGES EVALUATION SCRIPT
=============================================================================
This script evaluates the trained model on external real-world leaf images
from outside the PlantVillage training dataset to benchmark real-world
generalization performance and report distribution-shift limitations transparently.
=============================================================================
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predict import load_trained_model, predict_image


def get_actual_label_from_path(file_path: Path, class_mapping: dict) -> str:
    """Infers ground truth label from parent folder or filename pattern."""
    parent_name = file_path.parent.name
    if parent_name in class_mapping:
        return class_mapping[parent_name]
    
    filename_lower = file_path.name.lower()
    if "potato" in filename_lower and "late" in filename_lower:
        return "Potato Late Blight"
    elif "potato" in filename_lower and "early" in filename_lower:
        return "Potato Early Blight"
    elif "tomato" in filename_lower and "healthy" in filename_lower:
        return "Tomato Healthy"
    elif "tomato" in filename_lower and "late" in filename_lower:
        return "Tomato Late Blight"
    elif "tomato" in filename_lower and "early" in filename_lower:
        return "Tomato Early Blight"
    
    return "Unknown"


def test_external_images(external_dir: Path):
    """Evaluates images in target external directory."""
    print("=========================================================")
    print("        EXTERNAL REAL-WORLD GENERALIZATION TEST          ")
    print("=========================================================")
    print(f"Target Directory: {external_dir}\n")

    if not external_dir.exists():
        print(f"[INFO] External image directory does not exist at {external_dir}.")
        return []

    model, class_names = load_trained_model()
    
    # Load class mapping
    class_names_path = PROJECT_ROOT / "models" / "class_names.json"
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_config = json.load(f)
    class_mapping = class_config.get("mapping", {})

    image_files = [p for p in external_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    if not image_files:
        print("[INFO] No external image files found in target directory.")
        return []

    print(f"Found {len(image_files)} external real-world image(s):\n")
    print(f"| {'Image':<30} | {'Actual':<20} | {'Prediction':<20} | {'Confidence':<10} | {'Correct':<7} |")
    print(f"| {'-'*30} | {'-'*20} | {'-'*20} | {'-'*10} | {'-'*7} |")

    results_table = []
    correct_count = 0
    total_evaluable = 0

    for img_path in image_files:
        res = predict_image(model, class_names, img_path)
        actual = get_actual_label_from_path(img_path, class_mapping)
        pred = res["class_name"]
        conf_str = f"{res['confidence']*100:.2f}%"
        
        if actual != "Unknown":
            is_correct = (actual == pred)
            correct_str = "YES" if is_correct else "NO"
            if is_correct:
                correct_count += 1
            total_evaluable += 1
        else:
            correct_str = "N/A"

        print(f"| {img_path.name[:30]:<30} | {actual:<20} | {pred:<20} | {conf_str:<10} | {correct_str:<7} |")
        results_table.append({
            "image": img_path.name,
            "actual": actual,
            "prediction": pred,
            "confidence": conf_str,
            "correct": correct_str
        })

    if total_evaluable > 0:
        print("\n---------------------------------------------------------")
        print(f"External Validation Accuracy: {correct_count}/{total_evaluable} ({correct_count/total_evaluable*100:.2f}%)")
        print("---------------------------------------------------------")

    return results_table


if __name__ == "__main__":
    ext_dir = PROJECT_ROOT / "external_validation_images"
    if not ext_dir.exists():
        ext_dir = PROJECT_ROOT / "external_test_images"
    if len(sys.argv) > 1:
        ext_dir = Path(sys.argv[1])
    test_external_images(ext_dir)
