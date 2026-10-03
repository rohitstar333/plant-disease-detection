"""
=============================================================================
                    MODEL EVALUATION & METRICS MODULE
=============================================================================
This module evaluates the trained EfficientNetV2-B0 model on the unseen test set,
generating classification reports, confusion matrices, and metrics JSON exports.
=============================================================================
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_FILE = PROJECT_ROOT / "dataset" / "splits.json"

MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
REPORT_TXT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

IMG_SIZE = (224, 224)


def evaluate_model_pipeline():
    """Runs model evaluation exclusively on the test set."""
    print("=========================================================")
    print("      PLANT DISEASE MODEL EVALUATION (EFFICIENTNETV2-B0) ")
    print("=========================================================")

    # 1. Verify model and artifacts exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            f"Please run 'python src/train.py' before evaluating."
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class configuration not found at {CLASS_NAMES_PATH}.")
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Dataset splits file not found at {SPLITS_FILE}.")

    # 2. Load model and class configuration
    print(f"  [OK] Loading trained Keras model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_config = json.load(f)

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    folder_order = class_config["class_order_folders"]
    display_names = class_config["class_display_names"]
    folder_to_idx = {folder: i for i, folder in enumerate(folder_order)}

    test_items = splits_data["test"]
    print(f"  [OK] Found {len(test_items)} test samples in splits.json")

    # 3. Load test set images and ground truth labels
    test_images = []
    y_true = []

    for item in test_items:
        img_path = PROJECT_ROOT / item["path"]
        true_label = folder_to_idx[item["folder"]]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(IMG_SIZE)
                img_arr = np.array(img, dtype=np.float32)
                test_images.append(img_arr)
                y_true.append(true_label)
        except Exception as err:
            print(f"  [WARNING] Error reading test image {img_path}: {err}")

    if not y_true:
        raise ValueError("No valid test images could be evaluated!")

    test_images = np.array(test_images, dtype=np.float32)

    print("  [INFO] Computing model predictions on test images in batches...")
    probs = model.predict(test_images, batch_size=32, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4. Compute Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics_dict = {
        "test_accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4)
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n  [OK] Saved metrics summary to: {METRICS_JSON_PATH}")

    # 5. Generate Classification Report
    clr_report = classification_report(
        y_true,
        y_pred,
        target_names=display_names,
        digits=4,
        zero_division=0
    )

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("PLANT DISEASE DETECTION (EFFICIENTNETV2-B0) - TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(clr_report)
    print(f"  [OK] Saved classification report to: {REPORT_TXT_PATH}")

    # 6. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True
    )
    plt.title("Confusion Matrix - EfficientNetV2-B0 (PlantVillage Test Set)", fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"  [OK] Saved confusion matrix heatmap to: {CONFUSION_MATRIX_PATH}")

    # 7. Print Terminal Summary
    print("\n---------------------------------------------------------")
    print("EVALUATION RESULTS SUMMARY:")
    print("---------------------------------------------------------")
    print(f" Test Accuracy   : {acc * 100:.2f}%")
    print(f" Macro Precision : {macro_prec * 100:.2f}%")
    print(f" Macro Recall    : {macro_rec * 100:.2f}%")
    print(f" Macro F1-Score  : {macro_f1 * 100:.2f}%")
    print(f" Weighted F1     : {weighted_f1 * 100:.2f}%")
    print("---------------------------------------------------------")
    print("\nClassification Report:\n")
    print(clr_report)

    return metrics_dict


if __name__ == "__main__":
    evaluate_model_pipeline()
