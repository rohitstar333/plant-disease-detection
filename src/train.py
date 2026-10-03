"""
=============================================================================
           EFFICIENTNETV2-B0 TRANSFER LEARNING TRAINING MODULE
=============================================================================
This module executes a two-phase training workflow:
  Phase 1: Warmup training of custom classification head with frozen backbone.
  Phase 2: Fine-tuning top layers of EfficientNetV2-B0 base with reduced lr.
=============================================================================
"""

import json
import random
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.model import build_efficientnet_model, build_field_data_augmentation, unfreeze_for_finetuning
from src.prepare_dataset import prepare_dataset_pipeline, SEED, CLASS_MAPPING

# Constants & Directory Setup
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_FILE = PROJECT_ROOT / "dataset" / "splits.json"

MODEL_SAVE_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_NAMES_SAVE_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
WARMUP_EPOCHS = 8
FINE_TUNE_EPOCHS = 17


def set_reproducibility_seeds(seed: int = SEED):
    """Enforces random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset_from_splits(splits_data: dict, batch_size: int = BATCH_SIZE):
    """
    Creates tf.data.Dataset generators for train, validation, and test splits.
    Ensures images are read, converted to RGB, and resized to 224x224.
    Field data augmentation is applied strictly to training batches.
    """
    folder_to_index = {folder: i for i, folder in enumerate(splits_data["class_order"])}
    num_classes = len(splits_data["class_order"])
    aug = build_field_data_augmentation()

    def create_generator(split_items: list, is_training: bool):
        def generator():
            for item in split_items:
                img_path = PROJECT_ROOT / item["path"]
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(IMG_SIZE)
                        img_arr = np.array(img, dtype=np.float32)
                    
                    label_idx = folder_to_index[item["folder"]]
                    label_onehot = tf.keras.utils.to_categorical(label_idx, num_classes=num_classes)
                    yield img_arr, label_onehot
                except Exception as err:
                    print(f"[ERROR] Loading image {img_path}: {err}")

        output_signature = (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(split_items), seed=SEED)
            dataset = dataset.batch(batch_size)
            # Apply training-only data augmentation
            dataset = dataset.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.batch(batch_size)
        
        return dataset.prefetch(tf.data.AUTOTUNE)

    train_ds = create_generator(splits_data["train"], is_training=True)
    val_ds = create_generator(splits_data["val"], is_training=False)
    test_ds = create_generator(splits_data["test"], is_training=False)

    return train_ds, val_ds, test_ds, num_classes


def compute_dataset_class_weights(splits_data: dict) -> dict:
    """Calculates class weights to handle class sample imbalance during training."""
    folder_to_index = {folder: i for i, folder in enumerate(splits_data["class_order"])}
    train_labels = [folder_to_index[item["folder"]] for item in splits_data["train"]]
    
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    
    return class_weight_dict


def plot_and_save_combined_history(history_phase1, history_phase2, output_dir: Path = RESULTS_DIR):
    """Plots combined training vs validation accuracy and loss curves across both phases."""
    output_dir.mkdir(exist_ok=True)
    
    acc = history_phase1.history.get("accuracy", []) + history_phase2.history.get("accuracy", [])
    val_acc = history_phase1.history.get("val_accuracy", []) + history_phase2.history.get("val_accuracy", [])
    loss = history_phase1.history.get("loss", []) + history_phase2.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", []) + history_phase2.history.get("val_loss", [])
    
    epochs_range = range(1, len(acc) + 1)
    phase1_length = len(history_phase1.history.get("accuracy", []))

    # 1. Plot Accuracy Graph
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, acc, 'o-', color='#2b8cbe', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, 's-', color='#e34a33', label='Validation Accuracy', linewidth=2)
    plt.axvline(x=phase1_length + 0.5, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('EfficientNetV2-B0 Transfer Learning - Accuracy', fontsize=13, pad=12)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    accuracy_path = output_dir / "training_accuracy.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved combined accuracy curve plot: {accuracy_path}")

    # 2. Plot Loss Graph
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, loss, 'o-', color='#2b8cbe', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, 's-', color='#e34a33', label='Validation Loss', linewidth=2)
    plt.axvline(x=phase1_length + 0.5, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('EfficientNetV2-B0 Transfer Learning - Loss', fontsize=13, pad=12)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Categorical Crossentropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    loss_path = output_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved combined loss curve plot: {loss_path}")


def train_pipeline(warmup_epochs: int = WARMUP_EPOCHS, finetune_epochs: int = FINE_TUNE_EPOCHS):
    """Executes two-phase EfficientNetV2-B0 transfer learning training workflow."""
    print("=========================================================")
    print("    EFFICIENTNETV2-B0 TRANSFER LEARNING TRAINING PIPELINE ")
    print("=========================================================")
    
    set_reproducibility_seeds(SEED)

    # Load splits or prepare dataset
    if not SPLITS_FILE.exists():
        print("Splits file not found. Running dataset preparation...")
        splits_data = prepare_dataset_pipeline()
    else:
        with open(SPLITS_FILE, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save exact class order and display mapping
    ordered_display_names = [CLASS_MAPPING[folder] for folder in splits_data["class_order"]]
    class_config = {
        "class_order_folders": splits_data["class_order"],
        "class_display_names": ordered_display_names,
        "mapping": CLASS_MAPPING,
        "architecture": "EfficientNetV2-B0",
        "input_shape": [224, 224, 3]
    }
    with open(CLASS_NAMES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(class_config, f, indent=4)
    print(f"  [OK] Saved model class ordering to: {CLASS_NAMES_SAVE_PATH}")

    # Build Data Pipelines
    train_ds, val_ds, _, num_classes = load_dataset_from_splits(splits_data)
    class_weights = compute_dataset_class_weights(splits_data)
    print(f"  [OK] Computed class weights: {class_weights}")

    # Build EfficientNetV2-B0 Model
    model = build_efficientnet_model(input_shape=(224, 224, 3), num_classes=num_classes)
    model.summary()

    # Callbacks for Phase 1
    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n[PHASE 1] Training classification head ({warmup_epochs} epochs, lr=0.001)...\n")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase1
    )

    # Phase 2: Fine-tuning EfficientNetV2-B0 base
    print(f"\n[PHASE 2] Unfreezing EfficientNetV2 top layers for fine-tuning ({finetune_epochs} epochs, lr=1e-4)...\n")
    model = unfreeze_for_finetuning(model, unfreeze_layers=50, learning_rate=1e-4)

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    initial_epoch = len(history1.history.get("accuracy", []))
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + finetune_epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks_phase2
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] EfficientNetV2-B0 training complete. Model saved to: {MODEL_SAVE_PATH}")

    # Generate Combined Graphs
    plot_and_save_combined_history(history1, history2)


if __name__ == "__main__":
    train_pipeline()