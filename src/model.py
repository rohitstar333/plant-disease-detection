"""
=============================================================================
           EFFICIENTNETV2-B0 TRANSFER LEARNING MODEL DEFINITION
=============================================================================
This module defines an EfficientNetV2-B0 transfer-learning model architecture
with Squeeze-and-Excitation attention mechanisms and unfreezing fine-tuning controls.

Model:
    EfficientNetV2-B0 pretrained on ImageNet

Input:
    224 x 224 RGB image

Output:
    5-class softmax classification

Classes:
    1. Tomato Healthy
    2. Tomato Early Blight
    3. Tomato Late Blight
    4. Potato Early Blight
    5. Potato Late Blight
=============================================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# =============================================================================
# TRAINING-ONLY DATA AUGMENTATION
# =============================================================================

def build_field_data_augmentation() -> tf.keras.Sequential:
    """
    Build a realistic training-time augmentation pipeline.
    Active ONLY during model training to simulate field photography variations:
        - horizontal flip
        - small rotation (±10%)
        - zoom (±15%)
        - translation (±10%)
        - brightness (±15%)
        - contrast (±15%)
    """
    return models.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_random_flip"),
            layers.RandomRotation(0.10, name="aug_random_rotation"),
            layers.RandomZoom(0.15, name="aug_random_zoom"),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10, name="aug_random_translation"),
            layers.RandomBrightness(0.15, name="aug_random_brightness"),
            layers.RandomContrast(0.15, name="aug_random_contrast"),
        ],
        name="field_data_augmentation"
    )


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def build_efficientnet_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 5,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Build and compile an EfficientNetV2-B0 transfer-learning model.
    EfficientNetV2 features Fused-MBConv layers and Squeeze-and-Excitation (SE)
    attention modules optimized for fine-grained feature differentiation.
    """
    inputs = layers.Input(shape=input_shape, name="leaf_image_input")

    # EfficientNetV2 preprocessing (scales [0, 255] float32 inputs to [-1, 1])
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

    # ImageNet-pretrained EfficientNetV2-B0 backbone
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="swish", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantDiseaseEfficientNetV2")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =============================================================================
# FINE-TUNING
# =============================================================================

def unfreeze_for_finetuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 50,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Enable controlled fine-tuning of the top layers of EfficientNetV2-B0.
    BatchNormalization layers remain frozen to preserve statistics.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "efficientnet" in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 50:
                base_model = layer
                break

    if base_model is None:
        raise ValueError("EfficientNetV2 base model layer not found in architecture.")

    base_model.trainable = True
    num_total_layers = len(base_model.layers)
    fine_tune_at = max(0, num_total_layers - unfreeze_layers)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = 0
    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"  [OK] Fine-tuning enabled: Unfroze top {unfreeze_layers} layers of EfficientNetV2 (lr={learning_rate})")
    return model


if __name__ == "__main__":
    effnet_model = build_efficientnet_model()
    effnet_model.summary()