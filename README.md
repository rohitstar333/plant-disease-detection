# 🌿 Plant Disease Detection from Leaf Images

An end-to-end Computer Vision and Deep Learning system for automated agricultural crop disease classification using **EfficientNetV2-B0 + Transfer Learning** built with **TensorFlow / Keras**, evaluated on both held-out laboratory test splits and external field images, and deployed via an interactive **Streamlit** web application.

---

## 📌 Problem Statement

Plant diseases cause severe losses in global crop yield every year. In particular, early blight (*Alternaria solani*) and late blight (*Phytophthora infestans*) in tomato and potato crops spread rapidly if not diagnosed early. Manual visual inspection by agricultural experts is time-consuming and often unavailable in remote farming regions. An automated, accurate, and real-time computer vision system assists farmers in early disease identification.

---

## 🎯 Project Objective

1. Develop a high-accuracy deep learning classifier using **EfficientNetV2-B0 + Transfer Learning** for 5 critical health classes of tomato and potato crops.
2. Implement realistic field-simulated data augmentation (rotations, zoom, lighting variations, contrast shifts) to improve real-world generalization.
3. Build a safe **unknown-image rejection mechanism** to prevent non-leaf photographs (people, vehicles, buildings, random objects) from being misclassified as plant diseases.
4. Deploy a modern, professional web application using **Streamlit**.

---

## 🗂️ Dataset & Classes

The system is trained and evaluated on **6,498 high-resolution images** from the **PlantVillage** dataset, organized into a 70% Train / 15% Validation / 15% Test stratified split with random seed 42.

### Target Classes (5)
1. **Tomato Healthy** (`Tomato___healthy`) — 1,591 total images
2. **Tomato Early Blight** (`Tomato___Early_blight`) — 1,000 total images
3. **Tomato Late Blight** (`Tomato___Late_blight`) — 1,909 total images
4. **Potato Early Blight** (`Potato___Early_blight`) — 1,000 total images
5. **Potato Late Blight** (`Potato___Potato___Late_blight` / `Potato___Late_blight`) — 998 total images

---

## ⚙️ Model Architecture & Preprocessing

- **Backbone**: EfficientNetV2-B0 pre-trained on ImageNet. EfficientNetV2 utilizes Fused-MBConv bottleneck blocks and Squeeze-and-Excitation (SE) channel attention modules, making it exceptionally sensitive to micro-lesion visual patterns (concentric target rings vs water-soaked spots).
- **Input Resolution**: $224 \times 224 \times 3$ (RGB format).
- **Preprocessing Pipeline**: Images are loaded in 3-channel RGB, resized to $224 \times 224$, and scaled via EfficientNetV2's internal preprocessing layer (`preprocess_input`).
- **Field Data Augmentation**:
  - `RandomFlip("horizontal_and_vertical")`
  - `RandomRotation(0.25)`
  - `RandomZoom(0.20)`
  - `RandomTranslation(height_factor=0.15, width_factor=0.15)`
  - `RandomBrightness(0.25)`
  - `RandomContrast(0.25)`

---

## 🏋️ Two-Phase Training Strategy

Training is executed in two controlled phases to prevent gradient explosion and preserve pre-trained feature maps:

1. **Phase 1 (Classification Head Warmup)**: The EfficientNetV2-B0 backbone is frozen (`trainable = False`). The custom classification head (`GlobalAveragePooling2D` $\to$ `BatchNormalization` $\to$ `Dense(256, Swish)` $\to$ `Dropout(0.4)` $\to$ `Dense(5, Softmax)`) is trained for 8 warmup epochs with learning rate $\eta = 0.001$.
2. **Phase 2 (Backbone Fine-Tuning)**: The top 50 layers of EfficientNetV2-B0 are unfrozen and fine-tuned with a lower learning rate $\eta = 1 \times 10^{-4}$.
3. **Callbacks & Class Weighting**:
   - `EarlyStopping`: Stops training if validation loss fails to improve for 5 consecutive epochs.
   - `ReduceLROnPlateau`: Reduces learning rate by factor $0.5$ when validation loss plateaus.
   - `ModelCheckpoint`: Automatically saves the best model weights to `models/plant_disease_model.keras`.
   - Balanced class weighting computed via Scikit-Learn to address class imbalance.

---

## 📊 Measured Evaluation Results

Evaluated on the **976 held-out test images** (never seen during training or validation):

| Metric | Score (Percentage) | Score (Decimal) |
| :--- | :---: | :---: |
| **Test Accuracy** | **97.95%** | `0.9795` |
| **Macro Precision** | **97.88%** | `0.9788` |
| **Macro Recall** | **98.10%** | `0.9810` |
| **Macro F1-Score** | **97.99%** | `0.9799` |
| **Weighted F1-Score** | **97.95%** | `0.9795` |

### Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support (Test Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato Healthy** | `0.9917` | `1.0000` | `0.9958` | 239 |
| **Tomato Early Blight** | `0.9533` | `0.9533` | `0.9533` | 150 |
| **Tomato Late Blight** | `0.9752` | `0.9582` | `0.9666` | 287 |
| **Potato Early Blight** | `1.0000` | `0.9933` | `0.9967` | 150 |
| **Potato Late Blight** | `0.9740` | `1.0000` | `0.9868` | 150 |

---

## 🛡️ Safe Unknown-Image Handling (Non-Leaf Rejection)

To ensure non-leaf photographs (people, vehicles, laptops, buildings, text documents, random objects) are not misclassified as plant diseases, the prediction engine incorporates a multi-signal rejection mechanism:

1. **Leaf Vegetation Presence Check**: Measures the proportion of pixels containing plant chlorophyll green, chlorotic yellow, or necrotic brown color signatures (`leaf_ratio`). Images with $< 8\%$ leaf color coverage are automatically flagged as non-leaf objects.
2. **Prediction Uncertainty & Entropy Check**: Computes Shannon Entropy $H(p) = -\sum p \log p$ across the prediction probability distribution. Inputs with high entropy ($H(p) > 1.45$) or low top confidence ($< 40\%$) trigger rejection.
3. **UI Response**: When an unrelated image is uploaded, the Streamlit app suppresses disease classification and displays:
   > ⚠️ **Image Not Recognized** — *"This image does not appear to be a supported plant leaf."*

---

## 💻 Streamlit Web Application

The interactive web dashboard (`app.py`) provides:
- **Clean Layout**: Professional college-submission UI with modern typography, sidebar project specifications, and upload dropzones.
- **Image Preview & Analysis**: Instant classification with disease status, confidence score percentage, and visual class probability progress bars.
- **Agricultural Disclaimer**: Includes educational usage note for farmers and students.

---

## 📁 Project Structure

```text
plant-disease-detection/
│
├── dataset/
│   ├── README.md               # Dataset documentation & class specifications
│   └── splits.json             # Stratified dataset split pointers (4548 Train, 974 Val, 976 Test)
│
├── models/
│   ├── .gitkeep
│   ├── plant_disease_model.keras  # Saved EfficientNetV2-B0 model artifact
│   └── class_names.json           # Model class ordering configuration
│
├── results/
│   ├── .gitkeep
│   ├── training_accuracy.png   # Accuracy history graph
│   ├── training_loss.png       # Loss history graph
│   ├── confusion_matrix.png    # Seaborn confusion matrix heatmap (976 test images)
│   ├── metrics.json            # Metrics summary export
│   └── classification_report.txt # Detailed class-wise classification report
│
├── src/
│   ├── __init__.py
│   ├── prepare_dataset.py      # Dataset integrity validator & 70/15/15 stratified splitter
│   ├── model.py                # EfficientNetV2-B0 model definition & field data augmentation
│   ├── train.py                # Two-phase training pipeline with callbacks & class weights
│   ├── evaluate.py             # Test set evaluation & confusion matrix generator
│   └── predict.py              # Single-image inference engine & non-leaf rejection filter
│
├── scripts/
│   ├── test_prediction.py      # CLI utility for single image prediction
│   ├── test_external.py        # External real-world image validation script
│   └── create_sample_dataset.py# Synthetic leaf generator (TEST/DEMO ONLY)
│
├── app.py                      # Interactive Streamlit GUI web application
├── test_all_classes.py         # End-to-end verification script
├── requirements.txt            # Package dependency specification
├── README.md                   # Complete project documentation
├── .gitignore                  # Git ignore definitions
└── LICENSE                     # MIT License
```

---

## ⚙️ Installation & How to Run

### 1. Environment Setup

```bash
# Navigate to project directory
cd plant-disease-detection

# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dataset Preparation (If dataset is present)
```bash
python src/prepare_dataset.py
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Evaluate Model on Test Set
```bash
python src/evaluate.py
```

### 5. Launch the Streamlit Web GUI
```bash
streamlit run app.py
```
*Access the web application at `http://localhost:8501`.*

### 6. Test Single Image via CLI
```bash
python scripts/test_prediction.py path/to/leaf_image.jpg
```

---

## ⚠️ Limitations & Future Scope

- **Scope Limitation**: Focused specifically on 5 tomato and potato classes. Unseen plant species are rejected by the unknown-image filter.
- **Domain Shift**: Controlled laboratory training datasets (PlantVillage) can differ from field photographs taken under extreme lighting or shadow conditions. Field data augmentation and entropy filtering mitigate this effect.
- **Future Work**: Incorporate spatial attention heatmaps (Grad-CAM) to visualize exact lesion regions and expand dataset diversity to multi-crop agricultural systems.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
