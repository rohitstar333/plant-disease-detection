# PlantVillage Dataset Setup

To train the model on real plant disease leaf images:

1. Download the **PlantVillage dataset** (e.g. from Kaggle or official repository).
2. Extract the dataset into this directory (`dataset/`).
3. Ensure the exact following 5 subfolders exist inside `dataset/`:

```text
dataset/
├── Tomato___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Potato___Early_blight/
└── Potato___Late_blight/
```

### Class Mapping Table

| Folder Name | Clean Display Name |
| :--- | :--- |
| `Tomato___healthy` | Tomato Healthy |
| `Tomato___Early_blight` | Tomato Early Blight |
| `Tomato___Late_blight` | Tomato Late Blight |
| `Potato___Early_blight` | Potato Early Blight |
| `Potato___Late_blight` | Potato Late Blight |

---

### Quick Verification

After placing the folders, run:

```bash
python src/prepare_dataset.py
```

This will validate image integrity and create deterministic dataset splits (`dataset/splits.json`).

> [!NOTE]
> If you wish to quickly test the code pipeline (training, evaluation, Streamlit UI) without downloading the full dataset, you can generate synthetic test images using:
>
> ```bash
> python scripts/create_sample_dataset.py
> ```
>
> *(Note: `create_sample_dataset.py` is for **TEST/DEMO ONLY** and does not replace real field data).*
