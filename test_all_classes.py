from pathlib import Path
from src.predict import load_trained_model, predict_image


# Load model once
model, class_names = load_trained_model()


# Dataset folders and their expected labels
classes = {
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
}


print()
print("=" * 80)
print("TESTING ONE IMAGE FROM EACH CLASS")
print("=" * 80)
print()


total = 0
correct = 0


for folder, expected in classes.items():

    # Get one image from the class
    images = list(
        Path("dataset").glob(f"{folder}/*")
    )

    if not images:
        print(f"{expected:25} -> NO IMAGE FOUND")
        continue

    image_path = images[0]

    # Predict using the exact same pipeline as Streamlit
    result = predict_image(
        model,
        class_names,
        image_path
    )

    predicted = result["class_name"]
    confidence = result["confidence"]
    entropy = result["entropy"]

    is_correct = predicted == expected

    total += 1

    if is_correct:
        correct += 1

    status = "PASS" if is_correct else "FAIL"

    print(f"Expected   : {expected}")
    print(f"Predicted  : {predicted}")
    print(f"Confidence : {confidence:.4f}")
    print(f"Entropy    : {entropy:.4f}")
    print(f"Status     : {status}")
    print(f"Image      : {image_path.name}")
    print("-" * 80)


print()
print("=" * 80)
print(f"RESULT: {correct}/{total} classes correct")
print("=" * 80)
print()