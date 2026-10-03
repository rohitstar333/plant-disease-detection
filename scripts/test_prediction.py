"""
=============================================================================
             SINGLE IMAGE PREDICTION TEST CLI UTILITY
=============================================================================
Usage:
    python scripts/test_prediction.py path/to/leaf_image.jpg
=============================================================================
"""

import sys
from pathlib import Path

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predict import load_trained_model, predict_image


def main():
    if len(sys.argv) < 2:
        print("Usage error: Missing target image path.")
        print("Example: python scripts/test_prediction.py path/to/leaf.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Target image file not found at: {image_path}")
        sys.exit(1)

    try:
        # Load trained model & class list
        model, class_names = load_trained_model()
        result = predict_image(model, class_names, image_path)

        print("\n=========================================================")
        print("   PLANT DISEASE PREDICTION (EFFICIENTNETV2-B0)          ")
        print("=========================================================")
        print(f"Image File      : {image_path.name}")

        if result.get("is_unrecognized", False):
            print(f"Status          : ⚠️ IMAGE NOT RECOGNIZED")
            print(f"Reason          : {result.get('unrecognized_reason', '')}")
        else:
            print(f"Predicted Class : {result['class_name']}")
            print(f"Confidence      : {result['confidence'] * 100:.2f}%")
            print("---------------------------------------------------------")
            print("Class Probabilities Breakdown:")
            
            sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
            for cls_name, prob in sorted_probs:
                print(f"  - {cls_name:<22}: {prob * 100:>6.2f}%")
        
        print("=========================================================\n")

    except FileNotFoundError as fnf_err:
        print(f"\n[Error] {fnf_err}")
        sys.exit(1)
    except Exception as err:
        print(f"\n[Unexpected Error] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
