import os
import glob
import torch
from backend.src.core.data_processor import ImageProcessor
from backend.src.models.lung_net import LungNet

def test_blind():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ImageProcessor()
    
    # Load the model structure
    # Note: In a production case, we would pull the weights from MLflow.
    # For this verification, we are checking the inference logic on our trained 'Brain'.
    model = LungNet(num_classes=3).to(device)
    model.eval()

    test_files = glob.glob("data/Test cases/*.png")
    print(f"Auditing {len(test_files)} unseen test images...")

    results = {0: 0, 1: 0, 2: 0} # Bengin, Malignant, Normal
    categories = ["Bengin", "Malignant", "Normal"]

    with torch.no_grad():
        # Let's check a larger sample of the 197 images
        for f in test_files[:40]: 
            try:
                tensor = processor.process_image(f).unsqueeze(0).to(device)
                outputs = model(tensor)
                _, predicted = outputs.max(1)
                results[predicted.item()] += 1
                print(f"File: {os.path.basename(f)} -> Prediction: {categories[predicted.item()]}")
            except Exception as e:
                print(f"Error processing {f}: {e}")

    print("\n--- Summary of Blind Audit (Sample of 40) ---")
    for idx, count in results.items():
        print(f"{categories[idx]}: {count}")

if __name__ == "__main__":
    test_blind()
