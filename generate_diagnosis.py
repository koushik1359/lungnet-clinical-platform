import torch, cv2, numpy as np
import mlflow.pytorch
from backend.src.core.data_processor import ImageProcessor
from backend.src.core.xai import MedicalCAM

def run_diagnostic(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. LOAD THE TRAINED BRAIN (Upset-Shark-367)
    # This points directly to the 99% accuracy weights in your database
    tracking_uri = "sqlite:///C:/Users/koush/OneDrive/Desktop/ML%20Projects/med_ai_platform/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    
    run_id = "0cfc9bf3d0de49ca8d7c0aed62e91f05"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Loading Medical Intelligence from Run {run_id}...")
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    
    processor = ImageProcessor()
    
    # 2. Attach the XAI "Eyes"
    # Note: We use the target_layer name from your LungNet architecture
    cam_engine = MedicalCAM(model, model.target_layer)
    
    # 3. Process the scan
    img_tensor = processor.process_image(image_path).unsqueeze(0).to(device)
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224, 224))

    # 4. Generate the Diagnosis
    heatmap, class_idx = cam_engine.generate_heatmap(img_tensor)
    categories = ["Bengin", "Malignant", "Normal"]
    print(f"--- Final Clinical Diagnosis ---")
    print(f"Prediction: {categories[class_idx]}")

    # 5. Save the Heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_image, 0.6, heatmap_color, 0.4, 0)

    output_path = "diagnostic_output.png"
    cv2.imwrite(output_path, overlay)
    print(f"Deep Diagnostic Map saved to: {output_path}")

if __name__ == "__main__":
    # Test on a known MALIGNANT case
    target = "data/Test cases/000019_01_01_021.png" 
    run_diagnostic(target)
