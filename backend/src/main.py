import io
import base64
import torch
import cv2
import numpy as np
import mlflow.pytorch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import our research components
from backend.src.core.data_processor import ImageProcessor
from backend.src.core.xai import MedicalCAM

app = FastAPI(title="MedAI Diagnostic Platform")

# Professional CORS setup for our React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI Brain once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Points to our verified 99% accuracy run
tracking_uri = "sqlite:///C:/Users/koush/OneDrive/Desktop/ML%20Projects/med_ai_platform/mlflow.db"
mlflow.set_tracking_uri(tracking_uri)
run_id = "0cfc9bf3d0de49ca8d7c0aed62e91f05"
model_uri = f"runs:/{run_id}/model"

print(f"Server Startup: Loading Medical Intelligence (Run {run_id})...")
model = mlflow.pytorch.load_model(model_uri).to(device)
model.eval()

processor = ImageProcessor()
cam_engine = MedicalCAM(model, model.target_layer)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    raw_img_resized = cv2.resize(raw_img, (224, 224))

    # 2. Process for the Model
    pil_img = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    img_tensor = processor.process_image(pil_img).unsqueeze(0).to(device)

    # 3. Generate Diagnosis & Heatmap
    # We need the raw output for the Softmax calculation
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    heatmap, class_idx = cam_engine.generate_heatmap(img_tensor)
    
    categories = ["Bengin", "Malignant", "Normal"]
    diagnosis = categories[class_idx]
    certainty = probabilities[class_idx].item() * 100

    # 4. Create Overlay (Heatmap + Original)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_img_resized, 0.6, heatmap_color, 0.4, 0)

    # 5. Encode to Base64 (for the Web Dashboard)
    _, buffer = cv2.imencode('.png', overlay)
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "diagnosis": diagnosis,
        "certainty": round(certainty, 2),
        "heatmap": f"data:image/png;base64,{overlay_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
