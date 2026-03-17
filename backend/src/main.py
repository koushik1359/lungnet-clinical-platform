import io
import base64
import torch
import cv2
import numpy as np
import mlflow.pytorch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from fastapi.responses import FileResponse
import os
import logging

# Configure logging for Azure Portal traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedAI")

# Import our research components
from backend.src.core.data_processor import ImageProcessor
from backend.src.core.xai import MedicalCAM
from backend.src.models.lung_net import LungNet

app = FastAPI(title="MedAI Diagnostic Platform")

# Professional CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check for Azure Probes
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "LungNet AI"}

# Safer Static Assets Handling
# We serve the React build only if it's there (Atomic update protection)
static_path = "static"
if os.path.exists(static_path):
    logger.info(f"✓ Found static frontend folder at {static_path}")
    
    # Mount assets IF they exist (Standard Vite structure)
    assets_path = os.path.join(static_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        logger.info("✓ Assets folder mounted")

    @app.get("/")
    async def read_index():
        index_file = os.path.join(static_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "LungNet AI API Online. index.html missing."}
else:
    logger.warning("! static folder not found. Running in API-only mode.")

# Load the AI Brain once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standalone Deployment Mode: Load from .pth file
# Use environment variable or default Docker path
model_path = os.getenv("MODEL_PATH", "backend/src/models/lungnet_best.pth")
logger.info(f"Server Startup: Loading Medical Intelligence (Standalone Mode)...")

model = LungNet(num_classes=3).to(device)
if os.path.exists(model_path):
    try:
        # Load our fine-tuned weights into the architecture
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"✓ AI Weights Loaded from {model_path}")
    except Exception as e:
        logger.error(f"⚠ FAILED to load weights: {e}")
else:
    logger.error(f"⚠ Model weights NOT FOUND at {model_path}")

model.eval()

processor = ImageProcessor()
cam_engine = MedicalCAM(model, model.target_layer)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if raw_img is None:
        return {"status": "error", "message": "Invalid image format"}
        
    raw_img_resized = cv2.resize(raw_img, (224, 224))

    # 2. Process for the Model
    pil_img = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    img_tensor = processor.process_image(pil_img).unsqueeze(0).to(device)

    # 3. Generate Diagnosis & Heatmap
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

    # 5. Encode to Base64
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
    # Respect PORT env var for Azure/Cloud compatibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
