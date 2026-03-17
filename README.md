# 🏥 LungNet: Advanced Clinical AI Platform

[![AI Engine: ViT-B/16](https://img.shields.io/badge/AI%20Engine-ViT--B%2F16-blueviolet)](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html)
[![XAI: Grad-CAM](https://img.shields.io/badge/XAI-Grad--CAM-orange)](https://arxiv.org/abs/1610.02391)
[![Deployment: Azure $0](https://img.shields.io/badge/Deployment-Azure%20%240-0078D4)](https://azure.microsoft.com/en-us/products/container-apps)

**LungNet** is a production-grade medical imaging platform designed to bridge the gap between high-accuracy deep learning research and real-world clinical utility. It leverages SOTA Vision Transformers (ViT) and Explainable AI (XAI) to provide doctors with not just a diagnosis, but **visual proof**.

---

## 🚀 Live Demo
**Access the Clinical Dashboard**: [https://lungnet-app.delightfulmushroom-9dec91d8.eastus.azurecontainerapps.io](https://lungnet-app.delightfulmushroom-9dec91d8.eastus.azurecontainerapps.io)

---

## 🔬 Core AI Engine
*   **Architecture**: Vision Transformer (ViT-B/16) fine-tuned on clinical CT data.
*   **Metrics**: Consistent **99.5% Accuracy** across Benign, Malignant, and Normal tissue classes.
*   **Explainability**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)**. The system highlights the exact ROI (Region of Interest) the model is using to make its decision, fostering "Neural Trust."
*   **Experiment Tracking**: 100% of training runs tracked via **MLflow**, including loss curves, precision-recall, and model weights.

## 🏗️ Unified Cloud Architecture
To achieve atomic updates and 100% cost efficiency, LungNet uses a **Single-Unit Container Architecture**:
*   **Backend**: High-performance **FastAPI** gateway.
*   **Frontend**: Professional **React + Vite** dashboard with framer-motion animations.
*   **Storage**: Models handled via **Git LFS** and local-only weights to ensure 0-second deployment latency.
*   **Scaling**: Deployed to **Azure Container Apps (Consumption Plan)** with **Scale-to-Zero** enabled—resulting in a **$0/month** operational cost for personal research.

## 🛠️ Local Development

### 1. Model Preparation
Ensure you have the model weights and dependencies ready:
```bash
# Clone the repository
git clone https://github.com/koushik1359/lungnet-clinical-platform.git
cd lungnet-clinical-platform

# Install Backend Dependencies
pip install -r backend/requirements.txt
```

### 2. Run the Unified Application
```bash
# Set Python Path
export PYTHONPATH=$PYTHONPATH:.

# Start the FastAPI Server (The React frontend is served automatically)
python backend/src/main.py
```

## 🚥 CI/CD Pipeline
Fully automated via **GitHub Actions**:
1.  **Build**: Compiles the React dashboard and builds the Docker image.
2.  **Harden**: Implements **Headless OpenCV** and **Non-interactive** Debian builds for cloud stability.
3.  **Deploy**: Authenticates to Azure via OIDC/Service Principals and rotates revisions automatically.

---

### **Vision & Future Roadmap**
- [ ] Support for raw **DICOM** hospital imaging.
- [ ] 3D Voxel Segmentation for nodule volume calculation.
- [ ] Automated PDF Clinical Report generation for specialists.

**Developed with 💙 for the Medical AI Community.** 🏥🔬🧬💻
