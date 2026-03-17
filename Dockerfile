# Stage 1: Build Frontend
FROM node:20-alpine as frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Backend
FROM python:3.11-slim as backend-runner
WORKDIR /app

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV (Robustly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and model weights
COPY backend/ ./backend/
# Ensure the model weights are actually there (Note: Manual upload might be needed if Git LFS isn't used)
COPY backend/src/models/lungnet_best.pth ./backend/src/models/lungnet_best.pth

# Copy built frontend assets to a static folder for FastAPI to serve
COPY --from=frontend-build /app/frontend/dist ./static

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
