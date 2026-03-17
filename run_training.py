import os
import torch
from torch.utils.data import DataLoader
from backend.src.core.data_processor import ImageProcessor
from backend.src.core.dataset import LungDataset, get_data_splits
from backend.src.models.lung_net import LungNet
from backend.src.core.trainer import MedicalTrainer

def main():
    # 1. Configuration
    DATA_DIR = "data/raw"
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    print("--- Phase 1: Data Stratification ---")
    (train_paths, train_labels), (val_paths, val_labels) = get_data_splits(DATA_DIR)
    
    # 2. Initialize Components
    processor = ImageProcessor()
    
    train_ds = LungDataset(train_paths, train_labels, processor)
    val_ds = LungDataset(val_paths, val_labels, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Loaded {len(train_ds)} training images and {len(val_ds)} validation images.")

    print("\n--- Phase 2: Model Initialization (ViT) ---")
    model = LungNet(num_classes=3)
    
    print("\n--- Phase 3: Commencing Training with MLflow ---")
    trainer = MedicalTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        lr=LEARNING_RATE
    )
    
    trainer.fit(epochs=EPOCHS)
    
    print("\nTraining Complete. Model and metrics saved to MLflow.")

if __name__ == "__main__":
    main()
