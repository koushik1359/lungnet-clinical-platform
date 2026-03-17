import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class MedicalTrainer:
    """
    Senior-level Research Trainer:
    Handles the loop, hardware acceleration, and MLflow experiment tracking.
    """
    def __init__(self, model, train_loader, val_loader, lr=1e-4, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Professional standard: CrossEntropy for multi-class, Adam for stable gradients
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / len(self.val_loader), 100. * correct / total

    def fit(self, epochs=5):
        # Start MLflow run to log our progress
        with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("architecture", "ViT_B_16")

            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()
                
                print(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")
                
                # Log metrics to MLflow for professional tracking
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)

            # Save the final model artifact to MLflow
            mlflow.pytorch.log_model(self.model, "model")
