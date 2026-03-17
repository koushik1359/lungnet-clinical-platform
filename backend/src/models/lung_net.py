import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

class LungNet(nn.Module):
    """
    SOTA Medical AI Model: 
    Vision Transformer (ViT) fine-tuned for 3-class Lung Cancer classification.
    """
    def __init__(self, num_classes=3):
        super(LungNet, self).__init__()
        
        # Load pre-trained weights from ImageNet
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Replace the head for 3 classes
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

    @property
    def target_layer(self):
        # We hook into the very last Transformer Encoder block for the best features
        return self.vit.encoder.layers[-1].ln_1
