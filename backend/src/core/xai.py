import torch
import torch.nn.functional as F
import cv2
import numpy as np

class MedicalCAM:
    """
    Grad-CAM for Vision Transformers:
    Visualizes the "Attention" of the model by mapping gradients to the 14x14 sequence.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Using register_full_backward_hook for compatibility
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Clear gradients
        self.model.zero_grad()
        
        # 2. Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # 3. Backward pass
        output[0, class_idx].backward()
        
        # 4. Process ViT Tensors (Batch, 197, 768)
        # Importance weights: average over the spatial/patch dimension
        weights = torch.mean(self.gradients, dim=1, keepdim=True) # (1, 1, 768)
        
        # Weighted sum of activations across the hidden features
        cam = torch.sum(weights * self.activations, dim=2) # (1, 197)
        cam = cam.squeeze(0).detach().cpu().numpy()
        
        # 5. Extract the 196 patches (Removing CLS token at index 0)
        if cam.shape[0] == 197: 
             cam = cam[1:].reshape(14, 14)

        # 6. Normalize & Resize for High-Res Overlay
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        divisor = cam_max - cam_min + 1e-8
        cam = (cam - cam_min) / divisor
        cam = cv2.resize(cam, (224, 224))
        
        return cam, class_idx
