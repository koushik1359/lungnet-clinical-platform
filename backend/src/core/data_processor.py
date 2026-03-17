import torch
from torchvision import transforms
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_input):
        # Handle both file paths AND PIL images (for FastAPI)
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        return self.transform(image)
