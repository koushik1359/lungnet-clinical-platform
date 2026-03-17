import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .data_processor import ImageProcessor

class LungDataset(Dataset):
    """
    Senior Data Science approach:
    Encapsulates the data list and the processor into a PyTorch-compatible object.
    """
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Use our previously created ImageProcessor
        image_tensor = self.processor.process_image(img_path)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)

def get_data_splits(data_dir="data/raw", test_size=0.2):
    """
    Professional Stratified Split:
    Ensures that our Training and Validation sets have the SAME percentage 
    of each class to prevent model bias.
    """
    categories = ["Bengin cases", "Malignant cases", "Normal cases"]
    image_paths = []
    labels = []

    for idx, cat in enumerate(categories):
        cat_dir = os.path.join(data_dir, cat)
        files = glob.glob(os.path.join(cat_dir, "*.jpg"))
        image_paths.extend(files)
        labels.extend([idx] * len(files))

    # Senior Tip: Random State 42 is the industry 'meme' standard for reproducibility.
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    return (train_paths, train_labels), (val_paths, val_labels)

# Senior Note: 'stratify=labels' is critical. It guarantees that our 20% validation 
# set isn't accidentally 100% 'Normal' cases, which would make our metrics useless.
