# System related libraries
import os
import cv2

# General purpose libraries
from PIL import Image

# PyTorch related libraries  
from torch.utils.data import Dataset 
from torchvision import transforms 

# Define a class to load Training images from a folder
class TileDatasetTrain(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(root, f) 
                               for root, _, files in os.walk(folder_path) 
                               for f in files if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                           ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  # Get the file path
        img = Image.open(img_path).convert("RGB")  # Load the image 
        if self.transform:
            img = self.transform(img) 
        return img

# Define a class to load Training images from a folder
class TileDatasetTest(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(root, f) 
                               for root, _, files in os.walk(folder_path) 
                               for f in files if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                           ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  # Get the file path
        img = Image.open(img_path).convert("RGB")  # Load the image 
        
        if self.transform:
            img = self.transform(img) 
        return img, img_path  # Return both tensor and path