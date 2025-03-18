import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'pneumonia', 'covid', 'tuberculosis']
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        image_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, image_name))
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label from path
        label = self.classes.index(image_path.split('/')[-2])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """
    Returns transform compositions for training and validation
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
