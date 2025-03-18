import torch
import torch.nn as nn
from torchvision import models

class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(MedicalImageClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify final layers
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
