import torch
import torch.nn as nn
from pathlib import Path
from .resnet import resnet18

def get_model(model_name, num_classes):
    """Get model by name"""
    if model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported for MNIST")
    
    return model

def load_model(model_path, model_name, num_classes):
    """Load pre-trained model"""
    model = get_model(model_name, num_classes)
    
    model_path = Path(model_path)
    if model_path.exists():
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model path {model_path} not found, returning untrained model")
    
    model = model.to("cuda")
    return model