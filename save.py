import torch
from models.resnet import resnet18

model = ResNet18(num_classes=10) 

checkpoint_path = "experiments/mnist_resnet18_forget0/delete/_0217-09:56:15/ckpt.pth"
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint) 

print("Model loaded successfully!")