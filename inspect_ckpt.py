import torch
import os

ckpt_path = "experiments/mnist_resnet18_forget0/delete/_0217-09:56:15/ckpt.pth"

if not os.path.exists(ckpt_path):
    print(f"Error: File not found at {ckpt_path}")
    print("Please check the timestamp folder name in 'experiments/mnist_resnet18_forget0/delete/'")
    exit()


content = torch.load(ckpt_path, map_location='cpu')

print(f"Data type: {type(content)}")

if isinstance(content, dict):
    print("\nKeys found in the .pth file:")
    for key, value in content.items():
        if torch.is_tensor(value):
            print(f" - {key:<30} | Shape: {value.shape}")
        else:
            print(f" - {key:<30} | Value: {value}")
else:
    print("No file")