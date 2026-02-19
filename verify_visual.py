import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from models.resnet import resnet18

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Find the latest model automatically
base_dir = Path("experiments/mnist_resnet18_forget0/delete")
try:
    # Get all ckpt.pth files and sort so the latest timestamp is last
    checkpoints = sorted(list(base_dir.glob("*/ckpt.pth")))
    if not checkpoints:
        raise FileNotFoundError
    latest_ckpt = checkpoints[-1]
    print(f"Loading latest model: {latest_ckpt}")
except:
    print("Could not find the model automatically.")
    print("Please check your 'experiments' folder.")
    exit()

# 3. Load Model Architecture & Weights
model = resnet18(num_classes=10)
try:
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 4. Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Downloads to ./data if needed
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 5. Get specific examples
# We want to see some 0s (Forgotten) and some 1s (Retained)
zeros_idx = [i for i, (img, label) in enumerate(testset) if label == 0][:5]
ones_idx  = [i for i, (img, label) in enumerate(testset) if label == 1][:5]

indices = zeros_idx + ones_idx

# 6. Generate the Visual Proof
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle(f"Unlearning Verification\nModel: {latest_ckpt.parent.name}", fontsize=14)

print("\n--- Generating Visual Proof ---")

with torch.no_grad():
    for i, idx in enumerate(indices):
        img, label = testset[idx]
        img_tensor = img.unsqueeze(0).to(device)
        
        # Predict
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        pred_label = predicted.item()
        conf_val = confidence.item() * 100
        
        # Setup Plot Grid
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Show Image (MNIST is 1 channel, need to squeeze)
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        
        # --- LOGIC FOR SUCCESS ---
        # For Digit 0: Success means Prediction is NOT 0.
        # For Digit 1: Success means Prediction IS 1.
        
        if label == 0:
            if pred_label != 0:
                color = 'blue' # Good! We forgot it.
                status = "FORGOTTEN"
            else:
                color = 'red'  # Bad! We still remember it.
                status = "FAILED"
        else:
            if pred_label == label:
                color = 'green' # Good! We remembered it.
                status = "RETAINED"
            else:
                color = 'orange' # Bad? We accidentally damaged this knowledge.
                status = "DAMAGED"

        ax.set_title(f"True: {label}\nPred: {pred_label}\n{status}", color=color, fontweight='bold', fontsize=10)
        ax.axis('off')

# Save to file
save_path = "visual_proof.png"
plt.tight_layout()
plt.savefig(save_path)
print(f"Saved visualization to: {save_path}")
print("Please open this image file to see the results.")