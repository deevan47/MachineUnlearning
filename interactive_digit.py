import tkinter as tk
from tkinter import *
import PIL.Image
import PIL.ImageDraw
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.resnet import resnet18  # Importing your model structure

# ==========================================
# CONFIGURATION
# ==========================================
# PASTE YOUR CHECKPOINT PATH BELOW
CHECKPOINT_PATH = "experiments/mnist_resnet18_forget0/delete/_0217-09:56:15/ckpt.pth"
# ==========================================

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        # 1. Load the Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {CHECKPOINT_PATH}...")
        self.model = resnet18(num_classes=10)
        
        # Load weights
        try:
            # We use map_location='cpu' to be safe
            state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the path is correct and the architecture matches.")
            exit()

        # 2. Setup GUI
        self.canvas_width = 300
        self.canvas_height = 300
        
        # We draw White on Black because that is how MNIST data looks
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor="cross")
        self.canvas.pack(pady=10)

        # Labels
        self.label_pred = tk.Label(self.root, text="Prediction: ???", font=("Helvetica", 24))
        self.label_pred.pack()
        
        self.label_conf = tk.Label(self.root, text="Confidence: ???", font=("Helvetica", 14))
        self.label_conf.pack()

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack(side=LEFT, padx=10)
        
        self.btn_predict = tk.Button(btn_frame, text="Predict", command=self.predict_digit)
        self.btn_predict.pack(side=LEFT, padx=10)

        # Drawing variables
        self.image1 = PIL.Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = PIL.ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Draw on the canvas
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        
        # Draw on the memory image (that gets sent to the model)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = PIL.Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = PIL.ImageDraw.Draw(self.image1)
        self.label_pred.config(text="Prediction: ???")
        self.label_conf.config(text="Confidence: ???")

    def predict_digit(self):
        # 1. Resize image to 28x28 (Standard MNIST size)
        img = self.image1.resize((28, 28))
        
        # 2. Convert to Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
        ])
        
        input_tensor = transform(img).unsqueeze(0) # Add batch dimension [1, 1, 28, 28]
        input_tensor = input_tensor.to(self.device)

        # 3. Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get the winner
            conf, predicted = torch.max(probabilities, 1)
            
            digit = predicted.item()
            confidence = conf.item() * 100

        # 4. Update Labels
        self.label_pred.config(text=f"Prediction: {digit}")
        self.label_conf.config(text=f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()