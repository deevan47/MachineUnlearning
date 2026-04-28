# ERASE - Efficient Removal of Acquired Selective Experience

This project implements the ERASE algorithm for machine unlearning, enabling selective removal of specific classes from trained neural networks without full retraining. It applies the DELETE method to ResNet-18 models on MNIST and CIFAR-10 datasets, demonstrating effective unlearning of targeted categories while preserving performance on retained classes.

## Project Structure

```
minst/
├── main.py                 # Interactive main script for dataset selection and unlearning
├── trainer.py              # Model training, testing, and checkpoint management
├── utils.py                # Data preprocessing, seeding, and utility functions
├── config/
│   ├── mnist_resnet18.yaml     # MNIST experiment configuration
│   └── cifar10_resnet18.yaml   # CIFAR-10 experiment configuration
├── data/                   # Automatic dataset downloads (MNIST/CIFAR-10)
├── evaluation/             # Evaluation and attack implementations
│   ├── MIA.py                  # Membership inference attack
│   ├── SVC_MIA.py              # SVM-based MIA evaluation
│   ├── advanced_metrics.py     # t-SNE visualization and confusion matrices
│   └── __init__.py
├── experiments/            # Output directory for models and results
│   ├── pretrained_model/       # Pre-trained and unlearned model checkpoints
│   └── [experiment_name]/      # Individual experiment directories
├── method/                 # Unlearning algorithm implementations
│   ├── delete.py               # Core DELETE algorithm
│   ├── finetune.py             # Fine-tuning baseline method
│   ├── gradient_ascent.py      # Gradient ascent baseline
│   ├── boundary_shrink.py      # Boundary shrink baseline
│   ├── random_label.py         # Random label baseline
│   ├── _adv_generator.py       # Adversarial example utilities
│   └── utils.py                # Method evaluation utilities
└── models/
    ├── resnet.py               # ResNet-18 architecture
    └── __init__.py
```

## Tech Stack and Dependencies

- **Python** 3.7+
- **PyTorch** 1.7+ (neural network framework)
- **Torchvision** (dataset handling and transforms)
- **NumPy** (numerical operations)
- **Matplotlib** (plotting utilities)
- **OmegaConf** (YAML configuration loading)
- **tqdm** (progress indicators)
- **scikit-learn** (SVM for MIA)
- **seaborn** (enhanced plotting)

## Setup and Installation

1. Ensure Python 3.7+ is installed.

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install required packages:
   ```
   pip install torch torchvision numpy matplotlib omegaconf tqdm scikit-learn seaborn
   ```

4. Verify CUDA availability for GPU acceleration (optional but recommended).

## How It Works

The ERASE system provides an interactive pipeline for machine unlearning:

1. **Dataset Selection**: Choose between MNIST or CIFAR-10 datasets.

2. **Model Loading**: Load a pre-trained ResNet-18 model (optionally with existing forgotten classes).

3. **Class Selection**: Specify which classes to unlearn from the available unforgotten classes.

4. **Unlearning Execution**: Apply the DELETE algorithm:
   - Generate soft targets using the original model's predictions
   - Set logits for forgotten classes to negative infinity
   - Optimize using KL divergence loss to match modified targets
   - Monitor accuracy on retained classes during training

5. **Evaluation**: Generate comprehensive metrics:
   - Before/after accuracy comparison plots
   - t-SNE visualizations of feature space changes
   - Confusion matrices for forgotten class analysis
   - Membership inference attack (MIA) using SVM to verify unlearning effectiveness

## Dataset

Supports two standard vision datasets, downloaded automatically:

- **MNIST**: 70,000 handwritten digit images (60k train, 10k test), 28×28 grayscale, 10 classes
- **CIFAR-10**: 60,000 object images (50k train, 10k test), 32×32 RGB, 10 classes

Data partitioning occurs dynamically based on selected forget classes.

## Results or Output

Execution produces:

- **Model Checkpoints**: Saved unlearned models in `.pth` format with metadata
- **Accuracy Reports**: Per-class accuracy tables before and after unlearning
- **Visualization Plots**: 
  - Accuracy comparison bar charts
  - t-SNE embeddings showing feature space changes
  - Confusion matrices for detailed classification analysis
- **MIA Results**: SVM-based attack accuracy indicating unlearning success
- **Experiment Logs**: Timestamped directories containing all outputs

## Known Limitations or Notes

- Interactive execution requires manual input for dataset and class selection
- Pre-trained models may already have some classes forgotten
- Unlearning multiple classes simultaneously may affect retained class accuracy
- DELETE algorithm parameters are fixed; no hyperparameter tuning implemented
- Evaluation metrics focus on classification accuracy and MIA; no robustness testing
- GPU required for reasonable execution times on full datasets
- Console output with detailed accuracy tables and evaluation metrics
- Experiment results saved in timestamped directories under `experiments/`

## Known Limitations or Notes

- Currently implements only the DELETE algorithm; other methods are present but not fully integrated
- Unlearning may cause minor accuracy degradation (1-2%) on retained classes
- Requires GPU for reasonable training times; CPU training is possible but slow
- Membership inference attack evaluation uses shadow models for attack training
- Configuration is hardcoded for ResNet-18 architecture only
- No hyperparameter optimization implemented; uses fixed values from config files

## How to Actually Run This

### 1. Train from the beginning:

```bash
python main.py --train_from_scratch
```

This trains a fresh ResNet18 on all digits 0-9. Takes a while. Saves checkpoints for each epoch under `experiments/pretrained_model/`.

### 2. Run the actual unlearning:

```bash
python main.py
```

After this runs, you'll get prompted to enter which digits you want to forget. Like if you type `123`, the model will forget digits 1, 2, and 3. You can use formats like:
- `123` (compact)
- `1, 2, 3` (with spaces/commas)
- basically anything, the script splits it into individual digits

**Cool feature:** If you already unlearned `123` once, and you run this again and say `123`, it loads the ALREADY-unlearned model instead of starting over. So it remembers the forget state.

If you want to unlearn `456` after that, it starts from the original model (if `456` wasn't unlearned before) OR from the saved `456` model (if it was).

### 3. What Gets Saved

- Each run creates a folder with a timestamp under `experiments/`
- The unlearned model also gets saved to `experiments/pretrained_model/mnist_resnet18_forget<digits>_unlearned.pth`
- This lets you reload the same forgotten model later without redoing the work

### 4. Reset Everything

If you want a fresh start and clear all the unlearned models:

```bash
python main.py --train_from_scratch
```

This trains a brand new original model and wipes out all previous forget states. Next time you run the interactive part, it's like starting fresh.