import argparse
import copy
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import sys
import re
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import *
from trainer import *
from models import load_model
from method import random_label, finetune, gradient_ascent, boundary_shrink, delete
import evaluation
import log_utils
from evaluation.advanced_metrics import AdvancedMetrics

DATASET_CONFIG = {
    'mnist': {
        'num_classes': 10,
        'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'in_channels': 1,
        'loader': lambda path: (
            datasets.MNIST(str(path), train=True, download=True, transform=None),
            datasets.MNIST(str(path), train=False, download=True, transform=None)
        )
    },
    'cifar10': {
        'num_classes': 10,
        'class_names': ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
        'in_channels': 3,
        'loader': lambda path: (
            datasets.CIFAR10(str(path), train=True, download=True, transform=None),
            datasets.CIFAR10(str(path), train=False, download=True, transform=None)
        )
    }
}

def plot_accuracy_comparison(before_acc, after_acc, class_names, forgotten_classes, active_classes, save_path):
    """Generate before/after accuracy comparison graph for ALL classes (forgotten + active)"""
    all_classes = sorted(set(range(len(class_names))))
    num_classes = len(all_classes)
    x = np.arange(num_classes)
    width = 0.35
    
    # Prepare data and colors
    before_data = [before_acc[i] for i in all_classes]
    after_data = [after_acc[i] for i in all_classes]
    
    # Color bars differently for forgotten vs active classes
    colors_before = ['#ff6b6b' if i in forgotten_classes else '#4ecdc4' for i in all_classes]
    colors_after = ['#ff3333' if i in forgotten_classes else '#2d9d8f' for i in all_classes]
    
    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - width/2, before_data, width, label='Before Unlearning', color=colors_before, alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, after_data, width, label='After Unlearning', color=colors_after, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy: Before vs After Unlearning\n(Red = Forgotten Classes, Teal = Active Classes)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    all_class_names = [class_names[i] for i in all_classes]
    ax.set_xticklabels([f'{all_classes[i]}\n({all_class_names[i][:10]})' for i in range(num_classes)], fontsize=9)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for forgotten vs active
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ecdc4', edgecolor='black', label='Active Classes (Before)'),
        Patch(facecolor='#2d9d8f', edgecolor='black', label='Active Classes (After)'),
        Patch(facecolor='#ff6b6b', edgecolor='black', label='Forgotten Classes (Before)'),
        Patch(facecolor='#ff3333', edgecolor='black', label='Forgotten Classes (After)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved graph to {save_path}")

def get_class_accuracies(model, loader, device, num_classes):
    """Get per-class accuracy"""
    model.eval()
    class_correct = [0.0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            if c.ndim == 0:
                label = labels.item()
                class_correct[label] += c.item()
                class_total[label] += 1
            else:
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
    accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return accuracies, class_total

def print_class_accuracy(model, loader, device, class_names, active_classes=None, forgotten_classes=None, title="Model Accuracy"):
    """Print per-class accuracy table (shows all classes by default)"""
    num_classes = len(class_names)
    if active_classes is None:
        active_classes = set(range(num_classes))
    if forgotten_classes is None:
        forgotten_classes = set()
    
    print(f"\n-- {title} --")
    accuracies, class_total = get_class_accuracies(model, loader, device, num_classes)
    
    print(f"{'Class':<10} | {'Name':<20} | {'Accuracy':<10} | {'Count':<8}")
    print()
    for i in range(num_classes):
        name = class_names[i][:19]
        acc_str = f"{accuracies[i]:.2f}%" if class_total[i] > 0 else "N/A"
        print(f"{i:<10} | {name:<20} | {acc_str:<10} | {class_total[i]:<8}")
    print()
    
    return accuracies

def display_accuracies_from_list(accuracies, class_names, forgotten_classes, title="Accuracy"):
    """Display accuracies from a pre-computed list"""
    num_classes = len(class_names)
    if forgotten_classes is None:
        forgotten_classes = set()
    
    print(f"\n-- {title} --")
    print(f"{'Class':<10} | {'Name':<20} | {'Accuracy':<10}")
    print()
    for i in range(num_classes):
        name = class_names[i][:19]
        acc_str = f"{accuracies[i]:.2f}%"
        print(f"{i:<10} | {name:<20} | {acc_str:<10}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Machine Unlearning - MNIST & CIFAR-10")
    
    parser.add_argument('--method', type=str, default="delete", choices=['delete'])
    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18'])
    parser.add_argument('--train_from_scratch', action='store_true')
    
    args = parser.parse_args()
    
    # Load config and set defaults
    config = OmegaConf.load(f'config/{args.dataset_name}_{args.model_name}.yaml')
    args.batch_size = config.batch_size
    args.pretrain_epoch = config.pretrain_epoch
    args.pretrain_lr = config.pretrain_lr
    args.unlearn_epoch = config.unlearn_epoch
    args.unlearn_rate = config.unlearn_rate
    args.num_workers = getattr(config, 'num_workers', 2)
    args.seed = getattr(config, 'seed', 2022)
    args.optim_name = getattr(config, 'optim_name', 'sgd')
    args.soft_label = getattr(config, 'soft_label', 'inf')
    args.num_forget = getattr(config, 'num_forget', 5000)
    args.exps_dir = getattr(config, 'exps_dir', 'experiments')
    args.description = getattr(config, 'description', '')
    
    # Setup
    seed_torch(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path(args.exps_dir).expanduser()
    create_dir(path)
    
    if args.train_from_scratch:
        print("\nTraining MNIST model from scratch...")
        
        model_name = args.model_name
        ckpt_path = path / "pretrained_model"
        create_dir(ckpt_path)
        
        # Train MNIST
        print("\n Training MNIST Model ")
        args.dataset_name = 'mnist'
        dataset_cfg = DATASET_CONFIG[args.dataset_name]
        num_classes = dataset_cfg['num_classes']
        class_names = dataset_cfg['class_names']
        in_channels = dataset_cfg['in_channels']
        transform_train, transform_test = get_transforms(args.dataset_name, args.model_name)
        trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
        train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, args.num_workers)
        train_save_model(train_loader, test_loader, model_name, args.optim_name, args.pretrain_lr, args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_original_model", num_classes, in_channels)
        
        # Train CIFAR-10
        print("\n Training CIFAR-10 Model ")
        args.dataset_name = 'cifar10'
        dataset_cfg = DATASET_CONFIG[args.dataset_name]
        num_classes = dataset_cfg['num_classes']
        class_names = dataset_cfg['class_names']
        in_channels = dataset_cfg['in_channels']
        transform_train, transform_test = get_transforms(args.dataset_name, args.model_name)
        trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
        train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, args.num_workers)
        train_save_model(train_loader, test_loader, model_name, args.optim_name, args.pretrain_lr, args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_original_model", num_classes, in_channels)
        
        print("\nModel trained and saved.")
        exit(0)
    
    # Select dataset for unlearning
    print("\nSelect Dataset for Unlearning:")
    print("1) MNIST")
    print("2) CIFAR-10")
    try:
        dataset_choice = input("Enter choice: ").strip()
        if dataset_choice == '1':
            args.dataset_name = 'mnist'
        elif dataset_choice == '2':
            args.dataset_name = 'cifar10'
        else:
            print("Invalid input. Expected 1 or 2.")
            exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        exit(0)
    except Exception as e:
        print(f"Invalid input: {str(e)}")
        exit(1)
    
    dataset_cfg = DATASET_CONFIG.get(args.dataset_name)
    if not dataset_cfg:
        print(f"Dataset '{args.dataset_name}' not supported.")
        exit(1)
    
    num_classes = dataset_cfg['num_classes']
    class_names = dataset_cfg['class_names']
    in_channels = dataset_cfg['in_channels']
    
    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name)
    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, args.num_workers)

    ckpt_path = path / "pretrained_model"
    create_dir(ckpt_path)
    model_name = args.model_name

    # Load the most recent model for the selected dataset
    all_model_files = list(ckpt_path.glob(f'{args.dataset_name}_{model_name}_*.pth'))
    if not all_model_files:
        print(f"No pretrained {args.dataset_name.upper()} model found! Please run with --train_from_scratch first.")
        exit(1)
    
    most_recent_model = max(all_model_files, key=lambda x: x.stat().st_mtime)
    
    metadata_path = most_recent_model.with_suffix('.json')
    previously_forgotten = set()
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            previously_forgotten = set(metadata.get('forgotten', []))
    
    print(f"Loading {args.dataset_name.upper()} model from {most_recent_model}...")
    ori_model = load_model(most_recent_model, model_name, num_classes, in_channels)
    ori_model = ori_model.to(device)
    
    active_classes = set(range(num_classes)) - previously_forgotten
    if active_classes:
        # print(f"\nCurrent model has forgotten classes: {sorted(previously_forgotten) if previously_forgotten else 'None'}")
        print_class_accuracy(ori_model, test_loader, device, class_names, title=f"CURRENT {args.dataset_name.upper()} MODEL ACCURACY", forgotten_classes=previously_forgotten)
    else:
        print("All classes have been forgotten. No active classes left.")
        exit(0)

    print(f"\nAvailable classes to unlearn: {sorted(active_classes)}")
    print(f"Class names: {', '.join([f'{i}={class_names[i]}' for i in sorted(active_classes)])}")

    print("\nEnter classes to unlearn:")
    try:
        user_input = input(
            f"Enter the Class Numbers for DELETEING/FORGETTING (e.g., 2 or 123):\n")
        s = user_input.strip()
        
        if not s:
            print("No input provided. Exiting.")
            exit(0)
        
        invalid_chars = [c for c in s if not c.isdigit() and c not in [',', ' ', ';']]
        if invalid_chars:
            print(f"Invalid input: {invalid_chars}")
            exit(1)
        
        if any(sep in s for sep in [',', ' ', ';']):
            tokens = re.findall(r"\d+", s)
        else:
            tokens = list(s) if len(s) > 1 else [s]
        
        forget_classes = []
        for t in tokens:
            try:
                c = int(t)
                if c < 0 or c >= num_classes:
                    print(f"Invalid request: Class {c} is out of range [0-{num_classes-1}]")
                    exit(1)
                if c in previously_forgotten:
                    print(f"Invalid request/ Datapoint doesn't exist: Class")
                    exit(1)
                forget_classes.append(c)
            except ValueError:
                print(f"'{t}' is not a number")
                exit(1)
        
        forget_classes = sorted(list(set(forget_classes)))
        if not forget_classes:
            print("Invalid request: No valid classes entered")
            exit(1)
        args.forget_class = forget_classes
    except KeyboardInterrupt:
        print("\nAborted by user.")
        exit(0)
    except Exception as e:
        print(f"Invalid request: {str(e)}")
        exit(1)

    requested_set = set(args.forget_class)

    # Check for conflicts and warn about already-forgotten classes
    print()
    already_forgotten = previously_forgotten & requested_set
    newly_forgetting = requested_set - previously_forgotten
    
    if already_forgotten:
        print(f"Classes {sorted(already_forgotten)} do not exist in the dataset.")
    
    if not newly_forgetting:
        print("No valid classes to unlearn.\n")
        print_class_accuracy(ori_model, test_loader, device, class_names, title="MODEL ACCURACY", forgotten_classes=previously_forgotten)
        exit(0)
    
    # Only unlearn the new classes
    args.forget_class = sorted(list(newly_forgetting))
    print(f"Unlearning classes: {', '.join(map(str, args.forget_class))}\n")

    forget_class_index = args.forget_class
    train_forget_loader, train_remain_loader, test_forget_loader, \
    test_remain_loader, _, _, _, _, _ = \
        get_unlearn_loader(trainset, testset, forget_class_index,
                          args.batch_size, args.num_forget, args.num_workers)

    forget_str = ''.join(map(str, sorted(args.forget_class)))
    description = f"{args.dataset_name}_{model_name}_forget{forget_str}"
    vice_description = f"{args.description}"
    
    create_dir(path / description)
    method_description = f"{args.method}"
    create_dir(path / description / method_description)
    create_dir(path / description / method_description / vice_description)
    
    logger, console_handler = log_utils.setup_logger(
        path / description / method_description / vice_description,
        logger_name="train_log")
    
    loader_dict = {
        "train_forget": train_forget_loader,
        "train_remain": train_remain_loader,
        "test_forget": test_forget_loader,
        "test_remain": test_remain_loader,
        "test": test_loader,
    }

    experiment_path = path / description / method_description / vice_description
    print("Starting unlearning process...\n")
    
    # Capture accuracies before unlearning
    print("Before unlearning:")
    before_accuracies, _ = get_class_accuracies(ori_model, test_loader, device, num_classes)
    display_accuracies_from_list(before_accuracies, class_names, set(args.forget_class), 
                                 title=f"BEFORE UNLEARNING - {args.dataset_name.upper()} MODEL ACCURACY")
    
    if args.method == "delete":
        unlearn_model = delete(ori_model, train_forget_loader,
                               args.unlearn_epoch, args.unlearn_rate,
                               logger=logger,
                               console_handler=console_handler,
                               loader_dict=loader_dict,
                               experiment_path=experiment_path,
                               soft_label=args.soft_label)
    else:
        print(f"Method '{args.method}' is not supported (only 'delete' is available).")
        exit()

    # Capture accuracies after unlearning
    print("After unlearning:")
    after_accuracies, _ = get_class_accuracies(unlearn_model, test_loader, device, num_classes)
    
    # Calculate new forgotten and active classes
    new_forgotten = previously_forgotten | set(args.forget_class)
    new_active = set(range(num_classes)) - new_forgotten
    
    # Generate comparison graph
    print("Creating graph...")
    graph_filename = f"{args.dataset_name}_{model_name}_forget{forget_str}_comparison.png"
    plot_accuracy_comparison(before_accuracies, after_accuracies, class_names, 
                             sorted(args.forget_class), new_active,
                             ckpt_path / graph_filename)
    print(f"Graph saved to {ckpt_path / graph_filename}\n")
    print_class_accuracy(unlearn_model, test_loader, device, class_names, title="FINAL MODEL ACCURACY (After Unlearning)", forgotten_classes=new_forgotten)

    # Save model
    if unlearn_model:
        torch.save(unlearn_model.state_dict(), experiment_path / "ckpt.pth")
        logger.info(f"Model saved to {experiment_path / 'ckpt.pth'}")
        # also persist a copy in pretrained_model folder for reuse
        forget_str = ''.join(map(str, sorted(args.forget_class)))
        copy_path = ckpt_path / f'{args.dataset_name}_{model_name}_forget{forget_str}_unlearned.pth'
        torch.save(unlearn_model.state_dict(), copy_path)
        # save metadata
        metadata_path = copy_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({'forgotten': sorted(new_forgotten)}, f)
        logger.info(f"Unlearned model persisted to {copy_path}")
        print("Model saved.\n")
    
    # Generate metrics (t-SNE and confusion matrix)
    print("Generating metrics...")
    metrics_save_dir = experiment_path / "metrics"
    metrics = AdvancedMetrics(
        model_before=ori_model,
        model_after=unlearn_model,
        test_forget_loader=test_forget_loader,
        test_remain_loader=test_remain_loader,
        forgotten_classes=args.forget_class,
        previously_forgotten_classes=previously_forgotten,
        class_names=class_names,
        device=device,
        save_dir=str(metrics_save_dir)
    )
    
    try:
        metrics.plot_tsne_comparison()
        metrics.plot_confusion_matrix_forgotten()
        print(f"Metrics saved to {metrics_save_dir}\n")
    except Exception as e:
        print(f"Warning: Could not generate metrics: {str(e)}\n")
    
    # Evaluation
    logger.info("Starting MIA evaluation")
    mia_result = evaluation.SVC_MIA(
        shadow_train=train_remain_loader,
        shadow_test=test_remain_loader,
        target_train=train_forget_loader,
        target_test=None,
        model=unlearn_model,
    )
    logger.info(f"Unlearn model MIA result:\n{mia_result}")
    logger.info("Execution completed")
