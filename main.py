import argparse
import copy
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import *
from trainer import *
from models import load_model
from method import random_label, finetune, gradient_ascent, boundary_shrink, delete
import evaluation
import log_utils

def print_class_accuracy(model, loader, device, title="Model Accuracy"):
    print(f"\n{'-'*20} {title} {'-'*20}")
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    print("Evaluating...", end="\r")
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            # Handle batch size nuances
            if c.ndim == 0: # Single item batch
                label = labels.item()
                class_correct[label] += c.item()
                class_total[label] += 1
            else:
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    print(f"{'Class':<10} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 35)
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Digit {i:<4} | {acc:.2f}%     | {int(class_total[i])}")
        else:
            print(f"Digit {i:<4} | N/A        | 0")
    print("-" * 35 + "\n")

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST Machine Unlearning")
    
    # Method and dataset settings
    parser.add_argument('--method', type=str, default="delete", choices=['random_label', "finetune", "gradient_ascent", 'boundary_shrink', "delete"])
    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist'])
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18'])
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--pretrain_epoch', type=int, default=None)
    parser.add_argument('--pretrain_lr', type=float, default=None)
    parser.add_argument('--unlearn_epoch', type=int, default=None)
    parser.add_argument('--unlearn_rate', type=float, default=None)
    
    # Defaults
    parser.add_argument('--forget_class', type=int, default=0, help='Class to forget')
    parser.add_argument('--num_forget', type=int, default=5000)
    
    # Train/Retrain settings
    parser.add_argument('--train_from_scratch', action='store_true')
    parser.add_argument('--retrain_from_scratch', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'])
    
    # Method-specific
    parser.add_argument('--fixed_noise_label', type=str2bool, default=True)
    parser.add_argument('--approx_different', type=str2bool, default=True)
    parser.add_argument('--soft_label', type=str, default="inf")
    parser.add_argument('--adv_eps', type=float, default=0.4)
    parser.add_argument('--adv_lambda', type=float, default=0.1)
    
    # Other
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--exps_dir', type=str, default="experiments")
    parser.add_argument('--description', type=str, default="")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(f'config/{args.dataset_name}_{args.model_name}.yaml')
    keys = ["pretrain_epoch", "pretrain_lr", "batch_size", "unlearn_epoch", "unlearn_rate"]
    for key in keys:
        if getattr(args, key) is None: setattr(args, key, config[key])
    
    # Setup
    seed_torch(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path(args.exps_dir).expanduser()
    create_dir(path)
    
    # 1. Dataset - Get General Loader first
    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name)
    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, args.num_workers)
    num_classes = 10 

    # 2. Load Models
    ckpt_path = path / "pretrained_model"
    create_dir(ckpt_path)
    model_name = args.model_name
    
    # If user wants to train from scratch, do it and exit
    if args.train_from_scratch:
        print("Training original model from scratch...")
        train_save_model(train_loader, test_loader, model_name, args.optim_name, args.pretrain_lr, args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_original_model")
        exit(0)

    # Load the Original Model
    # Note: We load a generic original model (not specific to a forget class yet)
    # Adjust this filename if your saved model has a different naming convention
    original_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_forget0_original_model.pth' 
    
    if not original_model_path.exists():
        # Fallback to try and find any original model
        print(f"Warning: {original_model_path} not found.")
        print("Looking for generic original model...")
        # Try to find a file ending in original_model.pth
        candidates = list(ckpt_path.glob("*original_model.pth"))
        if candidates:
            original_model_path = candidates[0]
            print(f"Found: {original_model_path}")
        else:
            print("No pretrained model found! Please run with --train_from_scratch first.")
            exit(1)

    ori_model = load_model(original_model_path, model_name, num_classes)
    ori_model = ori_model.to(device)

    # ==========================================
    # INTERACTIVE SECTION
    # ==========================================
    
    # 1. Show Original Accuracies
    print_class_accuracy(ori_model, test_loader, device, title="ORIGINAL MODEL ACCURACY")
    
    # 2. Ask User for Input
    print("Check the table above.")
    try:
        user_input = input(">> Enter the Class Number (0-9) you want to DELETE/FORGET: ")
        args.forget_class = int(user_input.strip())
        if args.forget_class < 0 or args.forget_class > 9:
            raise ValueError
    except:
        print("Invalid input. Defaulting to Class 0.")
        args.forget_class = 0

    print(f"\nProcessing Unlearning for Class: {args.forget_class}...\n")

    # 3. Reload Data Loaders based on the specific Forget Class
    forget_class_index = [args.forget_class]
    train_forget_loader, train_remain_loader, test_forget_loader, \
    test_remain_loader, _, _, _, _, _ = \
        get_unlearn_loader(trainset, testset, forget_class_index,
                          args.batch_size, args.num_forget, args.num_workers)

    # Setup Logging Directory
    description = f"{args.dataset_name}_{model_name}_forget{args.forget_class}"
    now = datetime.now()
    formatted_time = now.strftime("%m%d-%H:%M:%S")
    vice_description = f"{args.description}_{formatted_time}"
    
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

    # 4. Run Unlearning
    experiment_path = path / description / method_description / vice_description
    
    if args.method == "delete":
        unlearn_model = delete(ori_model, train_forget_loader,
                               args.unlearn_epoch, args.unlearn_rate,
                               logger=logger,
                               console_handler=console_handler,
                               loader_dict=loader_dict,
                               experiment_path=experiment_path,
                               soft_label=args.soft_label)
    else:
        # Fallback for other methods if you use them later
        print(f"Method {args.method} not fully hooked up in this interactive demo yet.")
        exit()

    # ==========================================
    # VERIFICATION SECTION
    # ==========================================

    # 5. Show New Accuracies
    print_class_accuracy(unlearn_model, test_loader, device, title="UNLEARNED MODEL ACCURACY")
    
    print(f"Verify that accuracy for Class {args.forget_class} has dropped.")

    # Save model
    if unlearn_model:
        torch.save(unlearn_model.state_dict(), experiment_path / "ckpt.pth")
        logger.info(f"Model saved to {experiment_path / 'ckpt.pth'}")
    
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