import os
import random
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np
import time
from functools import wraps
from pathlib import Path


def note_print(*args, **kwargs):
    """Print text in red."""
    print('\033[91m' + " ".join(map(str, args)) + '\033[0m', **kwargs)

def timer(func):
    """Measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n{func.__name__} took {elapsed:.2f} seconds\n")
        return result
    return wrapper

def seed_torch(seed=2022):
    """Set random seed."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dir(dir_name):
    """Create directory if missing."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created {dir_name}")

class Identity:
    """Identity transformation (no augmentation)"""
    def __call__(self, x):
        return x
    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_transforms(dataset_name, model_name, wo_dataaug=False):
    """Return transforms."""
    if dataset_name == "mnist":
        # MNIST 
        if wo_dataaug:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "cifar10":
        # CIFAR-10 
        resize_transform = transforms.Identity() if "my" in model_name or model_name == "vgg16" \
                        else transforms.Resize((224, 224))
        if wo_dataaug:
            transform_train = transforms.Compose([
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
        
        transform_test = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return transform_train, transform_test

def get_dataset(dataset_name, transform_train, transform_test, 
                path=Path("~/data").expanduser()):
    """Load dataset"""
    if dataset_name == "mnist":
        trainset = datasets.MNIST(str(path), train=True, download=True,
                                 transform=transform_train)
        testset = datasets.MNIST(str(path), train=False, download=True,
                                transform=transform_test)
    elif dataset_name == "cifar10":
        trainset = datasets.CIFAR10(str(path), train=True, download=True,
                               transform=transform_train)
        testset = datasets.CIFAR10(str(path), train=False, download=True,
                              transform=transform_test)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return trainset, testset

def get_dataloader(trainset, testset, batch_size, num_workers, shuffle=True):
    """Create DataLoaders"""
    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def split_class_data(dataset, forget_class_index, num_forget):
    """
    Split dataset into forget/remain indices
    """
    targets = np.array(dataset.targets)
    
    # Find indices
    forget_mask = np.isin(targets, forget_class_index)
    forget_class_indices = np.where(forget_mask)[0].tolist()
    
    # remain indices
    remain_mask = ~forget_mask
    remain_class_indices = np.where(remain_mask)[0].tolist()
    
    # leftover forget indices
    train_forget_index = random.sample(forget_class_indices, 
                                       min(num_forget, len(forget_class_indices)))
    
    # Remaining forget indices (for constructing remain set)
    class_remain_index = [i for i in forget_class_indices if i not in train_forget_index]
    
    # final remain set
    train_remain_index = remain_class_indices + class_remain_index
    
    return train_forget_index, train_remain_index, class_remain_index

def get_unlearn_loader(trainset, testset, forget_class_index, batch_size,
                       num_forget, num_workers, repair_num_ratio=0.01):
    """
Create loaders for unlearning
    """
    train_forget_index, train_remain_index, class_remain_index = \
        split_class_data(trainset, forget_class_index, num_forget=num_forget)
    
    test_forget_index, test_remain_index, _ = \
        split_class_data(testset, forget_class_index, num_forget=len(testset))
    
    # repair subset
    repair_class_index = random.sample(class_remain_index,
                                       int(repair_num_ratio * len(class_remain_index)))
    
    #  samplers
    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)
    repair_class_sampler = SubsetRandomSampler(repair_class_index)
    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)
    
    #  loaders
    train_forget_loader = DataLoader(trainset, batch_size=batch_size,
                                    sampler=train_forget_sampler,
                                    num_workers=num_workers)
    train_remain_loader = DataLoader(trainset, batch_size=batch_size,
                                    sampler=train_remain_sampler,
                                    num_workers=num_workers)
    repair_class_loader = DataLoader(trainset, batch_size=batch_size,
                                    sampler=repair_class_sampler,
                                    num_workers=num_workers)
    test_forget_loader = DataLoader(testset, batch_size=batch_size,
                                   sampler=test_forget_sampler,
                                   num_workers=num_workers)
    test_remain_loader = DataLoader(testset, batch_size=batch_size,
                                   sampler=test_remain_sampler,
                                   num_workers=num_workers)
    
    return (train_forget_loader, train_remain_loader, test_forget_loader,
            test_remain_loader, repair_class_loader, train_forget_index,
            train_remain_index, test_forget_index, test_remain_index)

def inf_generator(iterable):
    """Infinite iterator"""
    while True:
        for item in iterable:
            yield item