import copy
import torch
from torch import nn
from torchvision import datasets
import tqdm
import random
import time
from functools import wraps

from utils import note_print
from method.utils import keys, eval_opt, evaluate_model_on_all_loaders
import log_utils

def noise_label(label, num_classes, approx_different):
    """Generate noisy labels"""
    if isinstance(label, list):
        label = torch.tensor(label)
    
    if approx_different:
        noisy_label = torch.randint(0, num_classes, label.shape)
    else:
        noisy_label = torch.zeros_like(label)
        for i in range(len(label)):
            available_labels = list(range(num_classes))
            available_labels.remove(label[i].item())
            noisy_label[i] = random.choice(available_labels)
    
    return noisy_label

def timer(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n{func.__name__} took {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper

@timer
def random_label(ori_model, train_forget_loader, num_classes,
                  unlearn_epoch, unlearn_rate,
                  fixed_noise_label,
                  logger, console_handler,
                  loader_dict, experiment_path,
                  approx_different=True, eval_opt=eval_opt, disable_bn=False):
    
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    train_forget_loader_randlabel = copy.deepcopy(train_forget_loader)
    
    if approx_different:
        note_print("Using approximately different noise labels")
    else:
        note_print("Using strictly different noise labels")
    
    if fixed_noise_label:
        if isinstance(train_forget_loader_randlabel.dataset, datasets.MNIST):
            targets = train_forget_loader_randlabel.dataset.targets
            targets = noise_label(targets, num_classes, approx_different)
            train_forget_loader_randlabel.dataset.targets = targets
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate,
                               momentum=0.9)
    
    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }
    
    log_utils.enable_console_logging(logger, console_handler, False)
    
    for epoch in tqdm.trange(unlearn_epoch):
        for x, y in train_forget_loader_randlabel:
            if not fixed_noise_label:
                y = noise_label(y, num_classes, approx_different)
            
            x, y = x.to("cuda"), y.to("cuda")
            
            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            
            unlearn_model.zero_grad()
            optimizer.zero_grad()
            
            logits = unlearn_model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
        
        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")
        
        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict,
                                                     eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])
    
    log_utils.enable_console_logging(logger, console_handler, True)
    
    return unlearn_model