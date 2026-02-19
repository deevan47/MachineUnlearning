import copy
import torch
from torch import nn
import tqdm
import time
from functools import wraps

# USE ABSOLUTE IMPORTS
from method.utils import keys, eval_opt, evaluate_model_on_all_loaders
import log_utils

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
def finetune(ori_model, train_remain_loader,
             unlearn_epoch, unlearn_rate,
             logger, console_handler,
             loader_dict, experiment_path,
             eval_opt=eval_opt, disable_bn=False):
    
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    
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
        for x, y in train_remain_loader:
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