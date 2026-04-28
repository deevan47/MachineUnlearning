import matplotlib.pyplot as plt
from trainer import test

keys = ["train_forget", "train_remain", "test_forget", "test_remain"]

eval_opt = {
    "train_forget": True,
    "train_remain": True,
    "test_forget": True,
    "test_remain": True,
}

def evaluate_model_on_all_loaders(model, loader_dict, eval_option, logger,
                                 extra_class=0):
    """Evaluate model on all data splits"""
    keys = ["train_forget", "train_remain", "test_forget", "test_remain"]
    current_accs_dict = {key: float("nan") for key in keys}
    
    for key in keys:
        if eval_option[key]:
            _, acc = test(model, loader_dict[key], extra_class)
            current_accs_dict[key] = acc

    def fmt(x):
        try:
            if x != x:  # nan check
                return 'N/A'
            return f"{x*100:.2f}%"
        except Exception:
            return 'N/A'

    logger.info(
        f"train_forget_acc: {fmt(current_accs_dict['train_forget'])}, "
        f"train_remain_acc: {fmt(current_accs_dict['train_remain'])}, "
        f"test_forget_acc: {fmt(current_accs_dict['test_forget'])}, "
        f"test_remain_acc: {fmt(current_accs_dict['test_remain'])}"
    )

    return current_accs_dict

def plot_unlearn_remain_acc_figure(epoch, accs_dict, experiment_path,
                                  plot_type="plot"):
    """Plot accuracy curves"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(accs_dict['train_forget'], label='Train Forget', marker='o')
    plt.plot(accs_dict['train_remain'], label='Train Remain', marker='s')
    plt.plot(accs_dict['test_forget'], label='Test Forget', marker='^')
    plt.plot(accs_dict['test_remain'], label='Test Remain', marker='d')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Unlearning Accuracy - Epoch {epoch}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(experiment_path / 'unlearn_acc_forget_remain.png')
    plt.close()