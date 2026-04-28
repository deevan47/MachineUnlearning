import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models import *
import tqdm

def optimizer_picker(optimization, param, lr):
    """Pick optimizer"""
    if optimization == 'sgd':
        return optim.SGD(param, lr=lr, momentum=0.9)
    elif optimization == 'adam':
        return optim.Adam(param, lr=lr)
    else:
        raise ValueError("Optimizer not found")

def train(model, data_loader, optimizer, epoch, tqdm_on=True):
    """One training epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if tqdm_on:
        iterator = tqdm.tqdm(data_loader)
    else:
        iterator = data_loader
    
    for batch_idx, (data, target) in enumerate(iterator):
        data = data.to("cuda")
        target = target.to("cuda")
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, data_loader, extra_class=0):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    # empty loader
    try:
        loader_len = len(data_loader)
    except Exception:
        loader_len = 0

    if loader_len == 0:
        return float('nan'), 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / loader_len if loader_len > 0 else float('nan')

    return avg_loss, accuracy

@timer
def train_save_model(train_loader, test_loader, model_name, optim_name,
                    learning_rate, num_epochs, path, description, num_classes=10, in_channels=1):
    """Train and save model"""
    
    model = get_model(model_name, num_classes, in_channels)
    model = model.to("cuda")
    print(f" {model_name} loaded")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)
    
    best_acc = 0
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    # always train fresh and save each epoch

    for epoch in tqdm.tqdm(range(num_epochs)):
        train(model=model, data_loader=train_loader, optimizer=optimizer,
              epoch=epoch, tqdm_on=False)

        train_loss, train_acc = test(model=model, data_loader=train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test(model=model, data_loader=test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Save checkpoint 
        epoch_ckpt = path / f'{description}_epoch{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
        }, epoch_ckpt)

        # save latest
        latest_ckpt = path / f'{description}_latest.pth'
        torch.save(model.state_dict(), latest_ckpt)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       path / f'{description}_best.pth')
            print(f"Best model updated at epoch {epoch+1} with acc {test_acc:.4f}")

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Plot 
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Testing Accuracy')
    
    plt.tight_layout()
    plt.savefig(path / f'{description}_training_curves.png')
    plt.close()
    
    # Load latest
    latest_path = path / f'{description}_latest.pth'
    if latest_path.exists():
        model.load_state_dict(torch.load(latest_path))
        print(f"Loaded {latest_path}")
        # save metadata 
        import json
        metadata_path = latest_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({'forgotten': []}, f)
    else:
        print(f"Latest model not found: {latest_path}")
    model = model.to("cuda")
    
    return model