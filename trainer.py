import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models import *
import tqdm

def optimizer_picker(optimization, param, lr):
    """Select optimizer"""
    if optimization == 'sgd':
        return optim.SGD(param, lr=lr, momentum=0.9)
    elif optimization == 'adam':
        return optim.Adam(param, lr=lr)
    else:
        raise ValueError("Optimizer not found")

def train(model, data_loader, optimizer, epoch, tqdm_on=True):
    """Train for one epoch"""
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
    """Test model accuracy"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    
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
    
    accuracy = correct / total
    avg_loss = loss_sum / len(data_loader)
    
    return avg_loss, accuracy

@timer
def train_save_model(train_loader, test_loader, model_name, optim_name,
                    learning_rate, num_epochs, path, description):
    """Train model from scratch and save"""
    num_classes = 10  # MNIST
    
    model = get_model(model_name, num_classes)
    model = model.to("cuda")
    print(f"Model {model_name} loaded")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)
    
    best_acc = 0
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        train(model=model, data_loader=train_loader, optimizer=optimizer,
              epoch=epoch, tqdm_on=False)
        
        train_loss, train_acc = test(model=model, data_loader=train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        test_loss, test_acc = test(model=model, data_loader=test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                      path / f'{description}.pth')
            print(f"Model saved at epoch {epoch+1} with acc {test_acc:.4f}")
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Plot results
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
    
    # Load best model
    model.load_state_dict(torch.load(path / f'{description}.pth'))
    model = model.to("cuda")
    
    return model