import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC

def entropy(p, dim=-1, keepdim=False):
    """Compute entropy"""
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    """Collect predictions from data loader"""
    probs = []
    labels = []
    model.eval()
    
    for data, target in data_loader:
        data = data.to("cuda")
        target = target.to("cuda")
        with torch.no_grad():
            output = model(data)
            prob = F.softmax(output, dim=-1)
        
        probs.append(prob.cpu().numpy())
        labels.append(target.cpu().numpy())
    
    return np.concatenate(probs), np.concatenate(labels)

def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    """Train SVM and compute accuracy"""
    # Prepare training data
    X_train = np.concatenate((shadow_train, shadow_test))
    y_train = np.concatenate((np.ones(len(shadow_train)),
                             np.zeros(len(shadow_test))))
    
    # Train SVM
    svc_model = SVC(kernel='rbf', gamma='auto')
    svc_model.fit(X_train, y_train)
    
    # Predict on target
    X_test = np.concatenate((target_train, target_test)) if target_test is not None else target_train
    y_test = np.concatenate((np.ones(len(target_train)),
                            np.zeros(len(target_test)))) if target_test is not None else np.ones(len(target_train))
    
    pred = svc_model.predict(X_test)
    accuracy = np.mean(pred == y_test)
    
    return accuracy

def SVC_MIA(shadow_train, shadow_test, target_train, target_test, model):
    """Run SVM-based membership inference attack"""
    # Collect probabilities
    shadow_train_probs, _ = collect_prob(shadow_train, model)
    shadow_test_probs, _ = collect_prob(shadow_test, model)
    target_train_probs, _ = collect_prob(target_train, model)
    
    if target_test is not None:
        target_test_probs, _ = collect_prob(target_test, model)
    else:
        target_test_probs = None
    
    # Compute entropy features
    shadow_train_entropy = entropy(torch.tensor(shadow_train_probs)).numpy()
    shadow_test_entropy = entropy(torch.tensor(shadow_test_probs)).numpy()
    target_train_entropy = entropy(torch.tensor(target_train_probs)).numpy()
    
    if target_test_probs is not None:
        target_test_entropy = entropy(torch.tensor(target_test_probs)).numpy()
    else:
        target_test_entropy = None
    
    # Accuracy on entropy
    accuracy = SVC_fit_predict(shadow_train_entropy.reshape(-1, 1),
                              shadow_test_entropy.reshape(-1, 1),
                              target_train_entropy.reshape(-1, 1),
                              target_test_entropy.reshape(-1, 1) if target_test_entropy is not None else None)
    
    return f"SVC_MIA_Accuracy: {accuracy:.4f}"