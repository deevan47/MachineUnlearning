import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset
import os


class AdvancedMetrics:
    """Advanced metrics for evaluating machine unlearning - t-SNE and Confusion Matrix"""
    
    def __init__(self, model_before, model_after, test_forget_loader, test_remain_loader, 
                 forgotten_classes, previously_forgotten_classes, class_names, device, save_dir):
        self.model_before = model_before
        self.model_after = model_after
        self.test_forget_loader = test_forget_loader
        self.test_remain_loader = test_remain_loader
        self.forgotten_classes = set(forgotten_classes)  # Currently being forgotten
        self.previously_forgotten_classes = set(previously_forgotten_classes)  # Already forgotten
        self.all_forgotten = self.forgotten_classes | self.previously_forgotten_classes
        self.retained_classes = set(range(len(class_names))) - self.all_forgotten
        self.class_names = class_names
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def get_features_and_labels(self, model, data_loader, filter_classes=None):
        """Extract features and labels from a data loader
        
        Args:
            model: Model to extract features from
            data_loader: DataLoader for data
            filter_classes: Set of classes to keep (removes others)
        """
        model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                
                # Extract features from layer4 (before classification head)
                features_list = []
                def hook_fn(module, input, output):
                    features_list.append(output.detach())
                
                hook = model.layer4[-1].register_forward_hook(hook_fn)
                _ = model(data)
                hook.remove()
                
                if features_list:
                    features = features_list[0]
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.numpy())
        
        if all_features:
            features = np.concatenate(all_features, axis=0)
            features = features.reshape(features.shape[0], -1)
            labels = np.concatenate(all_labels, axis=0)
        else:
            features = np.array([])
            labels = np.array([])
        
        # Filter to only keep specified classes
        if filter_classes is not None:
            mask = np.isin(labels, list(filter_classes))
            features = features[mask]
            labels = labels[mask]
        
        return features, labels
    
    def plot_tsne_comparison(self):
        """Plot t-SNE comparison showing cumulative unlearning
        
        BEFORE: All classes currently known by model (excluding previously forgotten)
        AFTER: Only retained classes (excluding all forgotten: previous + current)
        """
        print("Computing t-SNE for BEFORE state...")
        
        # BEFORE: Combine forgotten and retained test data, but EXCLUDE previously forgotten
        classes_to_show_before = self.retained_classes | self.forgotten_classes
        
        try:
            combined_dataset = ConcatDataset([self.test_forget_loader.dataset, 
                                             self.test_remain_loader.dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False, num_workers=0)
            features_before, labels_before = self.get_features_and_labels(
                self.model_before, combined_loader, filter_classes=classes_to_show_before
            )
        except:
            # Fallback: manually concatenate
            features_forget, labels_forget = self.get_features_and_labels(
                self.model_before, self.test_forget_loader, filter_classes=classes_to_show_before
            )
            features_retain, labels_retain = self.get_features_and_labels(
                self.model_before, self.test_remain_loader, filter_classes=classes_to_show_before
            )
            if len(features_forget) > 0 and len(features_retain) > 0:
                features_before = np.concatenate([features_forget, features_retain], axis=0)
                labels_before = np.concatenate([labels_forget, labels_retain], axis=0)
            elif len(features_retain) > 0:
                features_before = features_retain
                labels_before = labels_retain
            else:
                features_before = features_forget
                labels_before = labels_forget
        
        # AFTER: Only retained classes (no previously forgotten, no currently forgotten)
        print("Computing t-SNE for AFTER state...")
        features_after, labels_after = self.get_features_and_labels(
            self.model_after, self.test_remain_loader, filter_classes=self.retained_classes
        )
        
        # Run t-SNE
        print("Running t-SNE dimensionality reduction...")
        if len(features_before) > 0:
            tsne_before = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embedding_before = tsne_before.fit_transform(features_before)
        else:
            embedding_before = None
        
        if len(features_after) > 0:
            tsne_after = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embedding_after = tsne_after.fit_transform(features_after)
        else:
            embedding_after = None
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count classes in each plot
        if embedding_before is not None:
            unique_before = np.unique(labels_before)
        else:
            unique_before = []
        
        if embedding_after is not None:
            unique_after = np.unique(labels_after)
        else:
            unique_after = []
        
        # BEFORE plot - show all classes currently in model (except previously forgotten)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        
        if embedding_before is not None:
            for class_id in range(len(self.class_names)):
                mask = labels_before == class_id
                if np.any(mask):
                    if class_id in self.forgotten_classes:
                        # Currently being forgotten - mark with X
                        ax1.scatter(embedding_before[mask, 0], embedding_before[mask, 1], 
                                   c=[colors[class_id]], label=f'Class {class_id} (to forget)', 
                                   marker='x', s=150, alpha=0.8, linewidths=2)
                    else:
                        # Retained
                        ax1.scatter(embedding_before[mask, 0], embedding_before[mask, 1], 
                                   c=[colors[class_id]], label=f'Class {class_id}', 
                                   marker='o', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax1.set_title(f'BEFORE Unlearning\n({len(unique_before)} Classes)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('t-SNE 1', fontsize=10)
        ax1.set_ylabel('t-SNE 2', fontsize=10)
        ax1.legend(loc='best', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # AFTER plot - show only retained classes
        if embedding_after is not None:
            for class_id in sorted(self.retained_classes):
                if class_id in unique_after:
                    mask = labels_after == class_id
                    ax2.scatter(embedding_after[mask, 0], embedding_after[mask, 1], 
                               c=[colors[class_id]], label=f'Class {class_id}', 
                               marker='o', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax2.set_title(f'AFTER Unlearning\n({len(unique_after)} Retained Classes)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('t-SNE 1', fontsize=10)
        ax2.set_ylabel('t-SNE 2', fontsize=10)
        ax2.legend(loc='best', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'tsne_plot.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE plot saved: {save_path}")
        print(f"  BEFORE: {len(unique_before)} classes (excluding {len(self.previously_forgotten_classes)} already forgotten)")
        print(f"  AFTER: {len(unique_after)} classes (forgotten removed)\n")
    
    def plot_confusion_matrix_forgotten(self):
        """Plot confusion matrix for currently forgotten classes showing TP, TN, FP, FN labels"""
        
        # Get predictions on forgotten data
        predictions, true_labels = self.get_predictions_on_classes(
            self.model_after, self.test_forget_loader, self.forgotten_classes
        )
        
        if len(predictions) == 0:
            print("No data for forgotten classes")
            return
        
        # Compute confusion matrix for forgotten classes only
        cm = sklearn_confusion_matrix(true_labels, predictions, labels=sorted(self.forgotten_classes))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', cbar=True,
                    xticklabels=[f'Pred {c}' for c in sorted(self.forgotten_classes)],
                    yticklabels=[f'True {c}' for c in sorted(self.forgotten_classes)],
                    ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix: Currently Forgotten Classes\n(Model predictions should be ~0%)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
        
        # Adding metric annotations
        total = cm.sum()
        correct = np.trace(cm)
        accuracy = 100 * correct / total if total > 0 else 0
        
        # Calculate TP, TN, FP, FN for the forgotten classes
        tp = np.trace(cm)  # Diagonal sum
        fp = cm.sum(axis=0) - np.diag(cm)  # False positives per column
        fn = cm.sum(axis=1) - np.diag(cm)  # False negatives per row
        
        
        metrics_text = (f"Accuracy: {accuracy:.1f}%\n"
                       f"Total Samples: {int(total)}\n"
                       f"Correct: {int(correct)}\n"
                       f"\nMetrics:\n"
                       f"TP (Diagonal): {int(tp)}\n"
                       f"FP (False Pos): {int(fp.sum())}\n"
                       f"FN (False Neg): {int(fn.sum())}")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {save_path}")
        print(f"  Accuracy on forgotten data: {accuracy:.1f}% (good = ~0%)")
        print(f"  TP: {int(tp)}, FP: {int(fp.sum())}, FN: {int(fn.sum())}\n")
        
        return cm
    
    def get_predictions_on_classes(self, model, data_loader, target_classes):
        """Get predictions from model on specific classes"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Filter only  target classes
                mask = np.isin(labels.numpy(), list(target_classes))
                all_preds.append(predicted[mask].cpu().numpy())
                all_labels.append(labels[mask].numpy())
        
        if all_preds:
            predictions = np.concatenate(all_preds, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        else:
            predictions = np.array([])
            labels = np.array([])
        
        return predictions, labels
