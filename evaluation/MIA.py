import numpy as np
import torch
import torch.nn.functional as F

class black_box_benchmarks(object):
    """Black-box membership inference attack"""
    
    def __init__(self, shadow_train_performance, shadow_test_performance,
                target_train_performance, target_test_performance, num_classes):
        """
        Each input contains model predictions and ground-truth labels
        """
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        # Compute correctness
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)
        
        # Compute confidence
        self.s_tr_conf = np.take_along_axis(self.s_tr_outputs, 
                                            self.s_tr_labels[:, None], axis=1)
        self.s_te_conf = np.take_along_axis(self.s_te_outputs,
                                            self.s_te_labels[:, None], axis=1)
        self.t_tr_conf = np.take_along_axis(self.t_tr_outputs,
                                            self.t_tr_labels[:, None], axis=1)
        self.t_te_conf = np.take_along_axis(self.t_te_outputs,
                                            self.t_te_labels[:, None], axis=1)
        
        # Compute entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
    
    def _log_value(self, probs, eps=1e-30):
        """Safe log function"""
        return -np.log(np.maximum(probs, eps))
    
    def _entr_comp(self, probs):
        """Compute entropy"""
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        """Find optimal threshold"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_gap = 0, 0
        for th in np.arange(np.min(value_list), np.max(value_list), 0.001):
            gap = np.mean(tr_values >= th) - np.mean(te_values >= th)
            if gap > max_gap:
                max_gap = gap
                thre = th
        return thre
    
    def _mem_inf_via_corr(self):
        """Membership inference via correctness"""
        thre = self._thre_setting(self.s_tr_corr, self.s_te_corr)
        t_tr_pred = (self.t_tr_corr >= thre).astype(int)
        t_te_pred = (self.t_te_corr >= thre).astype(int)
        
        mem_inf_acc = np.mean(np.concatenate(
            (t_tr_pred == 1, t_te_pred == 0)))
        return mem_inf_acc
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, 
                     t_tr_values, t_te_values):
        """Membership inference via threshold"""
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_pred = (t_tr_values >= thre).astype(int)
        t_te_pred = (t_te_values >= thre).astype(int)
        
        mem_inf_acc = np.mean(np.concatenate(
            (t_tr_pred == 1, t_te_pred == 0)))
        return mem_inf_acc
    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        """Run all membership inference attacks"""
        results = {}
        
        results['correctness'] = self._mem_inf_via_corr()
        results['confidence'] = self._mem_inf_thre(
            'confidence', self.s_tr_conf, self.s_te_conf,
            self.t_tr_conf, self.t_te_conf)
        results['entropy'] = self._mem_inf_thre(
            'entropy', self.s_tr_entr, self.s_te_entr,
            self.t_tr_entr, self.t_te_entr)
        
        return results

def collect_performance(data_loader, model, device):
    """Collect model predictions"""
    probs = []
    labels = []
    model.eval()
    
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
            prob = F.softmax(output, dim=-1)
        
        probs.append(prob)
        labels.append(target)
    
    return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()

def MIA(retain_loader_train, retain_loader_test, forget_loader,
         test_loader, model, device):
    """Run membership inference attack"""
    shadow_train_performance = collect_performance(retain_loader_train, model, device)
    shadow_test_performance = collect_performance(test_loader, model, device)
    target_train_performance = collect_performance(retain_loader_test, model, device)
    target_test_performance = collect_performance(forget_loader, model, device)
    
    BBB = black_box_benchmarks(
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes=10,
    )
    
    return BBB._mem_inf_benchmarks()