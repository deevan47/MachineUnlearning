import torch
import torch.nn as nn

class AttackBase:
    """Base class for adversarial attacks"""
    
    def __init__(self, model, eps, norm, random_start, device):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.random_start = random_start
        self.device = device
    
    def perturb(self, x, y, target_y=None, model=None, device="cuda"):
        """Generate adversarial perturbation"""
        raise NotImplementedError

class FGSM(AttackBase):
    """Fast Gradient Sign Method"""
    
    def __init__(self, model, eps, norm=True, random_start=False, device="cuda"):
        super().__init__(model, eps, norm, random_start, device)
    
    def perturb(self, x, y, target_y=None, model=None, device="cuda"):
        """Generate FGSM perturbation"""
        if model is None:
            model = self.model
        
        x.requires_grad = True
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        grad = x.grad.data
        
        if self.norm:
            # Linf norm
            perturbation = self.eps * torch.sign(grad)
        else:
            # L2 norm
            perturbation = self.eps * grad / (grad.norm(dim=(1, 2, 3), keepdim=True) + 1e-8)
        
        x_adv = x.detach() + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv

class LinfPGD(AttackBase):
    """Projected Gradient Descent (Linf)"""
    
    def __init__(self, model, eps, step_size, num_steps, norm=True,
                 random_start=False, device="cuda"):
        super().__init__(model, eps, norm, random_start, device)
        self.step_size = step_size
        self.num_steps = num_steps
    
    def perturb(self, x, y, target_y=None, model=None, device="cuda"):
        """Generate PGD perturbation"""
        if model is None:
            model = self.model
        
        x_adv = x.clone().detach()
        
        if self.random_start:
            noise = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps).to(device)
            x_adv = (x + noise).clamp(0, 1)
        
        for _ in range(self.num_steps):
            x_adv.requires_grad = True
            output = model(x_adv)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            grad = x_adv.grad.data
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            
            # Project onto Linf ball
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = (x + delta).clamp(0, 1)
        
        return x_adv.detach()