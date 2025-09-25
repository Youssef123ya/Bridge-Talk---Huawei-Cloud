import torch
import torch.optim as optim
from typing import Dict, Any, Iterator
import math

class AdamW(optim.Optimizer):
    """AdamW optimizer with weight decay decoupling"""

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 amsgrad: bool = False):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class RAdam(optim.Optimizer):
    """Rectified Adam optimizer"""

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['step'] += 1
                buffered = [[state['step'], exp_avg, exp_avg_sq]]

                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                    ) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p.data.add_(exp_avg, alpha=-step_size)

        return loss

class Lookahead(optim.Optimizer):
    """Lookahead optimizer wrapper"""

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0

        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = p.data.clone()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_count': self.step_count
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_count = state_dict['step_count']

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_count += 1

        if self.step_count % self.k == 0:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    slow_weight = self.slow_weights[p]
                    slow_weight.add_(p.data - slow_weight, alpha=self.alpha)
                    p.data.copy_(slow_weight)

        return loss

class LAMB(optim.Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training (LAMB)"""

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.01,
                 bias_correction: bool = True):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                if group['weight_decay'] > 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    update = update / bias_correction1
                    update = update / math.sqrt(bias_correction2)

                trust_ratio = 1.0
                w_norm = p.data.norm()
                g_norm = update.norm()

                if w_norm > 0 and g_norm > 0:
                    trust_ratio = w_norm / g_norm

                step_size = group['lr'] * trust_ratio
                p.data.add_(update, alpha=-step_size)

        return loss

def get_optimizer(optimizer_name: str,
                 parameters: Iterator,
                 **kwargs) -> optim.Optimizer:
    """Factory function to get optimizer by name"""

    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        return optim.Adam(parameters, **kwargs)

    elif optimizer_name == 'adamw':
        return AdamW(parameters, **kwargs)

    elif optimizer_name == 'radam':
        return RAdam(parameters, **kwargs)

    elif optimizer_name == 'sgd':
        return optim.SGD(parameters, **kwargs)

    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, **kwargs)

    elif optimizer_name == 'lamb':
        return LAMB(parameters, **kwargs)

    elif optimizer_name == 'lookahead_adam':
        base_optimizer = optim.Adam(parameters, **kwargs)
        return Lookahead(base_optimizer)

    elif optimizer_name == 'lookahead_sgd':
        base_optimizer = optim.SGD(parameters, **kwargs)
        return Lookahead(base_optimizer)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(scheduler_name: str,
                 optimizer: optim.Optimizer,
                 **kwargs) -> optim.lr_scheduler._LRScheduler:
    """Factory function to get learning rate scheduler by name"""

    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)

    elif scheduler_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)

    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    elif scheduler_name == 'cosine_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

    elif scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    elif scheduler_name == 'cyclic':
        return optim.lr_scheduler.CyclicLR(optimizer, **kwargs)

    elif scheduler_name == 'one_cycle':
        return optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
