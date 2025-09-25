"""
Custom Optimizers Module for Arabic Sign Language Recognition
Provides optimizer configurations and custom optimizers optimized for sign language training
"""

import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from collections import defaultdict


class RAdam(torch.optim.Optimizer):
    """
    Rectified Adam (RAdam) optimizer
    Reference: Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond"
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                if N_sma >= 5:
                    if group['weight_decay'] > 0:
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                elif step_size > 0:
                    if group['weight_decay'] > 0:
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

        return loss


class AdaBound(torch.optim.Optimizer):
    """
    AdaBound optimizer that combines benefits of adaptive methods and SGD
    Reference: Luo et al. "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)

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
                if grad.is_sparse:
                    raise RuntimeError('AdaBound does not support sparse gradients')
                
                amsbound = group['amsbound']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                scaled_lr = group['lr'] / bias_correction1

                final_lr = group['final_lr'] * group['lr'] / group['lr']
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))

                step_size = torch.full_like(denom, scaled_lr)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(step_size, alpha=-1)

        return loss


class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimizer wrapper
    Reference: Zhang et al. "Lookahead Optimizer: k steps forward, 1 step back"
    """
    
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group, fast_params):
        if "slow_params" not in group:
            group["slow_params"] = []
            for param in group["params"]:
                group["slow_params"].append(param.clone())
        
        for fast_param, slow_param in zip(fast_params, group["slow_params"]):
            slow_param.data.add_(fast_param.data - slow_param.data, alpha=self.alpha)
            fast_param.data.copy_(slow_param.data)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group, group["params"])
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer
    Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    dtype=torch.float32
               )
        return norm.to(shared_device)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def state_dict(self):
        return super().state_dict()


# Optimizer configurations for different scenarios
def get_optimizer_config(scenario: str = 'default') -> Dict[str, Any]:
    """
    Get optimizer configuration for different training scenarios
    
    Args:
        scenario: Training scenario ('default', 'fast', 'stable', 'transfer', 'fine_tune')
        
    Returns:
        Optimizer configuration dictionary
    """
    configs = {
        'default': {
            'optimizer': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'fast': {
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 0.0001,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'stable': {
            'optimizer': 'sgd',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True
        },
        'transfer': {
            'optimizer': 'adamw',
            'lr': 0.0001,  # Lower learning rate for transfer learning
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'fine_tune': {
            'optimizer': 'sgd',
            'lr': 0.001,  # Very low learning rate for fine-tuning
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True
        }
    }
    
    return configs.get(scenario, configs['default'])


def get_scheduler_config(scenario: str = 'default', epochs: int = 100) -> Dict[str, Any]:
    """
    Get learning rate scheduler configuration
    
    Args:
        scenario: Training scenario
        epochs: Total number of epochs
        
    Returns:
        Scheduler configuration dictionary
    """
    configs = {
        'default': {
            'scheduler': 'cosine',
            'T_max': epochs,
            'eta_min': 1e-6
        },
        'step': {
            'scheduler': 'step',
            'step_size': epochs // 3,
            'gamma': 0.1
        },
        'plateau': {
            'scheduler': 'plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-6
        },
        'warmup': {
            'scheduler': 'warmup_cosine',
            'warmup_epochs': min(10, epochs // 10),
            'T_max': epochs,
            'eta_min': 1e-6
        }
    }
    
    return configs.get(scenario, configs['default'])


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config.pop('optimizer', 'adamw').lower()
    
    # Get model parameters
    params = model.parameters()
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = Adam(params, **config)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(params, **config)
    elif optimizer_name == 'sgd':
        optimizer = SGD(params, **config)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(params, **config)
    elif optimizer_name == 'radam':
        optimizer = RAdam(params, **config)
    elif optimizer_name == 'adabound':
        optimizer = AdaBound(params, **config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[object]:
    """
    Create learning rate scheduler from configuration
    
    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = config.pop('scheduler', None)
    
    if scheduler_name is None:
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, **config)
    elif scheduler_name == 'multistep':
        scheduler = MultiStepLR(optimizer, **config)
    elif scheduler_name == 'exponential':
        scheduler = ExponentialLR(optimizer, **config)
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, **config)
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, **config)
    elif scheduler_name == 'warmup_cosine':
        warmup_epochs = config.pop('warmup_epochs', 10)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_epochs, **config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm restarts and warmup
    """
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, 
                 warmup_steps=0, gamma=1., last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * \
                    (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) / \
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# Optimizer factory with different configurations
def create_optimizer_with_config(model: nn.Module, config_name: str = 'default', 
                               custom_config: Optional[Dict] = None) -> Tuple[torch.optim.Optimizer, Optional[object]]:
    """
    Create optimizer and scheduler with predefined configurations
    
    Args:
        model: PyTorch model
        config_name: Configuration name
        custom_config: Custom configuration to override defaults
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get base configuration
    opt_config = get_optimizer_config(config_name)
    sched_config = get_scheduler_config(config_name)
    
    # Override with custom config if provided
    if custom_config:
        opt_config.update(custom_config.get('optimizer', {}))
        sched_config.update(custom_config.get('scheduler', {}))
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, opt_config.copy())
    scheduler = create_scheduler(optimizer, sched_config.copy())
    
    return optimizer, scheduler


# Optimizer registry
OPTIMIZERS = {
    'adam': Adam,
    'adamw': AdamW,
    'sgd': SGD,
    'rmsprop': RMSprop,
    'radam': RAdam,
    'adabound': AdaBound,
}

SCHEDULERS = {
    'step': StepLR,
    'multistep': MultiStepLR,
    'exponential': ExponentialLR,
    'cosine': CosineAnnealingLR,
    'plateau': ReduceLROnPlateau,
    'warmup_cosine': CosineAnnealingWarmupRestarts,
}


def list_optimizers() -> List[str]:
    """Get list of available optimizers"""
    return list(OPTIMIZERS.keys())


def list_schedulers() -> List[str]:
    """Get list of available schedulers"""
    return list(SCHEDULERS.keys())


def create_differential_lr_optimizer(model: nn.Module, base_lr: float = 0.001, 
                                   layer_lr_decay: float = 0.9) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for different layers
    Useful for transfer learning where early layers need lower learning rates
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate for final layers
        layer_lr_decay: Decay factor for earlier layers
        
    Returns:
        Optimizer with differential learning rates
    """
    param_groups = []
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    total_layers = len(named_params)
    
    # Group parameters by layer depth
    for i, (name, param) in enumerate(named_params):
        if param.requires_grad:
            # Calculate learning rate based on layer depth
            lr_mult = layer_lr_decay ** (total_layers - i - 1)
            layer_lr = base_lr * lr_mult
            
            param_groups.append({
                'params': [param],
                'lr': layer_lr,
                'name': name
            })
    
    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)