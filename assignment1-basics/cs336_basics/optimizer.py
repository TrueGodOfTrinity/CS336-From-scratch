import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

def cosine_learning_rate_schedule(t , lr_max, lr_min, t_w, t_c):

    # t_w: number of warming up iteration
    # t_c: the last number of cosine annealing

    # warm up
    if t < t_w:  
        lr = t * lr_max /t_w
    # cosine annealing
    if t >= t_w and t <= t_c:
        lr = lr_min + 0.5 * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (lr_max - lr_min)
    # post annealing
    if t > t_c:
        lr = lr_min

    return lr


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):

    square_l2_norm_grad_ = torch.zeros(())
    _params = list()
    for param in params:
        _params.append(param)
        if param.grad is None:
            continue
        grad_sum = torch.sum(param.grad ** 2)
        square_l2_norm_grad_ += grad_sum
    l2_norm_grad = torch.sqrt(square_l2_norm_grad_)

    for param in _params:
        if param.grad is None or l2_norm_grad <= max_l2_norm:
            continue
        else:
            param.grad *= max_l2_norm / (l2_norm_grad + 1e-6)

    
class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, betas: tuple, eps, weight_decay):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2:
            raise ValueError(f"Invalid betas quantity")
        if betas[0] < 0 or betas[1] < 0 or betas[0] >= 1 or betas[1] >= 1:
            raise ValueError(f"Invalid betas")
        if eps <= 0:
            raise ValueError(f"Invalid eps")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay")

        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # similar to the SGD optimizer in the pdf 
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                t = state.get("t", 1) # get the iteration number, if it isn't existed, get 1
                grad = param.grad.data
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t) # Compute adjusted learning rate for iteration t
                param.data *= 1 - lr * weight_decay # apply weight decay
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad # update the first moment estimate
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad * grad # update the second moment estimate 
                param.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1
        return loss


