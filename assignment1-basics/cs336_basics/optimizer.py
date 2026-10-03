import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

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
