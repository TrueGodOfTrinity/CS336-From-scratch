import os
import typing
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    state = {"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}
    torch.save(state, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])

    return state["iteration"]