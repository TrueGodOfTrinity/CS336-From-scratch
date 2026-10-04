import torch
import numpy as np

def data_loader(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:

    start_indices = np.random.randint(0, len(dataset) - context_length,  batch_size)
    inputs = []
    outputs = []
    for index in start_indices:
        input_data = dataset[index: index + context_length]
        output_data = dataset[index + 1: index + context_length + 1]
        outputs.append(output_data)
        inputs.append(input_data)

    inputs = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
    outputs = torch.tensor(np.stack(outputs), dtype=torch.long, device=device)

    return (inputs, outputs)


    
