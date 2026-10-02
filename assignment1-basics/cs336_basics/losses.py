import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits.shape = [batch_size, seq_len, vocab_size], targets.shape = [batch_size, seq_len]
    stable_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    
    loss = torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True)) - torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1))
    average_loss = torch.mean(loss)
    # return a scalar tensor
    return average_loss