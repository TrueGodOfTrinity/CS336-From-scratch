import torch
import torch.nn as nn
import math

class Linear(nn.Module):


    def __init__(self,
                in_features: int, 
                out_features: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(
                        self.out_features, 
                        self.in_features, 
                        device=self.device,
                        dtype = self.dtype,
            )
        )
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight,
                            mean=0.0, 
                            std=std,
                            a=-3 * std,
                            b=3 * std,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x @ self.weight.T
    
    
class Embedding(nn.Module):
    
    """
    tokenID -> vector space(d_model)
    """
    
    def __init__(self, 
                num_embeddings: int,
                embedding_dim: int, 
                device: torch.device | None = None,
                dtype:torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings    # vocabulary size
        self.embedding_dim = embedding_dim      # d_model
        self.device = device 
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )
        nn.init.trunc_normal_(self.weight,
                            mean=0.0, 
                            std=1.0,
                            a=-3 * 1.0,
                            b=3 * 1.0,
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.weight[token_ids]