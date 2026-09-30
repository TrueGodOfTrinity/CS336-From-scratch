import torch
import torch.nn as nn
import math


def silu(x: torch.Tensor) -> torch.Tensor:
    
    return x / (1 + torch.exp( - x))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    max_values = x.max(dim=dim, keepdim=True).values
    shifted_x = x - max_values

    return torch.exp(shifted_x) / torch.sum(torch.exp(shifted_x), dim=dim, keepdim=True)


def scaled_dot_product_attention(queries: torch.Tensor, 
                                keys: torch.Tensor, 
                                values: torch.Tensor, 
                                mask:torch.Tensor | None = None,
) -> torch.Tensor:
    # q,k.shape = [batch_size, ..., seq_len, d_k]; v.shape = [batch_size, ..., seq_len, d_v]
    # mask.shape = [seq_len, seq_len]
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention = softmax(scores, dim=-1) @ values
    return attention


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
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    
    
    def __init__(self, 
                d_model: int,
                eps: float = 1e-5,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,  
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device 
        self.dtype = dtype 
        
        self.weight = nn.Parameter(
            torch.empty(
                d_model,
                device=self.device,
                dtype=self.dtype
            )
        )
        nn.init.ones_(self.weight)
                        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape() = [batch_size, seq_len, d_model]
        original_dtype = x.dtype
        x = x.to(torch.float32)
        
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x_rms = torch.sqrt(mean_square + self.eps)
        x_normalized = x / x_rms
        output = x_normalized * self.weight 
        
        return  output.to(original_dtype)
        
        
class SwiGLU(nn.Module):
    
    
    def __init__(self, 
                d_model: int,
                d_ff: int,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,             
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        # initialize the layers. Incidentally, the weights are included.
        self.w1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call submodules directly so PyTorch runs forward through nn.Module's call mechanism, 
        preserving registered hooks.
        """
        glu = silu(self.w1(x)) * self.w3(x)    
        result = self.w2(glu)                  
        return result 


class RotaryPositionalEmbedding(nn.Module):


    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k # dimention of key and query vectors
        self.max_seq_len = max_seq_len
        self.device = device 

        freqs = torch.tensor(
            [1 / theta ** ((2 * k - 2) / d_k) for k in range(1, d_k // 2 + 1)],
            dtype=torch.float32,
            device=self.device,
        )
        positions = torch.tensor(
            [position for position in range(self.max_seq_len)],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)
        # broadcasting: element-wise multiplication, be aware of the shapes!
        angles = positions * freqs
        # angle.shape = [max_seq_len, d_K // 2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer(
            "cos",
            cos,
            persistent=False
        )
        self.register_buffer(
            "sin",
            sin,
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, seq_len, d_k]; token_position.shape = [batch_size, seq_len]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_x_even = x_even * self.cos[token_position] - x_odd * self.sin[token_position]
        rotated_x_odd = x_even * self.sin[token_position] + x_odd * self.cos[token_position]
        # [..., d_k/2] -> [..., d_k/2, 2] -> [..., d_k]
        return torch.stack((rotated_x_even, rotated_x_odd), dim=-1).flatten(start_dim=-2)
        

class CausalMultiHeadSelfAttentionWithoutRoPE(nn.Module):

    def __init__(self, 
            d_model: int,
            num_heads: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_o = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, seq_len, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.reshape(*q.shape[:-1], self.num_heads, self.d_model // self.num_heads) 
        q = q.transpose(-3, -2) #reshape only keeps their linear order
        k = k.reshape(*k.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        k = k.transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        v = v.transpose(-3, -2)

        causal_mask = torch.ones(
            x.shape[-2],
            x.shape[-2],
            device=x.device,
            dtype=torch.bool
        )
        causal_mask = torch.tril(causal_mask, diagonal=0)

        head = scaled_dot_product_attention(q, k, v, causal_mask)
        heads = self.w_o(head.transpose(-2, -3).flatten(-2,-1))
        # heads.shape = [batch_size, seq_len, d_model]
        return heads


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(self, 
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_o = Linear(d_model, d_model, device, dtype)
        self.positional_embed = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x.shape = [batch_size, seq_len, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.reshape(*q.shape[:-1], self.num_heads, self.d_model // self.num_heads) 
        q = q.transpose(-3, -2) #reshape only keeps their linear order
        k = k.reshape(*k.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        k = k.transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        v = v.transpose(-3, -2)

        causal_mask = torch.ones(
            x.shape[-2],
            x.shape[-2],
            device=x.device,
            dtype=torch.bool
        )
        causal_mask = torch.tril(causal_mask, diagonal=0)

        if token_positions is None:
            token_positions = torch.arange(
                x.shape[-2],
                device=x.device,
                dtype=torch.long,
            ) # token_positon.shape = [seq_len, ]
        if token_positions.ndim == 2:
            token_positions = token_positions.unsqueeze(-2)
        # if token_position.shape = [batch_size, seq_len], we should tranform it into [B, 1, L]
        positional_q = self.positional_embed(q, token_positions)
        positional_k = self.positional_embed(k, token_positions)

        head = scaled_dot_product_attention(positional_q, positional_k, v, causal_mask)
        heads = self.w_o(head.transpose(-2, -3).flatten(-2,-1))
        # heads.shape = [batch_size, seq_len, d_model]
        return heads

        