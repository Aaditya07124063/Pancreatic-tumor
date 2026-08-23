import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from scratch in PyTorch.
    Computes Q, K, V projections, splits into multiple heads, performs scaled
    dot-product attention with optional mask, and projects outputs back.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # Projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Split the d_model dimension into num_heads and depth, and permute to:
        (batch_size, num_heads, seq_len, depth)
        """
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, depth)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: queries of shape (batch_size, seq_len_q, d_model)
            k: keys of shape (batch_size, seq_len_k, d_model)
            v: values of shape (batch_size, seq_len_v, d_model)
            mask: Optional tensor of shape broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
                  where 1 indicates valid elements, and 0 indicates masked/ignored elements.
        Returns:
            output: attention outputs of shape (batch_size, seq_len_q, d_model)
            attention_weights: raw attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        
        # 1. Project inputs
        # (batch_size, seq_len, d_model)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)
        
        # 2. Split heads
        # (batch_size, num_heads, seq_len, depth)
        q_heads = self.split_heads(q_proj, batch_size)
        k_heads = self.split_heads(k_proj, batch_size)
        v_heads = self.split_heads(v_proj, batch_size)
        
        # 3. Scaled dot-product attention
        # Q K^T / sqrt(d_k)
        # q_heads: (batch_size, num_heads, seq_len_q, depth)
        # k_heads^T: (batch_size, num_heads, depth, seq_len_k)
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.depth)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
                
        # Softmax over the last dimension (seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)
        
        # Multiply weights by values
        # (batch_size, num_heads, seq_len_q, seq_len_k) * (batch_size, num_heads, seq_len_v, depth)
        # Result shape: (batch_size, num_heads, seq_len_q, depth)
        context = torch.matmul(attn_weights_drop, v_heads)
        
        # 4. Concatenate heads and project output
        # Permute back: (batch_size, seq_len_q, num_heads, depth)
        context = context.permute(0, 2, 1, 3).contiguous()
        # View as original d_model shape: (batch_size, seq_len_q, d_model)
        context = context.view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attn_weights
