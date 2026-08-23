import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention is All You Need".
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a buffer for positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    A single Encoder Layer consisting of:
    1. Multi-Head Attention (Self-Attention)
    2. Feed Forward Network
    Residual connections and Layer Normalization are applied to both.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.norm_first = norm_first
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        if self.norm_first:
            norm_x = self.norm1(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask)
            x = x + self.dropout1(attn_out)
            
            norm_x2 = self.norm2(x)
            ffn_out = self.ffn(norm_x2)
            x = x + self.dropout2(ffn_out)
        else:
            attn_out, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_out))
            
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))
            
        return x


class DecoderLayer(nn.Module):
    """
    A single Decoder Layer consisting of:
    1. Masked Multi-Head Self-Attention (Causal attention)
    2. Multi-Head Cross-Attention (Query from decoder self-attention, Key/Value from encoder output)
    3. Feed Forward Network
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.norm_first = norm_first
        
        # Self-attention block
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention block (queries from decoder, keys/values from encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, 
                self_mask: torch.Tensor = None, cross_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Decoder hidden state of shape (batch_size, seq_len_tgt, d_model)
            enc_out: Encoder outputs of shape (batch_size, seq_len_src, d_model)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
        """
        if self.norm_first:
            norm_x = self.norm1(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, self_mask)
            x = x + self.dropout1(attn_out)
            
            norm_x2 = self.norm2(x)
            cross_out, _ = self.cross_attn(norm_x2, enc_out, enc_out, cross_mask)
            x = x + self.dropout2(cross_out)
            
            norm_x3 = self.norm3(x)
            ffn_out = self.ffn(norm_x3)
            x = x + self.dropout3(ffn_out)
        else:
            attn_out, _ = self.self_attn(x, x, x, self_mask)
            x = self.norm1(x + self.dropout1(attn_out))
            
            cross_out, _ = self.cross_attn(x, enc_out, enc_out, cross_mask)
            x = self.norm2(x + self.dropout2(cross_out))
            
            ffn_out = self.ffn(x)
            x = self.norm3(x + self.dropout3(ffn_out))
            
        return x


class Encoder(nn.Module):
    """
    Encoder stack of N EncoderLayers.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, max_len: int = 5000, dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Token indices of shape (batch_size, seq_len)
            mask: Optional source attention mask
        """
        x = self.embedding(src)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)


class Decoder(nn.Module):
    """
    Decoder stack of N DecoderLayers.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, max_len: int = 5000, dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor, 
                self_mask: torch.Tensor = None, cross_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt: Token indices of shape (batch_size, seq_len_tgt)
            enc_out: Encoder outputs of shape (batch_size, seq_len_src, d_model)
            self_mask: Causal/target self-attention mask
            cross_mask: Source padding mask
        """
        x = self.embedding(tgt)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
            
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete sequence-to-sequence Transformer model from scratch.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = 512, num_layers: int = 6, num_heads: int = 8, 
                 d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1, 
                 norm_first: bool = True):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size, d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, d_ff=d_ff, max_len=max_len, dropout=dropout,
            norm_first=norm_first
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, d_ff=d_ff, max_len=max_len, dropout=dropout,
            norm_first=norm_first
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        
        tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(1)
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
        cross_mask = src_mask
        
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, cross_mask)
        
        logits = self.fc_out(dec_out)
        return logits
