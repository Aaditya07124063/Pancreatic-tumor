import torch
import torch.nn as nn
from .transformer import EncoderLayer

class PatchEmbedding(nn.Module):
    """
    Project 2D image patches into 1D vectors for Transformer input.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv projection: (B, C, H, W) -> (B, embed_dim, grid_size, grid_size)
        x = self.proj(x)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) from scratch in PyTorch.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 num_classes: int = 2, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 norm_first: bool = True):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        d_ff = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            EncoderLayer(
                d_model=embed_dim, num_heads=num_heads, d_ff=d_ff, 
                dropout=dropout, norm_first=norm_first
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits
