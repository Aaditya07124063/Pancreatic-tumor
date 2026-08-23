import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# MODEL 1: DEEP LEARNING MODEL (CUSTOM CNN FROM SCRATCH)
# ==============================================================================

class SimpleCNN(nn.Module):
    """
    A classic Convolutional Neural Network (CNN) for image classification.
    Processes images using localized convolution kernels, batch normalization,
    releasing non-linearity (ReLU), and downsampling (MaxPooling).
    """
    def __init__(self, in_chans=3, num_classes=2):
        super().__init__()
        print("[SimpleCNN] Initializing components...")
        
        # Block 1: local spatial features (224x224 -> 112x112 after maxpool)
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: mid-level spatial features (112x112 -> 56x56 after maxpool)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: high-level spatial features (56x56 -> 28x28 after maxpool)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling to collapse spatial dimensions (28x28 -> 1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected (Linear) Layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        print(f"\n--- SimpleCNN Forward Pass (Input: {x.shape}) ---")
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        print(f"After Block 1 (Conv + BN + ReLU + MaxPool): {x.shape}")
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        print(f"After Block 2 (Conv + BN + ReLU + MaxPool): {x.shape}")
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        print(f"After Block 3 (Conv + BN + ReLU + MaxPool): {x.shape}")
        
        # Global pooling: collapses spatial grid to a single vector per image
        x = self.global_pool(x) # Shape: (B, 128, 1, 1)
        print(f"After Global Average Pooling: {x.shape}")
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.view(x.size(0), -1)
        print(f"After Flatten: {x.shape}")
        
        # Final classification
        logits = self.fc(x)
        print(f"Final Output Logits: {logits.shape}")
        return logits


# ==============================================================================
# MODEL 2: TRANSFORMER MODEL (VISION TRANSFORMER - ViT FROM SCRATCH)
# ==============================================================================

class PatchEmbeddingScratch(nn.Module):
    """
    Splits an image into 2D patches and projects them into a 1D sequence of vectors.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x) # (B, embed_dim, 14, 14)
        x = x.flatten(2) # (B, embed_dim, 196)
        x = x.transpose(1, 2) # (B, 196, embed_dim)
        return x


class MultiHeadAttentionScratch(nn.Module):
    """
    Multi-Head Attention from scratch. Computes QKV, splits heads, scale dot-product,
    applies softmax, and projects output.
    """
    def __init__(self, embed_dim=192, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.out_proj(out), attn_weights


class TransformerEncoderBlockScratch(nn.Module):
    """
    Standard Transformer encoder block with Multi-head attention and Feed-forward network.
    """
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionScratch(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        attn_out, weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x, weights


class SimpleViT(nn.Module):
    """
    A simplified Vision Transformer (ViT) implemented from scratch.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=192, depth=2, num_heads=3):
        super().__init__()
        print("[SimpleViT] Initializing components...")
        
        self.patch_embed = PatchEmbeddingScratch(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlockScratch(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        print(f"\n--- SimpleViT Forward Pass (Input: {x.shape}) ---")
        
        x = self.patch_embed(x)
        print(f"After Patch Embedding Projection: {x.shape} (B, num_patches, embed_dim)")
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"After Prepending CLS Token: {x.shape} (B, num_patches + 1, embed_dim)")
        
        x = x + self.pos_embed
        print(f"After Adding Positional Embedding: {x.shape}")
        
        attn_maps = []
        for idx, block in enumerate(self.blocks):
            x, weights = block(x)
            attn_maps.append(weights)
            print(f"After Transformer Encoder Block {idx + 1}: {x.shape}")
            
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        print(f"Extracted CLS Token representation: {cls_rep.shape} (B, embed_dim)")
        
        logits = self.head(cls_rep)
        print(f"Final Output Logits: {logits.shape}")
        
        return logits, attn_maps


# ==============================================================================
# MAIN RUN AND COMPARISON TEST
# ==============================================================================

def main():
    print("=" * 70)
    print("   Comparing Scratch Models: Custom CNN vs. Custom Vision Transformer")
    print("=" * 70)
    
    print("\n1. Generating dummy batch...")
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    print(f"Batch created! Images shape: {dummy_images.shape} | Labels: {dummy_labels.tolist()}")
    
    print("\n2. Instantiating models...")
    cnn_model = SimpleCNN(in_chans=3, num_classes=2)
    vit_model = SimpleViT(img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=192, depth=2, num_heads=3)
    
    print("\n3. Executing CNN pass...")
    cnn_logits = cnn_model(dummy_images)
    
    print("\n4. Executing Transformer (ViT) pass...")
    vit_logits, attention_maps = vit_model(dummy_images)
    print(f"ViT Attention Maps generated: {len(attention_maps)} maps (one per block)")
    print(f"Shape of first block attention weights: {attention_maps[0].shape} (B, heads, tokens, tokens)")
    
    print("\n5. Running Micro Training Step (Optimization & Gradients)...")
    criterion = nn.CrossEntropyLoss()
    
    cnn_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01)
    cnn_optimizer.zero_grad()
    cnn_loss = criterion(cnn_logits, dummy_labels)
    cnn_loss.backward()
    cnn_optimizer.step()
    print(f"✓ CNN backward pass complete! Loss: {cnn_loss.item():.4f}")
    
    vit_optimizer = torch.optim.AdamW(vit_model.parameters(), lr=0.0001)
    vit_optimizer.zero_grad()
    vit_loss = criterion(vit_logits, dummy_labels)
    vit_loss.backward()
    vit_optimizer.step()
    print(f"✓ ViT backward pass complete! Loss: {vit_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("                      ARCHITECTURAL SUMMARY")
    print("=" * 70)
    print("   Component                 | Custom CNN                 | Custom ViT (Transformer)")
    print("   --------------------------+----------------------------+-------------------------")
    print("   1. Basic Input Processing | Full grid convolution      | Flattened 16x16 Patches")
    print("   2. Receptive Field        | Local (3x3 kernel size)    | Global (Self-Attention)")
    print("   3. Position Awareness     | Implicit (via spatial grid)| Explicit (Positional Embed)")
    print("   4. Primary Operation      | Weight shared Convolution  | Scaled Dot-Product Attn")
    print("   5. Downsampling / Pooling | MaxPooling layers          | Global Average Pooling  ")
    print("                             |                            | of CLS/All tokens       ")
    print("=" * 70)
    print("Both models executed successfully from scratch!")

if __name__ == "__main__":
    main()
