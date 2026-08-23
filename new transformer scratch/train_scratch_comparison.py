import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Use absolute path to resolve dataset correctly from the subfolder
DATA_DIR   = "/Users/chudamaniray/Desktop/research/train"
CLASSES    = ["normal", "pancreatic_tumor"]
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 2  # Keep it small (2 epochs) for quick verification of components
LR         = 1e-4

# Select Device (GPU / Mac MPS / CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[Config] Using device: {DEVICE}")

# ─── 1. DATA LOADING (Real Pancreatic CT Scan Images) ──────────────────────────
def load_dataset(data_dir, classes):
    paths, labels = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return np.array(paths), np.array(labels)

class PancreaticDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ─── 2. DEEP LEARNING MODEL: CUSTOM CNN FROM SCRATCH ──────────────────────────
class ScratchCNN(nn.Module):
    """
    Standard Deep Learning model: Convolutional Neural Network (CNN).
    Main components: Convolutions (spatial feature extraction), Max Pooling (downsampling),
    Batch Normalization (stabilizing activation distribution), and Fully Connected linear layers.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Conv block 1: extract low-level features (edges/textures)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # 224 -> 112
        
        # Conv block 2: mid-level patterns (shapes)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # 112 -> 56
        
        # Conv block 3: high-level semantic representation (tumor boundaries)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # 56 -> 28
        
        # Global Average Pool
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # collapses to (B, 64, 1, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 1. Feature Map Extraction (Inductive bias: local translation invariance)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 2. Classifier Head
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten to feature vector
        logits = self.fc(x)
        return logits


# ─── 3. TRANSFORMER MODEL: VISION TRANSFORMER (ViT) FROM SCRATCH ──────────────
class ScratchPatchEmbedding(nn.Module):
    """
    Divides the input image into grid patches and projects each patch linearly.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)       # Convolution projects patches: (B, embed_dim, 14, 14)
        x = x.flatten(2)       # Flatten spatial grid: (B, embed_dim, 196)
        x = x.transpose(1, 2)  # Reshape to token sequence: (B, 196, embed_dim)
        return x

class ScratchMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Computes attention scores between all patch tokens globally.
    """
    def __init__(self, embed_dim=128, num_heads=4):
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
        # Linear projection to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context extraction: Attention * V
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.out_proj(out)

class ScratchTransformerEncoder(nn.Module):
    """
    Stack of Self-Attention and MLP Block.
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ScratchMultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ScratchViT(nn.Module):
    """
    Complete Vision Transformer from scratch.
    Main components: PatchProjection, ClassToken, PositionalEmbedding, TransformerEncoder, classification head.
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=2, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.patch_embed = ScratchPatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 1. Classification Token (CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 2. Positional Embedding (explicit token positions)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 3. Encoder Stack
        self.blocks = nn.ModuleList([
            ScratchTransformerEncoder(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # Patch Projection: image (B, 3, 224, 224) -> patches (B, 196, embed_dim)
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 197, embed_dim)
        
        # Add spatial positions explicitly
        x = x + self.pos_embed
        
        # Global Transformer Encoder Attention computations
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Extract representation of CLS token for classification
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits


# ─── 4. TRAINING PIPELINE ─────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, model_name):
    print(f"\n--- Training {model_name} from scratch ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        # Validation epoch
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"  Epoch {epoch:02d} | Train Loss: {train_loss / total:.4f} | Train Acc: {correct / total * 100:.2f}% "
              f"| Val Loss: {val_loss / val_total:.4f} | Val Acc: {val_correct / val_total * 100:.2f}%")


def main():
    paths, labels = load_dataset(DATA_DIR, CLASSES)
    if len(paths) == 0:
        print(f"Dataset not found at '{DATA_DIR}'. Please ensure train directory is present.")
        return
        
    print(f"Dataset Loaded. Total CT scans: {len(paths)} | Classes: {CLASSES}")
    
    tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    
    train_loader = DataLoader(PancreaticDataset(tr_p, tr_l, get_transforms()), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PancreaticDataset(va_p, va_l, get_transforms()), batch_size=BATCH_SIZE, shuffle=False)
    
    # 1. Train Custom CNN
    cnn = ScratchCNN().to(DEVICE)
    train_model(cnn, train_loader, val_loader, "ScratchCNN")
    
    # 2. Train Custom ViT
    vit = ScratchViT(embed_dim=128, depth=2, num_heads=4).to(DEVICE)
    train_model(vit, train_loader, val_loader, "ScratchViT")
    
    print("\n" + "="*80)
    print("                      COMPONENT COMPARISON SUMMARY")
    print("="*80)
    print("  Component         | CNN Scratch Model            | ViT (Transformer) Scratch Model")
    print("  ------------------+------------------------------+--------------------------------")
    print("  Input Layer       | Raw grid: (B, 3, 224, 224)   | 14x14 grid of 16x16 pixel patches")
    print("  Feature Extraction| Local filters (Conv2D 3x3)   | Global Self-Attention (Multi-Head)")
    print("  Positional Info   | Preserved implicitly by grid | Added explicitly (Positional Embed)")
    print("  Feature Aggregation| Max Pooling / Downsampling  | Class (CLS) Token representing image")
    print("  Final Classifier  | Flat Average Pool + Linear   | CLS representation + Linear Head")
    print("="*80)


if __name__ == "__main__":
    main()
