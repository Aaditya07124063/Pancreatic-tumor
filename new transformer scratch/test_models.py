import torch
import torch.nn as nn
from models import MultiHeadAttention, Transformer, VisionTransformer

def test_multi_head_attention():
    print("Testing Multi-Head Attention...")
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 12
    d_model = 64
    num_heads = 4
    
    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_k, d_model)
    v = torch.randn(batch_size, seq_len_k, d_model)
    
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    out, weights = attn(q, k, v)
    assert out.shape == (batch_size, seq_len_q, d_model), f"Expected {(batch_size, seq_len_q, d_model)}, got {out.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k), f"Expected {(batch_size, num_heads, seq_len_q, seq_len_k)}, got {weights.shape}"
    
    mask = torch.ones(batch_size, 1, 1, seq_len_k, dtype=torch.bool)
    mask[:, :, :, -2:] = False
    out_masked, weights_masked = attn(q, k, v, mask=mask)
    
    assert (weights_masked[:, :, :, -2:] < 1e-6).all(), "Masked weights are not close to zero"
    print("✓ Multi-Head Attention checks passed!")


def test_transformer_seq2seq():
    print("\nTesting Sequence-to-Sequence Transformer...")
    batch_size = 2
    src_vocab_size = 100
    tgt_vocab_size = 150
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    
    src = torch.randint(1, src_vocab_size, (batch_size, 15))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, 20))
    src[0, -3:] = 0
    tgt[0, -2:] = 0
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    logits = model(src, tgt, src_pad_idx=0, tgt_pad_idx=0)
    assert logits.shape == (batch_size, 20, tgt_vocab_size), f"Expected {(batch_size, 20, tgt_vocab_size)}, got {logits.shape}"
    
    loss = logits.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} was not computed"
            
    print("✓ Seq2Seq Transformer checks passed!")


def test_vision_transformer():
    print("\nTesting Vision Transformer (ViT)...")
    batch_size = 2
    img_size = 224
    patch_size = 16
    in_chans = 3
    num_classes = 2
    embed_dim = 192
    depth = 4
    num_heads = 3
    
    x = torch.randn(batch_size, in_chans, img_size, img_size)
    
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=0.1
    )
    
    logits = vit(x)
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    
    loss = logits.sum()
    loss.backward()
    
    for name, param in vit.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} was not computed"
            
    print("✓ Vision Transformer checks passed!")


if __name__ == "__main__":
    test_multi_head_attention()
    test_transformer_seq2seq()
    test_vision_transformer()
    print("\nAll models verified successfully!")
