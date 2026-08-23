# Transformer & Vision Transformer (ViT) From Scratch in PyTorch

This directory contains a clean, modular, and fully documented implementation of both the standard **sequence-to-sequence Transformer** and the **Vision Transformer (ViT)** implemented from scratch using pure PyTorch.

---

## Directory Structure

```text
new transformer scratch/
├── models/
│   ├── __init__.py          # Package entrypoint exporting all modules
│   ├── attention.py         # Multi-Head Attention layer
│   ├── transformer.py       # Positional Encoding, EncoderLayer, DecoderLayer, Seq2Seq Transformer
│   └── vit.py               # Patch Embedding and Vision Transformer (ViT) model
├── test_models.py           # Unit tests checking shapes, masks, and gradients
├── compare_scratch_models.py# Trace and compare CNN vs. ViT features side-by-side on dummy data
├── train_scratch_comparison.py # Train scratch CNN vs. ViT side-by-side on the real pancreatic CT dataset
├── train_nlp_dummy.py       # Example sequence-to-sequence reversal training loop
├── train_vit_pancreatic.py  # Drop-in script to train custom ViT on the real Pancreatic CT dataset
└── README.md                # Mathematical formulations and instructions
```

---

## Architectural Breakdown

### 1. Multi-Head Attention (`models/attention.py`)
Computes query ($Q$), key ($K$), and value ($V$) projections of dimension $d_{model}$ split into $h$ heads of depth $d_k = d_{model}/h$.
The scaled dot-product attention formula is:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

where $M$ is an optional mask (e.g., causal/look-ahead mask or padding mask) where elements corresponding to masked positions are filled with $-\infty$ (or $-10^9$) before the softmax operation.

### 2. Positional Encoding (`models/transformer.py`)
Since Transformers process tokens concurrently without recurrence, sinusoidal positional encodings are added to the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3. Seq2Seq Transformer (`models/transformer.py`)
Combines:
- **Encoder**: Multiple `EncoderLayer` blocks consisting of Self-Attention and a position-wise Feed Forward Network (FFN).
- **Decoder**: Multiple `DecoderLayer` blocks adding Cross-Attention where keys/values come from the Encoder outputs, and queries come from the decoder self-attention output.
- **Normalization**: Supports `norm_first` (Pre-LN architecture), which is more stable for deep training than the original Post-LN architecture.

### 4. Vision Transformer (`models/vit.py`)
Adapts the Transformer for 2D images by:
- **Patch Embedding**: Splitting an image $(H, W, C)$ into $N = \frac{HW}{P^2}$ patches of size $(P, P)$, projecting them to $d_{model}$ dimensions.
- **CLS Token**: Prepending a learnable classification token to the start of the sequence.
- **Positional Embedding**: Adding a learnable $1D$ positional embedding of size $(N + 1, d_{model})$.
- **Encoder Stack**: Passing the sequence through a stack of custom Encoder blocks.
- **Classifier Head**: Extracting the final representation of the CLS token and feeding it through a multi-layer perceptron (MLP) for classification.

---

## How to Run & Verify

### 1. Execute Unit Verification Tests
Run the test script to verify forward pass shapes, attention masking, and parameter gradient updates:
```bash
python test_models.py
```

### 2. Trace CNN vs. ViT Shapes Side-by-Side (Dummy Data)
Run the comparison demo to print out intermediate tensor shapes after each layer:
```bash
python compare_scratch_models.py
```

### 3. Train Scratch CNN vs. ViT Side-by-Side (Real Pancreatic CT Scan Images)
Run the short comparative training script (2 epochs) on your actual pancreatic dataset:
```bash
python train_scratch_comparison.py
```

### 4. Run NLP Dummy Copy/Reversal Training
Run the sequence reversal training script to see how the sequence-to-sequence model trains and converges on a toy problem:
```bash
python train_nlp_dummy.py
```

### 5. Train ViT on your Pancreatic Cancer CT Scan Dataset
Train your custom Vision Transformer from scratch on the dataset:
```bash
python train_vit_pancreatic.py
```
This script runs a 5-seed cross-validation splits evaluation, computes loss/accuracy/Precision/Recall/F1/Cohen's Kappa metrics, generates training curve plots and confusion matrices, and logs results under `./vit_scratch_outputs/`.
