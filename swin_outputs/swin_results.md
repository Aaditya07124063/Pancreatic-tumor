# Swin Transformer — Pancreatic CT Classification Results

This document contains the compiled results of the **Swin Transformer** (PyTorch, MPS accelerated) fine-tuning evaluation across 5 random seeds (stratified 80/10/10 split).

## 📊 Summary Performance Table

| Seed | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score | Kappa | Test Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 98.00% | 99.00% | 99.00% | 98.31% | 100.00% | 99.15% | 97.94% | 0.0274 |
| **7** | 98.00% | 99.00% | 99.00% | 100.00% | 98.28% | 99.13% | 97.95% | 0.0315 |
| **21** | 98.25% | 98.00% | 98.00% | 96.67% | 100.00% | 98.31% | 95.87% | 0.0271 |
| **99** | 98.25% | 99.00% | 97.00% | 100.00% | 94.83% | 97.35% | 93.90% | 0.0686 |
| **123** | 98.00% | 99.00% | 99.00% | 98.31% | 100.00% | 99.15% | 97.94% | 0.0149 |
| **AVG** | **98.10%** | **98.80%** | **98.40%** | **98.66%** | **98.62%** | **98.61%** | **96.72%** | **0.0339** |

---

### 📝 Key Observations:
- **Exceptional Accuracy**: Swin Transformer achieves an outstanding average test accuracy of **98.40%** and weighted F1-score of **98.61%**.
- **Excellent Test Loss**: The average test loss of **0.0339** is the lowest and most stable among all evaluated deep learning networks, indicating high classification confidence and clean decision boundaries.
- **Top-Tier Agreement**: The average Cohen's Kappa score of **96.72%** matches MobileViT as the best overall deep model, showing near-perfect classification agreement.
