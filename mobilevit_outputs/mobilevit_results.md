# MobileViT — Pancreatic CT Classification Results

This document contains the compiled results of the **MobileViT** (PyTorch, MPS accelerated) fine-tuning evaluation across 5 random seeds (stratified 80/10/10 split).

## 📊 Summary Performance Table

| Seed | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score | Kappa | Test Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 98.00% | 99.00% | 99.00% | 98.31% | 100.00% | 99.15% | 97.94% | 0.0181 |
| **7** | 98.00% | 99.00% | 99.00% | 100.00% | 98.28% | 99.13% | 97.95% | 0.2118 |
| **21** | 98.12% | 98.00% | 98.00% | 96.67% | 100.00% | 98.31% | 95.87% | 0.0515 |
| **99** | 98.25% | 99.00% | 97.00% | 100.00% | 94.83% | 97.35% | 93.90% | 0.2463 |
| **123** | 98.00% | 99.00% | 99.00% | 98.31% | 100.00% | 99.15% | 97.94% | 0.0096 |
| **AVG** | **98.07%** | **98.80%** | **98.40%** | **98.66%** | **98.62%** | **98.61%** | **96.72%** | **0.1075** |

---

### 📝 Key Observations:
- **Outstanding Classification Performance**: MobileViT achieves an outstanding average test accuracy of **98.40%** and an average weighted F1-score of **98.61%**.
- **Excellent Kappa Agreement**: The average Cohen's Kappa score is **96.72%**, indicating strong non-random classification agreement.
- **Perfect Recall Seeds**: In 3 out of 5 seeds (42, 21, 123), the model achieved a **100.00% recall rate** on the test set, indicating zero false negatives.
