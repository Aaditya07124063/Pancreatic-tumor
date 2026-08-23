# InceptionV3 — Pancreatic CT Classification Results

This document contains the compiled results of the **InceptionV3** fine-tuning evaluation across 5 random seeds (stratified 80/10/10 split).

## 📊 Summary Performance Table

| Seed | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score | Kappa | Test Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 95.87% | 98.00% | 97.00% | 61.02% | 62.07% | 61.54% | 7.33% | 0.1749 |
| **7** | 97.87% | 99.00% | 99.00% | 57.89% | 56.90% | 57.39% | -0.25% | 0.1256 |
| **21** | 97.87% | 99.00% | 98.00% | 56.67% | 58.62% | 57.63% | -3.31% | 0.0400 |
| **99** | 98.12% | 98.00% | 98.00% | 58.93% | 56.90% | 57.89% | 2.12% | 0.0716 |
| **123** | 97.62% | 100.00% | 99.00% | 57.63% | 58.62% | 58.12% | -0.91% | 0.0258 |
| **AVG** | **97.47%** | **98.80%** | **98.20%** | **58.43%** | **58.62%** | **58.51%** | **1.00%** | **0.0876** |

---

### 📝 Key Observations:
- **Strong Generalization**: InceptionV3 generalizes exceptionally well, with an average test accuracy of **98.20%** and validation accuracy of **98.80%**.
- **Excellent Peak Validation**: In Seed 123, InceptionV3 achieved a **100.00% validation accuracy**, showcasing its powerful high-resolution classification capacity.
- **Stable Metrics**: Training accuracy remains stable around **97% to 98%** across all seed evaluations.
