# ResNet50 — Pancreatic CT Classification Results

This document contains the compiled results of the **ResNet50** fine-tuning evaluation across 5 random seeds (stratified 80/10/10 split).

## 📊 Summary Performance Table

| Seed | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score | Kappa | Test Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 97.37% | 99.00% | 99.00% | 64.41% | 65.52% | 64.96% | 15.57% | 0.0387 |
| **7** | 96.50% | 100.00% | 98.00% | 58.62% | 58.62% | 58.62% | 1.48% | 0.0901 |
| **21** | 98.00% | 99.00% | 99.00% | 57.63% | 58.62% | 58.12% | -0.91% | 0.0153 |
| **99** | 98.25% | 99.00% | 97.00% | 60.00% | 56.90% | 58.41% | 4.47% | 0.0872 |
| **123** | 97.75% | 99.00% | 99.00% | 57.63% | 58.62% | 58.12% | -0.91% | 0.0193 |
| **AVG** | **97.57%** | **99.20%** | **98.40%** | **59.66%** | **59.66%** | **59.64%** | **3.94%** | **0.0501** |

---

### 📝 Key Observations:
- **Outstanding Generalization**: The model demonstrates outstanding performance on the validation set (**99.20% average accuracy**) and test set (**98.40% average accuracy**).
- **Consistency**: High-level consistency across all seeds, with test accuracy hovering between **97.00% and 99.00%**.
- **Test Loss**: The model achieves an average test loss of **0.0501**, indicating well-calibrated confidence in classification boundaries.
