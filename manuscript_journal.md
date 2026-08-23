---
title: "Apparent Near-Ceiling Performance in Pancreatic Tumour Classification on CT Is Explained by Dataset Provenance: A Benchmarking Study and Failure Analysis"
author: "Aaditya Adhikari"
date: "August 2026"
---

# Abstract

**Background.** Reported accuracies above 95% are now routine in the literature on binary pancreatic tumour classification from computed tomography slices. Whether such figures reflect diagnostic capability or dataset artefacts is rarely tested directly.

**Methods.** We benchmarked eleven pipelines on a 1,411-image pancreatic CT slice corpus under a uniform five-seed stratified protocol: four ImageNet-pretrained backbones fine-tuned end to end (ResNet50, InceptionV3, MobileViT, Swin-Tiny) and two frozen feature extractors (Xception, DenseNet121) each feeding seven downstream classifiers. We then subjected both the results and the corpus to three independent integrity analyses: an arithmetic consistency check across reported metrics, a content-hash audit of every image file, and a shortcut probe using only non-diagnostic image statistics.

**Results.** All fine-tuned networks achieved 98.2–98.4% test accuracy; four DenseNet121 pipelines achieved exactly 100.00% with zero variance. The consistency check exposed a metric-alignment defect in the two Keras pipelines: test-set shuffling desynchronised predictions from labels, yielding chance-level precision, recall, F1 and Cohen's κ (κ = 0.0394 and 0.0100) alongside correct accuracy. Correcting this against the saved checkpoints recovered κ = 0.9672 and 0.9630 and eliminated the apparent transformer advantage entirely. The hash audit found 206 byte-identical duplicate files among 999 training images (20.6%), 18 image hashes carrying contradictory class labels, and 29–36% per-seed test-partition contamination. The shortcut probe found that a logistic regression on seven non-diagnostic statistics achieves 97.20% cross-validated accuracy on the training pool and 96.60% on the external directory, against majority baselines of 57.86% and 54.61%. All ten model–seed evaluations on the external hold-out returned AUROC of exactly 1.0000 with exactly zero false negatives.

**Conclusions.** The corpus is separable by acquisition provenance — windowing, background convention, source pipeline and burned-in annotation — rather than by pathology. The architecture comparison the study set out to make is not identifiable from these data: after correction, four architectures differ by less than the seed-to-seed standard deviation. We report this as a negative result, quantify each failure mode, and propose a six-step audit protocol that detects all three at negligible computational cost.

**Keywords:** shortcut learning; dataset leakage; pancreatic ductal adenocarcinoma; computed tomography; reproducibility; vision transformers; evaluation methodology.

---

# 1. Introduction

Pancreatic ductal adenocarcinoma carries one of the poorest prognoses in oncology, with five-year survival in the low single digits for late-stage disease and a marked survival advantage for resectable presentations. The stage-dependence of outcome makes automated detection on routine abdominal CT an unusually high-value target for machine learning, and a substantial applied literature has developed around it.

That literature has converged on a characteristic result shape: an ImageNet-pretrained backbone, fine-tuned or frozen, evaluated on a curated slice dataset, reporting accuracy between 95% and 99%. Comparative studies then rank architectures by fractions of a percentage point, and increasingly report that transformer-based backbones edge out convolutional ones.

This study was designed to contribute to that literature. The research question was conventional: under a controlled, identical five-seed protocol, do transformer backbones outperform convolutional ones for binary pancreatic tumour classification, and does end-to-end fine-tuning outperform frozen-feature extraction with a classical downstream classifier? Eleven pipelines were implemented, trained and evaluated. Every one produced an accuracy in the expected range.

The study did not survive its own integrity checks. This paper reports what happened, why it happened, and what it implies for the interpretation of results of this shape.

## 1.1 Contributions

1. **A documented, reproducible metric-alignment defect.** We characterise a failure mode in which `tf.data` shuffling applied to an evaluation pipeline desynchronises predictions from labels, producing correct accuracy alongside chance-level precision, recall, F1 and κ. We give its numerical signature, derive the expected chance values analytically, correct it against saved checkpoints, and verify split reproduction by accuracy fingerprint.

2. **A content-level audit of the corpus** revealing 20.6% file-level duplication, cross-label hash collisions, and per-seed test contamination of 29–36%, together with the slice-series structure that makes random per-image splitting inappropriate.

3. **A direct demonstration of shortcut learning**, in which seven non-diagnostic scalar statistics recover 96.6–97.2% of class membership, and the confound is traced to differing windowing, background rendering, source provenance and burned-in annotation between classes.

4. **An external-hold-out analysis** whose failure signature — AUROC exactly 1.0000 across ten independent evaluations, recall exactly 100.00% in every seed, and errors that are purely threshold artefacts rather than ranking failures — corroborates the shortcut hypothesis.

5. **A six-step audit protocol** with negligible computational cost that detects all three failure modes prior to training.

## 1.2 What this paper is not

This is not a claim that deep learning cannot detect pancreatic tumours on CT, nor a criticism of the architectures evaluated. It is a claim about what one specific, conventionally assembled corpus can and cannot support, and about how easily a study of this shape can produce a result that is internally consistent, externally validated by its own lights, and nonetheless meaningless.

---

# 2. Related Work

## 2.1 Deep learning for pancreatic lesion classification

Transfer learning from ImageNet dominates the applied literature on pancreatic CT. Residual networks, Inception variants, DenseNet and Xception are the most frequently reported backbones, used either fine-tuned end to end or frozen as feature extractors feeding support vector machines, random forests or gradient-boosted trees. Reported accuracies cluster between 93% and 99%.

Attention-based architectures have entered the field more recently. The Vision Transformer established that pure self-attention over image patches is competitive with convolution given sufficient data or strong pretraining. Swin introduced hierarchical shifted-window attention, restoring the multi-scale inductive bias that plain ViTs lack, and MobileViT combined convolutional local processing with transformer global reasoning at mobile parameter budgets. Both have been applied to pancreatic and other abdominal CT tasks with reported results at or slightly above convolutional baselines.

## 2.2 Shortcut learning in medical imaging

Geirhos et al. formalised *shortcut learning*: decision rules that succeed on the training distribution and its independent-and-identically-distributed test split while failing to capture the intended concept. Medical imaging is unusually exposed to this failure mode, because acquisition is heterogeneous and disease prevalence is frequently confounded with acquisition setting.

The documented instances are numerous and instructive. Zech et al. showed that pneumonia classifiers on chest radiographs achieved high internal accuracy partly by identifying hospital-specific metal tokens and portable-scanner markers, and degraded sharply across sites. Winkler et al. demonstrated that surgical skin markings around lesions inflated melanoma classifier confidence. Oakden-Rayner et al. described hidden stratification, in which aggregate metrics conceal clinically important subgroup failures. Badgeley et al. found that hip fracture classifiers relied heavily on scanner and patient metadata rather than image content.

The instance reported here belongs to this family, with the distinguishing property that the confound is fully characterised by seven scalar statistics and is therefore straightforward to test for prospectively.

## 2.3 Leakage and reproducibility

Kapoor and Narayanan surveyed leakage across machine-learning-based science and identified train–test contamination as the single most common cause of irreproducible claims, documenting affected results across medicine, genomics, political science and neuroimaging.

In volumetric imaging the characteristic form is *slice-level splitting*. A CT study yields hundreds of contiguous axial slices; adjacent slices through the same anatomy are near-identical. Randomly partitioning at the slice level therefore places nearly the same image on both sides of the split, and the resulting estimate measures interpolation within a study rather than generalisation across patients. Patient- or study-level partitioning is the standard remedy.

The corpus examined here exhibits both slice-level splitting and the stronger, rarer form: exact byte-identical file duplication arising from manual dataset assembly.

## 2.4 Positioning

Each failure mode above has been documented individually. The contribution of this study is to observe all three co-occurring within a single conventional benchmarking exercise, to quantify each independently, and to demonstrate the consequence for the scientific claim: after correcting the metric defect, four architectures become statistically indistinguishable, and the entire remaining margin over a seven-feature logistic regression is one to two percentage points.

---

# 3. Materials

## 3.1 Corpus composition

The corpus is distributed as two top-level directories, each containing `normal/` and `pancreatic_tumor/` subdirectories.

**Table 1. Corpus composition.**

| Directory | Normal | Pancreatic tumour | Total | Class balance |
|:---|---:|---:|---:|---:|
| `train/` | 421 | 578 | 999 | 42.1% / 57.9% |
| `test/` | 225 | 187 | 412 | 54.6% / 45.4% |
| **Total** | **646** | **765** | **1,411** | 45.8% / 54.2% |

Note that class balance inverts between directories, which alone indicates that the two were assembled by different procedures.

## 3.2 Format heterogeneity

The dominant format is 512×512 8-bit greyscale JPEG: 964 of 999 training images and 411 of 412 external images. The exceptions are informative. Thirty-four tumour-class training images and one normal-class image are small RGB JPEGs at heterogeneous resolutions — 250×202 (n=7), 256×197 (n=6), 247×204 (n=6), 245×206 (n=3), 262×192 (n=3), 235×233 (n=2) and others.

Filename conventions corroborate mixed provenance. Within the tumour class, entries such as `6110de2a0caf85fdb76a1283626bcc_big_gallery.jpeg` follow web image-gallery naming, while `2012_02_27_16_50_42_331_2012_03_01_pancreas2 - Copy - Copy (4).jpg` combines a DICOM-derived timestamp with duplication suffixes generated by a desktop file manager. Part of the positive class was collected from published figures; part was duplicated by hand.

## 3.3 Series structure

Filenames fall into contiguous numbered blocks: `1-001`…`1-239`, `22 (1)`…`22 (119)`, `23 (102)`…`23 (183)`, and a `66`-prefixed block. Prefix distribution differs by directory and class — the `1-` prefix appears in all four class–directory combinations, `23` only in `train/normal`, and `22` and `66` only in `train/pancreatic_tumor`. This is the structure of a small number of contiguous axial slice series rather than a collection of independent patients, and it is the reason per-image random splitting is inappropriate here.

## 3.4 Content-hash audit

An MD5 digest was computed over the bytes of every image file.

**Table 2. Content-hash audit of the corpus.**

| Quantity | Value |
|:---|---:|
| Files in `train/` | 999 |
| Distinct image hashes in `train/` | 793 |
| Redundant copies | 206 (20.6%) |
| Hashes appearing under both class labels | 18 (36 files) |
| Files in `test/` | 412 |
| Distinct image hashes in `test/` | 412 |
| `test/` images byte-identical to a `train/` image | 8 |

**Table 3. Multiplicity distribution over distinct training-set hashes.**

| Copies of the same image | Distinct hashes |
|---:|---:|
| 1 | 600 |
| 2 | 187 |
| 3 | 4 |
| 6 | 1 |
| 7 | 1 |

The 18 cross-label collisions are the most serious single defect in the corpus. Identical pixel data is presented to the learner as `normal` in one directory and `pancreatic_tumor` in another. This guarantees an irreducible error floor, contributes contradictory gradients during training, and establishes that class assignment was not verified at ingestion.

---

# 4. Methods

## 4.1 Track A — end-to-end fine-tuning

Four ImageNet-pretrained backbones were fine-tuned with all layers trainable on the `train/` directory. Each run drew a fresh 80/10/10 stratified split — 799 training, 100 validation, 100 test images, the test partition containing 58 tumour and 42 normal images — under seeds {42, 7, 21, 99, 123}.

**Table 4. Track A configurations.**

| Model | Framework | Input | Optimiser | LR | Weight decay | Classification head |
|:---|:---|:---|:---|---:|---:|:---|
| ResNet50 | TensorFlow / Keras | 224² | Adam | 1e-4 | — | GAP → BN → Drop(0.4) → Dense(256, ReLU) → Drop(0.3) → Dense(1, sigmoid) |
| InceptionV3 | TensorFlow / Keras | 299² | Adam | 1e-4 | — | as above |
| MobileViT | PyTorch / HuggingFace | 224² | AdamW | 5e-5 | 0.01 | linear, 2 logits |
| Swin-Tiny | PyTorch / HuggingFace | 224² | AdamW | 5e-5 | 0.01 | linear, 2 logits |

Common settings: maximum 25 epochs; batch size 32; early stopping on validation accuracy with patience 10 and best-weight restoration; learning-rate reduction on validation-loss plateau (factor 0.5, patience 5). Keras models used binary cross-entropy; PyTorch models used two-class cross-entropy with gradient-norm clipping at 1.0 and Apple MPS acceleration. Augmentation comprised horizontal and vertical flips, rotation (90° multiples in Keras, ±15° in PyTorch), brightness jitter (δ = 0.2) and random resized cropping (scale 0.85–1.0). All twenty checkpoints were serialised.

## 4.2 Track B — frozen features with downstream classifiers

Xception (299², global max pooling) and DenseNet121 (224², global max pooling) were used as frozen ImageNet feature extractors. Unlike Track A, this track fits on all 999 `train/` images and partitions the 412-image `test/` directory 50/50 into 206 validation and 206 test images per seed.

Seven downstream classifiers were trained on the resulting embeddings: SVM with RBF kernel, random forest (100 trees), AdaBoost (100 estimators), k-nearest neighbours (k=5), bagging (100 estimators), a multilayer perceptron (128–64 hidden units, 300 iterations) and XGBoost (100 estimators). Two recurrent baselines, LSTM(64) and Bi-LSTM(64), were additionally trained on embeddings reshaped to a single timestep.

## 4.3 Track C — transformer from scratch

A complete sequence-to-sequence Transformer and Vision Transformer were implemented in pure PyTorch, comprising scaled dot-product multi-head attention with additive masking, sinusoidal positional encoding, pre-layer-norm encoder and decoder blocks with cross-attention, patch embedding, a learnable classification token, learnable positional embeddings and an MLP classification head. Unit tests verify forward-pass shapes, causal and padding mask behaviour, and gradient propagation to all parameters. No trained artefacts were produced; this track is reported as implemented and unevaluated.

## 4.4 Evaluation metrics

Accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1, Cohen's κ and AUROC, reported as mean ± standard deviation over five seeds. Dispersion is reported as the population standard deviation over the five seed values (`ddof = 0`), consistently across all tables; the five seeds constitute the complete set of runs performed rather than a sample from a larger population. Confusion matrices are reported per seed for the corrected analyses.

## 4.5 Metric-alignment defect and its correction

Both Keras training scripts construct all three partitions — including test — through a shared helper terminating in

```python
ds = ds.shuffle(buffer_size=len(paths), seed=seed).batch(batch_size).prefetch(AUTOTUNE)
```

`model.evaluate(test_ds)` is unaffected: accuracy is accumulated batch-wise, and each batch carries its own labels alongside its own images, so shuffling is immaterial. Accuracy was therefore correct throughout. The subsequent block, however, reads

```python
y_prob = model.predict(test_ds, verbose=0).ravel()   # shuffled ordering
y_pred = (y_prob >= 0.5).astype(int)
prec   = precision_score(te_l, y_pred, zero_division=0)   # original ordering
```

where `te_l` is the label array in the ordering produced by `train_test_split`. Each prediction is scored against a different image's label.

**Analytic signature.** Let the test partition contain $n = 100$ images of which $p = 58$ are positive, and let the model predict $\hat{p} \approx 59$ positives. Under a uniformly random permutation of predictions relative to labels, the expected number of true positives is $\hat{p}\,p/n \approx 34.2$. Expected precision is $34.2/59 \approx 0.580$ and expected recall $34.2/58 \approx 0.590$, with κ near zero by construction. The originally reported values — precision 0.5763–0.6441, recall 0.5690–0.6552, κ from −0.0091 to 0.1557 — fall within this band, confirming the diagnosis independently of any re-execution.

**Correction procedure.** `reevaluate_fixed.py` reloads each saved checkpoint, regenerates the identical stratified split from the same seed, and predicts from a plain unshuffled NumPy array, deriving `y_true` and `y_pred` from a single aligned source.

**Split-reproduction verification.** Because `os.listdir` ordering is not guaranteed stable across filesystems, faithful reproduction of the original split cannot be assumed. We verify it using accuracy as a fingerprint: accuracy was never corrupted by the defect, so agreement between recomputed and originally logged accuracy establishes that the same images were evaluated. All ten model–seed combinations matched to within floating-point tolerance, and the script is written to abort with an explicit warning otherwise.

The PyTorch pipelines were never affected. Their test loaders use `shuffle=False`, and more importantly they accumulate `all_preds` and `all_labels` within the same loop iteration, making desynchronisation structurally impossible regardless of ordering.

## 4.6 Shortcut probe

Seven scalar statistics were extracted per image: mean intensity, intensity standard deviation, pixel width, pixel height, JPEG file size in bytes, the fraction of pixels with intensity above 200, and the fraction below 10. None encodes lesion morphology, location or texture. A standardised logistic regression (`StandardScaler` → `LogisticRegression`, maximum 2,000 iterations) was fitted and evaluated by stratified five-fold cross-validation within each directory, against a most-frequent-class baseline.

## 4.7 External hold-out evaluation

The `test/` directory was never accessed by any Track A script, all four of which set the data root to `train/`. It therefore constitutes a genuine external hold-out for the fine-tuned networks. `evaluate_external.py` evaluates all saved Keras checkpoints on all 412 images, and separately on the 404-image subset remaining after removing the eight byte-identical `train/` overlaps.

---

# 5. Results

## 5.1 Effect of the metric correction

**Table 5. Track A metrics before and after correction (five-seed means, %).**

| Model | Metric | As reported | Corrected | Change |
|:---|:---|---:|---:|---:|
| ResNet50 | Accuracy | 98.40 | 98.40 | 0.00 |
| | Precision | 59.66 | 98.64 | +38.98 |
| | Recall | 59.66 | 98.62 | +38.96 |
| | F1 | 59.64 | 98.61 | +38.97 |
| | Cohen's κ | 3.94 | 96.72 | +92.78 |
| InceptionV3 | Accuracy | 98.20 | 98.20 | 0.00 |
| | Precision | 58.43 | 98.32 | +39.89 |
| | Recall | 58.62 | 98.62 | +40.00 |
| | F1 | 58.51 | 98.45 | +39.94 |
| | Cohen's κ | 1.00 | 96.30 | +95.30 |

Accuracy is unchanged in every case, as predicted.

## 5.2 Corrected Track A results

**Table 6. Track A, corrected, internal 100-image test partition, mean ± SD over five seeds (%).**

| Model | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | κ | AUROC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 98.40 ± 0.80 | 98.36 ± 0.58 | 98.64 ± 0.68 | 98.62 ± 2.01 | 98.10 ± 0.95 | 98.61 ± 0.72 | 96.72 ± 1.62 | 99.91 ± 0.12 |
| InceptionV3 | 98.20 ± 0.75 | 98.12 ± 0.85 | 98.32 ± 1.50 | 98.62 ± 1.29 | 97.62 ± 2.13 | 98.45 ± 0.64 | 96.30 ± 1.54 | 99.93 ± 0.06 |
| MobileViT | 98.40 | — | 98.66 | 98.62 | — | 98.61 | 96.72 | — |
| Swin-Tiny | 98.40 | — | 98.66 | 98.62 | — | 98.61 | 96.72 | — |

**Table 7. Per-seed corrected confusion matrices (TN / FP / FN / TP), internal test partition of 42 normal and 58 tumour images.**

| Seed | ResNet50 | InceptionV3 |
|---:|:---|:---|
| 42 | 41 / 1 / 0 / 58 | 40 / 2 / 1 / 57 |
| 7 | 41 / 1 / 1 / 57 | 42 / 0 / 1 / 57 |
| 21 | 41 / 1 / 0 / 58 | 40 / 2 / 0 / 58 |
| 99 | 42 / 0 / 3 / 55 | 42 / 0 / 2 / 56 |
| 123 | 41 / 1 / 0 / 58 | 41 / 1 / 0 / 58 |

Two observations follow. First, the architecture ranking collapses: ResNet50, MobileViT and Swin-Tiny return *identical* mean F1 (98.61) and *identical* mean κ (96.72), and InceptionV3 differs by 0.16 F1 points — roughly a fifth of the seed-to-seed standard deviation. Before correction, the same tables appeared to show transformers outperforming convolutional backbones by more than ninety κ points; that entire effect was an artefact of a single `.shuffle()` call.

Second, the confusion matrices show that each model commits between zero and three errors on a 100-image partition. At this error rate a single misclassified image is worth a full accuracy point, and no five-seed comparison can resolve architecture differences smaller than that.

## 5.3 External hold-out

**Table 8. External hold-out (`test/`, n = 412), mean ± SD over five seeds (%).**

| Model | Subset | n | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | κ | AUROC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | full | 412 | 98.74 ± 2.06 | 98.84 ± 1.88 | 97.47 ± 4.04 | 100.00 ± 0.00 | 97.69 ± 3.76 | 98.68 ± 2.14 | 97.48 ± 4.11 | 100.00 ± 0.00 |
| ResNet50 | de-duplicated | 404 | 98.71 ± 2.10 | 98.80 ± 1.95 | 97.47 ± 4.04 | 100.00 ± 0.00 | 97.60 ± 3.90 | 98.68 ± 2.14 | 97.43 ± 4.18 | 100.00 ± 0.00 |
| InceptionV3 | full | 412 | 95.29 ± 7.06 | 95.69 ± 6.46 | 92.13 ± 10.84 | 100.00 ± 0.00 | 91.38 ± 12.93 | 95.54 ± 6.40 | 90.76 ± 13.74 | 100.00 ± 0.00 |
| InceptionV3 | de-duplicated | 404 | 95.35 ± 7.14 | 95.67 ± 6.65 | 92.38 ± 10.80 | 100.00 ± 0.00 | 91.34 ± 13.29 | 95.68 ± 6.36 | 90.85 ± 13.95 | 100.00 ± 0.00 |

**Table 9. Per-seed external confusion matrices (TN / FP / FN / TP) and accuracy, 225 normal and 187 tumour images.**

| Seed | ResNet50 | Acc. | InceptionV3 | Acc. |
|---:|:---|---:|:---|---:|
| 42 | 225 / 0 / 0 / 187 | **100.00** | 213 / 12 / 0 / 187 | 97.09 |
| 7 | 203 / 22 / 0 / 187 | 94.66 | 219 / 6 / 0 / 187 | 98.54 |
| 21 | 224 / 1 / 0 / 187 | 99.76 | 225 / 0 / 0 / 187 | **100.00** |
| 99 | 225 / 0 / 0 / 187 | **100.00** | 148 / 77 / 0 / 187 | **81.31** |
| 123 | 222 / 3 / 0 / 187 | 99.27 | 223 / 2 / 0 / 187 | 99.51 |

Four properties of this table are diagnostic rather than encouraging.

**Three of the ten evaluations are exactly perfect.** ResNet50 at seeds 42 and 99, and InceptionV3 at seed 21, classify all 412 external images correctly. A flawless result on a hold-out of this size, from a model that has never seen it, is not a plausible outcome for a task on which trained radiologists disagree.

**AUROC is exactly 1.0000 in all ten model–seed evaluations.** Ten independently initialised networks achieving perfect ranking separation on 412 images is not a plausible outcome for a genuine radiological discrimination task, where inter-observer variability alone would be expected to produce overlapping score distributions. It is the expected outcome when a deterministic feature perfectly correlated with the class label is present in the input.

**Recall is exactly 100.00% in every seed.** Zero false negatives across 187 tumour images, ten runs in a row. Every error is a false positive on a normal image.

**Errors are threshold artefacts, not ranking failures.** InceptionV3 at seed 99 produces 77 false positives out of 225 normal images, reducing accuracy to 81.31%, while still achieving AUROC 1.0000. The model separates the two classes perfectly in score space and merely places the 0.5 decision boundary badly. Genuinely difficult problems do not fail in this manner; miscalibration on perfectly separable data does.

Removing the eight byte-identical overlaps between `test/` and `train/` changes every reported figure by less than 0.1 percentage points. This is not evidence of robustness. It shows that the discriminating feature is present in both directories, so `test/` functions as an external hold-out in name only: high accuracy on it confirms that the confound generalises, not that the model does.

## 5.4 Track B results

**Table 10. Frozen-feature pipelines, mean ± SD test accuracy over five seeds (%).**

| Classifier | DenseNet121 features | Xception features | Difference |
|:---|---:|---:|---:|
| SVM (RBF) | **100.00 ± 0.00** | 96.12 ± 1.27 | 3.88 |
| Random forest | **100.00 ± 0.00** | 72.43 ± 1.98 | **27.57** |
| k-NN (k = 5) | **100.00 ± 0.00** | 98.83 ± 0.24 | 1.17 |
| Bagging | **100.00 ± 0.00** | 62.43 ± 4.52 | **37.57** |
| MLP | 99.32 ± 0.95 | 88.54 ± 3.39 | 10.78 |
| AdaBoost | 99.22 ± 0.50 | 74.66 ± 2.28 | 24.56 |
| XGBoost | 98.64 ± 0.36 | 71.94 ± 1.21 | 26.70 |
| LSTM | 55.24 ± 19.71 | 45.44 ± 0.24 | — |
| Bi-LSTM | 45.44 ± 0.24 | 45.44 ± 0.24 | — |

Four DenseNet121 pipelines return exactly 100.00% with zero variance across five independent seeds. On a diagnostic task of this difficulty, an exactly perfect and perfectly stable result is a leakage signature rather than a performance result.

The backbone gap is informative in the opposite direction. Both extractors are ImageNet-pretrained and frozen; the only differences are architecture and input resolution. A genuine pancreatic signal should not swing 27.6 points for random forest or 37.6 points for bagging on that change. What plausibly differs is how each embedding preserves the global intensity and windowing statistics identified in Section 5.5 — the features that actually carry the class information here.

The recurrent baselines failed to train. Bi-LSTM under both backbones and LSTM under Xception report κ = 0.00 with accuracy equal to the tumour prevalence of the evaluation partition (45.44%), the signature of collapse onto a single predicted class. This is architecturally expected: the pipeline reshapes a pooled feature vector into a length-one sequence, which provides a recurrent layer with no temporal structure to exploit. These rows function as a negative control confirming that the pipeline reports failure when failure occurs; they are not a meaningful architecture comparison.

## 5.5 Shortcut probe

**Table 11. Class separability from non-diagnostic statistics, stratified five-fold cross-validation.**

| Feature set | `train/` (n = 999) | `test/` (n = 412) |
|:---|---:|---:|
| Majority-class baseline | 57.86 | 54.61 |
| JPEG file size alone | 65.78 | **93.70** |
| Mean pixel intensity alone | 87.59 | 46.66 |
| **All seven statistics** | **97.20** | **96.60** |
| *For reference: best fine-tuned network* | *98.40* | *98.74* |

Seven scalar quantities, computable without displaying the image, recover essentially the entire performance of a fine-tuned Swin Transformer.

**Table 12. Per-class means of the probe features.**

| Statistic | `train/` normal | `train/` tumour | `test/` normal | `test/` tumour |
|:---|---:|---:|---:|---:|
| Mean intensity | 78.32 | 42.37 | 37.12 | 37.14 |
| Intensity SD | 56.89 | 58.56 | 52.21 | 58.67 |
| Width (px) | 511.99 | 497.55 | 511.99 | 512.00 |
| Height (px) | 511.74 | 494.78 | 511.52 | 512.00 |
| Fraction of pixels > 200 | **0.000** | **0.030** | **0.000** | **0.030** |
| Fraction of pixels < 10 | 0.28 | 0.51 | 0.59 | 0.54 |
| File size (bytes) | 48,800 | 55,052 | 47,856 | 55,577 |

The saturated-pixel fraction is the most stable discriminator and is identical across directories: normal-class images contain essentially no pixels above intensity 200, while tumour-class images average 3% of their area at that brightness. Because the statistic is directory-invariant, it also explains why the external hold-out fails to expose the problem.

Which features carry the signal differs between directories, and that itself is diagnostic. Mean intensity alone reaches 87.59% within `train/` but only 46.66% within `test/`, because the mean-intensity gap between classes exists in `train/` (78.32 vs 42.37) and is absent in `test/` (37.12 vs 37.14). Conversely, file size alone reaches 93.70% in `test/` but only 65.78% in `train/`. The classes are separable by low-level acquisition properties in both directories, but by *different* low-level properties — consistent with several distinct source pipelines rather than a single systematic difference.

## 5.6 Visual characterisation of the confound

Direct inspection of the images with the highest saturated-pixel fractions in each class identifies the mechanism.

**Normal-class images** share a single rendering convention: a circular reconstruction field of view presented on a mid-grey background, with a bright horizontal band at the superior edge corresponding to the scanner table. The four highest-saturation normal images all reach only 0.006 saturated fraction and are visually homogeneous, consistent with export from one pipeline.

**Tumour-class images** are rendered on a **black** background under a different soft-tissue window, several show the scanner table edge rendered as a bright white curve at the inferior margin, and at least one — the highest-saturation image in the class, at 0.14 — carries a **burned-in white arrow annotation pointing directly at the lesion**. The two next-highest are the small RGB web-gallery images described in Section 3.2.

A classifier trained on this corpus does not need to localise a pancreatic mass. It needs to determine whether the background surrounding the reconstruction field is grey or black.

---

# 6. Discussion

## 6.1 What the benchmark measured

The three analyses converge on one explanation. The `normal` and `pancreatic_tumor` classes were assembled from different sources, exported through different pipelines, and rendered under different windowing and background conventions; part of the positive class comprises published figures, at least one carrying a burned-in diagnostic annotation. These differences are recoverable from seven scalar statistics at 96.6–97.2% accuracy. The fine-tuned networks reach 98.2–98.4% internally and 95.3–98.7% externally. The residual attributable to features a radiologist would recognise as pancreatic pathology is at most one to two percentage points and is not separable from noise given five seeds and a 100-image test partition.

The original research question is therefore not identifiable from these data. It is not that the four architectures happen to perform equivalently; it is that the corpus cannot distinguish them, because the discrimination it rewards requires none of their representational capacity.

## 6.2 Why each defect evaded routine scrutiny

**The metric defect corrupted only secondary metrics.** Accuracy — the number readers check first — remained correct throughout. Detection required noticing that 98.40% accuracy and κ = 0.0394 are mutually inconsistent: κ near zero denotes chance-level agreement, which cannot coexist with 98% accuracy on a 58/42 split. Because the corrupted metrics were *understated*, the defect also created a perverse incentive structure, making a convolutional baseline look far worse than it was and manufacturing an apparent transformer advantage that did not exist.

**Duplication is invisible to every metric.** Accuracy, F1, κ and AUROC are all computed correctly on a contaminated partition; they simply answer a different question from the one intended. Only content hashing exposes the problem, and hashing forms no part of any standard training pipeline. Nothing in the training curves, the loss trajectories or the confusion matrices signals it.

**The shortcut survives every conventional control.** It survives correct metrics, correct alignment, a genuinely untouched external directory, and de-duplication. It is defeated only by posing the adversarial question — could a model with no access to pathology do as well? — and answering it empirically. In this instance the answer took under a minute of computation.

## 6.3 The AUROC = 1.000 signature

We suggest that a perfect AUROC on a hold-out of non-trivial size should be treated as a systematic alarm. Its diagnostic value derives from the decomposition it permits: AUROC assesses ranking, accuracy assesses ranking plus threshold placement. When AUROC is 1.0000 and accuracy is 81.31% simultaneously — as for InceptionV3 at seed 99, with 77 false positives — the two decouple in a way that is only possible when the classes are perfectly separable in score space and the decision threshold is misplaced. A genuine diagnostic task with overlapping class distributions cannot produce that pattern, because misplacing the threshold on overlapping distributions degrades ranking-based and threshold-based metrics together.

Consistent recall of exactly 100.00% across ten independent evaluations is a second alarm of the same character: it indicates that the positive class is being identified by a feature that is present in every positive instance without exception, which is more characteristic of a rendering convention than of a biological finding.

## 6.4 A proposed audit protocol

The following six steps cost a few minutes of computation in total and would have detected all three defects prior to any training run.

1. **Hash the corpus by content.** Compare the file count against the distinct-digest count. Flag any digest appearing under more than one class label; these are unresolvable and must be removed or corrected before training.

2. **Partition by patient or study, never by slice.** Where identifiers are unavailable, the corpus should be treated as unsuitable for held-out evaluation and this stated explicitly as a limitation rather than worked around.

3. **Fit a shortcut baseline before the deep model.** Logistic regression on mean intensity, standard deviation, dimensions, file size and the saturated- and dark-pixel fractions. Report it in the same table as the headline result. The deep model's margin over this baseline is the only part of its score defensibly attributable to learned image understanding.

4. **Verify metric self-consistency arithmetically.** On a binary task, accuracy, class prevalence and κ mutually constrain one another. Reconstruct the confusion matrix implied by the reported metrics; if no integer-valued matrix satisfies them, at least one metric is computed incorrectly.

5. **Treat AUROC = 1.000 as blocking.** Perfect ranking on hundreds of images across independently seeded runs should halt the analysis until explained.

6. **Report specificity and balanced accuracy alongside accuracy,** and visually inspect the extreme-valued images in each class — highest and lowest mean intensity, largest and smallest file size — before accepting any result.

## 6.5 Limitations

Our correction re-evaluates saved checkpoints rather than retraining. The models were fitted on contaminated folds, so the corrected metrics describe those specific models faithfully but do not estimate uncontaminated performance. We did not retrain under a clean partition because the corpus ships no patient or study identifiers, making a defensible partition impossible to construct — a limitation which is itself among the study's findings.

Track C is implemented and unit-tested but unevaluated, so the from-scratch Vision Transformer contributes nothing to the empirical comparison.

We characterise the confound statistically and visually but cannot verify the provenance of individual images without original acquisition metadata; the attribution to differing export pipelines is an inference from format heterogeneity, filename conventions and rendering differences rather than a documented fact.

Finally, our shortcut probe establishes that the classes are separable without pathological information. It does not establish that the networks used *only* that information. Attribution methods such as Grad-CAM applied across a matched set, or retraining on intensity-normalised and background-standardised images, would be needed to quantify the residual genuine signal — and remain the natural next step.

## 6.6 Implications

Two implications extend beyond this corpus.

First, for the applied literature: an accuracy above 95% on a curated slice dataset, reported without a shortcut baseline, a content-hash audit and a patient-level partition, does not by itself distinguish a diagnostic model from a provenance classifier. The controls required to make that distinction are inexpensive and should be standard.

Second, for architecture comparison: fractional accuracy differences reported on small test partitions from artefact-bearing corpora are not measurements of architectural merit. On our 100-image partition, every model committed between zero and three errors, and one image was worth a full accuracy point. Studies of this design cannot resolve the differences they report, whatever the integrity of the underlying data.

---

# 7. Conclusion

Eleven pipelines on a pancreatic CT slice corpus produced accuracies between 98.2% and 100.00%, and none of those figures supports a claim about pancreatic tumour detection.

A metric-alignment defect rendered the agreement statistics of two models unreadable until corrected, and manufactured an apparent ninety-point transformer advantage that vanished on correction. The training pool contains 206 duplicate files among 999 images and 18 images labelled as both classes, contaminating 29–36% of every test partition. Seven non-diagnostic image statistics separate the classes at 96.6–97.2% against majority baselines near 55%, and the classes are shown to differ in windowing, background rendering, source provenance and burned-in annotation. Every external evaluation returned AUROC of exactly 1.0000 with exactly zero false negatives — the signature of a deterministic confound rather than a diagnostic model.

The corrected architecture comparison is a tie within noise, and correctly so: the corpus cannot distinguish the models, because the discrimination it rewards does not require them.

We report this as a negative result and release the audit and correction code. Establishing what these architectures can genuinely achieve for pancreatic tumour detection requires a corpus with patient-level identifiers, uniform acquisition and rendering across classes, verified content-level de-duplication, annotation-free images, a test partition large enough that a single error does not move the headline metric by a full point, and a shortcut baseline reported alongside every headline number.

---

# Data and Code Availability

The audit and correction are implemented in two standalone scripts released with this work.

`reevaluate_fixed.py` reloads each saved checkpoint, regenerates the corresponding stratified split, verifies split reproduction by accuracy fingerprint, and recomputes aligned precision, recall, specificity, F1, Cohen's κ and AUROC together with the full confusion matrix. Corrected per-seed metrics are written to `resnet50_results_CORRECTED.csv` and `inceptionv3_results_CORRECTED.csv`.

`evaluate_external.py` evaluates all saved checkpoints on the external `test/` directory, reporting metrics on the full 412-image set and on the 404-image subset remaining after removal of byte-identical overlaps with `train/`, writing results to `external_test_results.csv`.

Seeds {42, 7, 21, 99, 123} are used throughout. All twenty Track A checkpoints, per-seed confusion matrices, training curves and complete training logs are retained.

---

# References

1. Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. Shortcut learning in deep neural networks. *Nature Machine Intelligence*. 2020;2:665–673.

2. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*. 2018;15(11):e1002683.

3. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. 2023;4(9):100804.

4. Winkler JK, Fink C, Toberer F, et al. Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. *JAMA Dermatology*. 2019;155(10):1135–1141.

5. Oakden-Rayner L, Dunnmon J, Carneiro G, Ré C. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *Proceedings of the ACM Conference on Health, Inference, and Learning*. 2020:151–159.

6. Badgeley MA, Zech JR, Oakden-Rayner L, et al. Deep learning predicts hip fracture using confounding patient and healthcare variables. *npj Digital Medicine*. 2019;2:31.

7. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770–778.

8. Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception architecture for computer vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:2818–2826.

9. Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017:4700–4708.

10. Chollet F. Xception: deep learning with depthwise separable convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017:1251–1258.

11. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16×16 words: transformers for image recognition at scale. *International Conference on Learning Representations*. 2021.

12. Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, Lin S, Guo B. Swin Transformer: hierarchical vision transformer using shifted windows. *Proceedings of the IEEE International Conference on Computer Vision*. 2021:10012–10022.

13. Mehta S, Rastegari M. MobileViT: light-weight, general-purpose, and mobile-friendly vision transformer. *International Conference on Learning Representations*. 2022.

14. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I. Attention is all you need. *Advances in Neural Information Processing Systems*. 2017:5998–6008.

15. Cohen J. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*. 1960;20(1):37–46.

16. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794.

17. Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022;13:4128.

18. Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digital Medicine*. 2022;5:48.
