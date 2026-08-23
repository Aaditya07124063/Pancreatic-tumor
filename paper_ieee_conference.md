---
title: "When 98% Means Nothing: Shortcut Learning and Evaluation Failures in Deep Learning for Pancreatic Tumour Classification on CT"
author: "Aaditya Adhikari"
date: "August 2026"
---

# Abstract

We benchmark eleven deep and classical learning pipelines — ResNet50, InceptionV3, MobileViT and Swin-Tiny fine-tuned end-to-end, plus Xception and DenseNet121 frozen-feature extractors feeding seven downstream classifiers — on a 1,411-image binary pancreatic CT dataset, under a five-seed stratified protocol. Every fine-tuned network reaches 98.2–98.4% test accuracy, and four frozen-feature classifiers reach exactly 100.00%. We then show that these numbers measure almost nothing about pancreatic pathology.

Three findings support this. First, a metric-alignment defect in two of the training scripts caused precision, recall, F1 and Cohen's κ to be computed against shuffled predictions, reporting chance-level scores (κ ≈ 0.01–0.04) alongside 98% accuracy; we diagnose it, correct it against the saved checkpoints, and recover the true values (κ = 0.967 and 0.963). Second, the training pool contains 206 byte-identical duplicate files among 999 images (20.6%), including 18 image hashes labelled as *both* classes; under the random per-slice split used, 29–36% of every held-out test set is a byte-identical copy of a training image. Third, and decisively, a logistic regression on seven non-diagnostic image statistics — mean intensity, standard deviation, width, height, JPEG file size, and the fractions of saturated and near-black pixels — achieves 97.20% cross-validated accuracy on the training pool and 96.60% on the external folder, against majority baselines of 57.86% and 54.61%. The two classes are rendered with different windowing and background conventions, part of the tumour class is web-sourced imagery including at least one burned-in arrow annotation, and every one of ten model–seed evaluations on the external hold-out returns an AUROC of exactly 1.0000 with exactly zero false negatives.

We conclude that the dataset is separable by acquisition provenance rather than by disease, that the reported architecture ranking is not identifiable from this data, and that the correct deliverable from this study is a negative result plus a reusable audit protocol. We release the audit and re-evaluation code.

**Index Terms** — shortcut learning, dataset leakage, pancreatic cancer, computed tomography, reproducibility, vision transformers, medical image classification.

---

# I. Introduction

Pancreatic ductal adenocarcinoma has among the worst five-year survival rates of any solid tumour, and the gap between early- and late-stage survival is large enough that automated detection on routine abdominal CT is an attractive target. A substantial literature now reports accuracies above 95% for binary pancreatic tumour classification on curated slice datasets, and it is common to see near-ceiling numbers presented as evidence that a given architecture is well suited to the task.

This paper began as one of those studies. The objective was a straightforward architecture comparison: convolutional backbones against transformer backbones, end-to-end fine-tuning against frozen-feature extraction, on a single pancreatic CT slice dataset, under an identical five-seed protocol. All eleven pipelines were trained, all reached the expected accuracy range, and the results table looked publishable.

It was not. Working through three separate consistency checks — an arithmetic inconsistency between reported metrics, a content-hash audit of the image files, and a shortcut probe using only non-diagnostic image statistics — revealed that the benchmark measures dataset provenance rather than pathology. This paper reports that finding, because the failure modes involved are common, are individually easy to miss, and are collectively sufficient to invalidate a result that would otherwise pass casual review.

Our contributions are:

1. **A reproducible metric-misalignment defect and its correction.** Applying `tf.data` shuffling to an evaluation pipeline while computing metrics against the original label array yields accuracy that is correct and precision/recall/F1/κ that are chance-level. We show the resulting signature (98% accuracy with κ ≈ 0), diagnose it, and recompute correct metrics from the saved checkpoints, verifying the split reproduction by accuracy fingerprint.

2. **A content-level dataset audit.** MD5 hashing exposes 206 duplicate files in a 999-image pool, 18 hashes carrying contradictory labels, and 29–36% per-seed test-set contamination under random slice-level splitting.

3. **A shortcut-learning demonstration.** Seven trivially computable, non-diagnostic image statistics reproduce nearly the whole reported performance of a Swin Transformer, and the classes are shown to differ systematically in windowing, background, file size and burned-in annotation.

4. **A practical audit protocol** that costs a few minutes of compute and would have caught all three problems before any model was trained.

---

# II. Related Work

**Transfer learning for pancreatic CT.** ImageNet-pretrained convolutional backbones — ResNet, Inception, DenseNet, Xception, EfficientNet — dominate reported pipelines for pancreatic lesion classification, typically fine-tuned end-to-end or used as frozen feature extractors with a downstream classical classifier. Vision transformers and hybrid convolution–attention models, notably Swin and MobileViT, have more recently been applied to the same task, generally with comparable or marginally superior reported accuracy.

**Shortcut learning.** Geirhos et al. characterised the tendency of deep networks to solve tasks via unintended decision rules that satisfy the training objective without capturing the intended concept. In medical imaging the phenomenon is well documented: models have been shown to recover scanner manufacturer, hospital site, patient sex and even the presence of chest drains rather than the target pathology, and to transfer poorly when the confound is removed. Zech et al. demonstrated that pneumonia classifiers on chest radiographs learned hospital-specific markers; Winkler et al. showed that surgical skin markings inflated melanoma classifier confidence. Our finding is the pancreatic-CT instance of this family, with the additional property that the confound is measurable in seven scalar statistics.

**Leakage and evaluation integrity.** Kapoor and Narayanan catalogue leakage as a leading cause of irreproducible machine-learning claims across scientific fields, with train–test contamination the most common variant. In volumetric medical imaging the characteristic form is slice-level rather than patient-level splitting: adjacent axial slices from the same study are near-duplicates, so a random split over slices places nearly identical images on both sides of the partition. Our dataset exhibits both this and the stronger form — exact byte-identical file duplication.

**What is new here.** Prior work has documented these failure modes individually. The contribution of this paper is to show all three co-occurring in a single conventional benchmarking study, to quantify each, and to demonstrate that the resulting architecture comparison is not identifiable — the differences between four architectures fall entirely inside the seed-to-seed noise band that the artefacts induce.

---

# III. Dataset and Audit

## A. Composition

The dataset is distributed as two directories, `train/` and `test/`, each with `normal/` and `pancreatic_tumor/` subdirectories.

**TABLE I — DATASET COMPOSITION**

| Directory | Normal | Pancreatic tumour | Total |
|:---|---:|---:|---:|
| `train/` | 421 | 578 | 999 |
| `test/` | 225 | 187 | 412 |
| **Total** | **646** | **765** | **1,411** |

The dominant format is 512×512 8-bit greyscale JPEG (964 of 999 training images, 411 of 412 external images). The remainder — 34 tumour images and one normal image — are small RGB JPEGs at heterogeneous resolutions (250×202, 256×197, 247×204, 262×192, 235×233 and others). Several filenames in the tumour class follow web-gallery conventions (`…_big_gallery.jpeg`) or carry duplication suffixes from a desktop file manager (`… - Copy - Copy (4).jpg`), indicating that part of the positive class was assembled from published figures rather than exported from a DICOM archive.

## B. Duplicate and label-conflict audit

We computed an MD5 digest over the bytes of every image file.

**TABLE II — CONTENT-HASH AUDIT**

| Quantity | Value |
|:---|---:|
| Files in `train/` | 999 |
| Distinct image hashes in `train/` | 793 |
| Redundant copies | 206 (20.6%) |
| Hashes appearing under *both* class labels | 18 (36 files) |
| `test/` images byte-identical to a `train/` image | 8 |

The multiplicity distribution over training hashes is 600 singletons, 187 hashes appearing twice, 4 three times, one six times and one seven times. The 18 hashes carrying contradictory labels are the more serious defect: identical pixel data is presented to the network as `normal` in one directory and `pancreatic_tumor` in the other, guaranteeing an irreducible error floor and indicating that class assignment was not verified at ingestion.

## C. Consequences for the split protocol

All four end-to-end training scripts draw an 80/10/10 stratified split from `train/` alone, at the level of individual image files, reseeded per run. With 206 redundant copies distributed at random, a large fraction of each held-out partition has an exact twin in its own training fold.

**TABLE III — PER-SEED TEST CONTAMINATION**

| Seed | Train | Val | Test | Test images with a byte-identical twin in train | Val images likewise |
|---:|---:|---:|---:|---:|---:|
| 42 | 799 | 100 | 100 | 29 | 37 |
| 7 | 799 | 100 | 100 | 33 | 34 |
| 21 | 799 | 100 | 100 | 36 | 29 |
| 99 | 799 | 100 | 100 | 33 | 36 |
| 123 | 799 | 100 | 100 | 31 | 34 |

Roughly one third of every reported test set was memorisable verbatim. This is a lower bound on contamination: it counts only byte-identical files and ignores near-duplicates. The filename structure — contiguous runs such as `1-001…1-239`, `22 (1)…22 (119)`, `23 (102)…` — indicates that the corpus is drawn from a small number of contiguous axial slice series. Adjacent slices within a series are visually near-identical, so slice-level random splitting distributes the same anatomy across train, validation and test even where the files differ. Patient- or series-level partitioning is required; no patient identifiers are distributed with the data, which is itself a reason to treat any result from this corpus as provisional.

## D. Shortcut probe

To test whether the classes are separable without any pathological information, we extracted seven scalar statistics per image: mean intensity, intensity standard deviation, pixel width, pixel height, JPEG file size in bytes, the fraction of pixels with intensity above 200, and the fraction below 10. None encodes lesion morphology. We fitted a standardised logistic regression and evaluated by five-fold cross-validation.

**TABLE IV — CLASS SEPARABILITY FROM NON-DIAGNOSTIC STATISTICS**

| Feature set | `train/` (n=999) | `test/` (n=412) |
|:---|---:|---:|
| Majority-class baseline | 57.86% | 54.61% |
| JPEG file size alone | 65.78% | 93.70% |
| Mean pixel intensity alone | 87.59% | 46.66% |
| All seven statistics | **97.20%** | **96.60%** |

Seven numbers computable without opening the image in a viewer recover essentially the entire performance of the fine-tuned networks. The underlying class differences are large and systematic:

**TABLE V — PER-CLASS IMAGE STATISTICS (MEANS)**

| Statistic | `train/` normal | `train/` tumour | `test/` normal | `test/` tumour |
|:---|---:|---:|---:|---:|
| Mean intensity | 78.32 | 42.37 | 37.12 | 37.14 |
| Intensity SD | 56.89 | 58.56 | 52.21 | 58.67 |
| Fraction > 200 | 0.000 | 0.030 | 0.000 | 0.030 |
| Fraction < 10 | 0.28 | 0.51 | 0.59 | 0.54 |
| File size (bytes) | 48,800 | 55,052 | 47,856 | 55,577 |

The saturated-pixel fraction is the clearest signal and is stable across both directories: normal images contain essentially no pixels above intensity 200, while tumour images average 3% of their area at that level. Visual inspection of the extreme cases identifies the source. Normal-class images share a single rendering convention — a circular reconstruction field on a mid-grey background, consistent with one export pipeline. Tumour-class images use a different soft-tissue window on a black background, several carry a bright rendering of the scanner table edge, and at least one carries a burned-in white arrow annotation pointing directly at the lesion. A classifier need only decide whether the background is grey or black.

---

# IV. Experimental Setup

## A. Track A — end-to-end fine-tuning

Four ImageNet-pretrained backbones were fine-tuned with all layers trainable on `train/` under an 80/10/10 stratified split (799/100/100; the test partition contains 58 tumour and 42 normal images), repeated over seeds {42, 7, 21, 99, 123}.

**TABLE VI — TRACK A CONFIGURATIONS**

| Model | Framework | Input | Optimiser | LR | Head |
|:---|:---|:---|:---|---:|:---|
| ResNet50 | TensorFlow/Keras | 224² | Adam | 1e-4 | GAP → BN → Drop .4 → Dense 256 → Drop .3 → sigmoid |
| InceptionV3 | TensorFlow/Keras | 299² | Adam | 1e-4 | as above |
| MobileViT | PyTorch / HF | 224² | AdamW (wd 0.01) | 5e-5 | 2-way linear |
| Swin-Tiny | PyTorch / HF | 224² | AdamW (wd 0.01) | 5e-5 | 2-way linear |

Maximum 25 epochs, batch size 32, early stopping on validation accuracy with patience 10 and best-weight restoration, plateau-based learning-rate reduction. All dispersion figures in this paper are population standard deviations over the five seed values (`ddof = 0`), the five seeds being the complete set of runs performed. Augmentation comprised horizontal and vertical flips, rotation, brightness jitter and random resized cropping. PyTorch runs used Apple MPS acceleration; gradients were clipped to unit norm.

## B. Track B — frozen features plus downstream classifiers

Xception (299², global max pooling) and DenseNet121 (224², global max pooling) were used as frozen ImageNet feature extractors. Features from all 999 `train/` images were used for fitting; the 412-image `test/` directory was split 50/50 into 206 validation and 206 test images per seed. Seven downstream classifiers were trained: SVM (RBF), random forest, AdaBoost, k-NN (k=5), bagging, a two-layer MLP (128–64), and XGBoost, each with 100 estimators where applicable. Two recurrent classifiers, LSTM(64) and Bi-LSTM(64), were additionally trained on the feature vectors reshaped to a single timestep.

## C. Track C — from-scratch transformer

A complete sequence-to-sequence Transformer and Vision Transformer were implemented in pure PyTorch: scaled dot-product multi-head attention with masking, sinusoidal positional encoding, pre-layer-norm encoder and decoder blocks with cross-attention, patch embedding, a learnable classification token and learnable positional embeddings, and an MLP head. Shape, masking and gradient-flow unit tests pass. No trained result artefacts were produced, so this track is reported as implemented but unevaluated.

## D. Metric-alignment defect and correction

Both TensorFlow scripts construct every partition, including test, through a helper that terminates in

```
ds = ds.shuffle(buffer_size=len(paths), seed=seed).batch(batch_size)
```

`model.evaluate` is unaffected, because each batch carries its own labels; accuracy is therefore correct. However

```
y_prob = model.predict(test_ds).ravel()   # shuffled order
prec   = precision_score(te_l, y_pred)    # original order
```

compares each prediction against the label of a different image. On a 100-image partition with 58 positives, random alignment yields precision ≈ 58/59, recall ≈ 58/58 in expectation over a randomly permuted assignment — that is, ≈ 0.58 for both — and κ ≈ 0. The published tables show exactly this: 98% accuracy beside κ of 0.0394 and 0.0100.

We corrected this by reloading each saved checkpoint, regenerating the identical split, and predicting from an unshuffled array. Faithful split reproduction was verified by accuracy fingerprint: because accuracy was never corrupted, agreement between the recomputed and originally logged accuracy confirms that the same images were evaluated. All ten model–seed combinations matched exactly.

---

# V. Results

## A. Corrected Track A results

**TABLE VII — EFFECT OF THE CORRECTION (5-SEED MEANS, INTERNAL 100-IMAGE TEST SPLIT)**

| Model | Metric | As originally reported | Corrected |
|:---|:---|---:|---:|
| ResNet50 | Precision | 59.66 | **98.64** |
| | Recall | 59.66 | **98.62** |
| | F1 | 59.64 | **98.61** |
| | Cohen's κ | 3.94 | **96.72** |
| InceptionV3 | Precision | 58.43 | **98.32** |
| | Recall | 58.62 | **98.62** |
| | F1 | 58.51 | **98.45** |
| | Cohen's κ | 1.00 | **96.30** |

Accuracy was unaffected and is unchanged.

**TABLE VIII — TRACK A, CORRECTED, MEAN ± SD OVER FIVE SEEDS (%)**

| Model | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | κ | AUROC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 98.40 ± 0.80 | 98.36 ± 0.58 | 98.64 ± 0.68 | 98.62 ± 2.01 | 98.10 ± 0.95 | 98.61 ± 0.72 | 96.72 ± 1.62 | 99.91 ± 0.12 |
| InceptionV3 | 98.20 ± 0.75 | 98.12 ± 0.85 | 98.32 ± 1.50 | 98.62 ± 1.29 | 97.62 ± 2.13 | 98.45 ± 0.64 | 96.30 ± 1.54 | 99.93 ± 0.06 |
| MobileViT | 98.40 | — | 98.66 | 98.62 | — | 98.61 | 96.72 | — |
| Swin-Tiny | 98.40 | — | 98.66 | 98.62 | — | 98.61 | 96.72 | — |

Once corrected, the four architectures are indistinguishable. ResNet50, MobileViT and Swin-Tiny return identical mean F1 (98.61) and identical κ (96.72); InceptionV3 differs by 0.16 F1 points, an order of magnitude below the seed-to-seed standard deviation. Inspection of the per-seed confusion matrices explains why: across all seeds, the models commit between zero and three errors on a 100-image partition, and largely the same ones. There is no statistical basis for ranking these architectures on this data.

## B. External hold-out and the AUROC signature

The `test/` directory was never touched by Track A training. Evaluating the saved checkpoints on all 412 images gives:

**TABLE IX — EXTERNAL HOLD-OUT (`test/`, n=412), MEAN ± SD OVER FIVE SEEDS (%)**

| Model | Subset | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | κ | AUROC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | full | 98.74 ± 2.06 | 98.84 ± 1.88 | 97.47 ± 4.04 | 100.00 ± 0.00 | 97.69 ± 3.76 | 98.68 ± 2.14 | 97.48 ± 4.11 | 100.00 ± 0.00 |
| ResNet50 | de-duplicated | 98.71 ± 2.10 | 98.80 ± 1.95 | 97.47 ± 4.04 | 100.00 ± 0.00 | 97.60 ± 3.90 | 98.68 ± 2.14 | 97.43 ± 4.18 | 100.00 ± 0.00 |
| InceptionV3 | full | 95.29 ± 7.06 | 95.69 ± 6.46 | 92.13 ± 10.84 | 100.00 ± 0.00 | 91.38 ± 12.93 | 95.54 ± 6.40 | 90.76 ± 13.74 | 100.00 ± 0.00 |
| InceptionV3 | de-duplicated | 95.35 ± 7.14 | 95.67 ± 6.65 | 92.38 ± 10.80 | 100.00 ± 0.00 | 91.34 ± 13.29 | 95.68 ± 6.36 | 90.85 ± 13.95 | 100.00 ± 0.00 |

Four features of this table are diagnostic rather than encouraging.

**Three of the ten evaluations are exactly perfect.** ResNet50 at seeds 42 and 99, and InceptionV3 at seed 21, classify all 412 external images correctly. Flawless performance on a hold-out of this size, from models that never saw it, is not a plausible outcome for a task on which trained radiologists disagree.

**AUROC is exactly 1.0000 in all ten model–seed evaluations.** Perfect ranking separation of 412 images by ten independently initialised models is not a plausible outcome for a genuine radiological task; it is the expected outcome when a deterministic, perfectly class-correlated feature is present in the input.

**Recall is exactly 100.00% in every seed** — zero false negatives across 187 tumour images, ten times over. Every error is a false positive on a normal image.

**Errors are threshold artefacts, not ranking failures.** InceptionV3 at seed 99 produces 77 false positives out of 225 normals, dropping accuracy to 81.31%, while still achieving AUROC 1.0000. The model separates the classes perfectly and merely places the 0.5 decision boundary badly. A genuinely difficult task does not fail this way.

Removing the eight byte-identical `test/`–`train/` overlaps changes every figure by less than 0.1 points. This is not reassurance: it shows the shortcut is present in both directories, so the external hold-out is external in name only. High external accuracy here confirms that the confound generalises, not that the model does.

## C. Track B results

**TABLE X — FROZEN-FEATURE PIPELINES, MEAN TEST ACCURACY OVER FIVE SEEDS (%)**

| Classifier | DenseNet121 features | Xception features |
|:---|---:|---:|
| SVM (RBF) | **100.00 ± 0.00** | 96.12 ± 1.27 |
| Random forest | **100.00 ± 0.00** | 72.43 ± 1.98 |
| k-NN (k=5) | **100.00 ± 0.00** | 98.83 ± 0.24 |
| Bagging | **100.00 ± 0.00** | 62.43 ± 4.52 |
| MLP | 99.32 ± 0.95 | 88.54 ± 3.39 |
| AdaBoost | 99.22 ± 0.50 | 74.66 ± 2.28 |
| XGBoost | 98.64 ± 0.36 | 71.94 ± 1.21 |
| LSTM | 55.24 ± 19.71 | 45.44 ± 0.24 |
| Bi-LSTM | 45.44 ± 0.24 | 45.44 ± 0.24 |

Four DenseNet121 pipelines return exactly 100.00% with zero variance across five seeds. On any real diagnostic task this is a leakage signature, not a result.

The gap between the two backbones is instructive in the opposite direction. Both are ImageNet-pretrained and frozen; only the input resolution and architecture differ. Yet random forest scores 100.00% on DenseNet121 features and 72.43% on Xception features. A genuine pancreatic signal would not swing 28 points on that change. What differs is how cleanly each embedding preserves the global intensity and windowing statistics that carry the confound.

The recurrent baselines failed to train: κ = 0.00 for Bi-LSTM under both backbones and for LSTM under Xception, with accuracy equal to the tumour prevalence of the evaluation partition (45.44%), the signature of collapse onto a single class. Presenting a pooled feature vector to a recurrent layer as a length-one sequence provides no temporal structure to model; these rows should be read as a negative control confirming that the pipeline reports failure when failure occurs, not as an architecture comparison.

---

# VI. Discussion

## A. What the benchmark actually measured

Taken together, the audit results admit one consistent explanation. The `normal` and `pancreatic_tumor` classes were assembled from different sources, exported through different pipelines, and rendered with different windowing and background conventions; part of the positive class consists of published figures, at least one of which carries a burned-in arrow pointing at the finding. These differences are recoverable from seven scalar statistics at 96–97% accuracy. The fine-tuned networks reach 98.2–98.4%. The residual attributable to anything a radiologist would recognise as pancreatic pathology is at most one to two percentage points, and is not separable from noise given five seeds and a 100-image test partition.

The architecture comparison that motivated the study is therefore not identifiable. It is not that ResNet50 and Swin-Tiny happen to tie; it is that the data cannot distinguish them, because both are solving a task that neither was needed for.

## B. Why each check was individually insufficient

Each of the three defects survives the checks normally applied.

The metric misalignment produced 98% accuracy — the headline number — and only corrupted the secondary metrics. It would have passed any review that read the accuracy column first. It was caught only by noticing that 98% accuracy and κ = 0.01 cannot coexist: κ near zero means chance-level agreement, which is incompatible with 98% accuracy on a 58/42 split.

The duplication is invisible to every metric. Accuracy, F1, κ and AUROC are all computed correctly on a contaminated split; they simply answer a different question than intended. Only content hashing exposes it, and hashing is not part of any standard training pipeline.

The shortcut survives correct metrics, correct alignment, a clean external hold-out, and de-duplication. It is defeated only by asking the adversarial question — can a model with no access to pathology do just as well? — and answering it empirically.

## C. A minimal audit protocol

The following costs a few minutes of compute and would have caught all three defects before any training run.

1. **Hash the corpus.** Compare file count with distinct-digest count. Flag any digest appearing under more than one label.
2. **Split by patient or series, never by slice.** Where identifiers are absent, treat the dataset as unsuitable for held-out evaluation and say so.
3. **Fit a shortcut baseline.** Logistic regression on mean, SD, dimensions, file size and saturated/dark-pixel fractions. If it approaches the deep model, the deep model's margin over it is the only defensible claim.
4. **Check metric arithmetic for self-consistency.** For binary tasks, accuracy, prevalence and κ constrain one another. Reconstruct the confusion matrix from the reported numbers; if no integer matrix satisfies them, the metrics are wrong.
5. **Treat AUROC = 1.000 as an alarm.** Perfect ranking on hundreds of images across independent seeds indicates a deterministic confound.
6. **Report specificity and balanced accuracy** alongside accuracy, and inspect the extreme-value images in each class before believing any result.

## D. Limitations

Our correction re-evaluates saved checkpoints rather than retraining; the models themselves were fitted on contaminated folds, so the corrected metrics describe those models faithfully but do not estimate uncontaminated performance. We did not retrain under a clean partition because no patient or series identifiers accompany the data, making a defensible partition impossible to construct — which is itself the finding. Track C is implemented but unevaluated. Finally, we characterise the confound statistically and visually but cannot verify the provenance of individual images without the original acquisition metadata.

---

# VII. Conclusion

Eleven pipelines on a pancreatic CT dataset produce accuracies between 98.2% and 100.00%, and none of those numbers supports a claim about pancreatic tumour detection. A metric-alignment defect made two models' agreement statistics unreadable until corrected; 20.6% of the training pool is duplicated, contaminating 29–36% of every test partition; and seven non-diagnostic image statistics separate the classes at 96.6–97.2%, against majority baselines near 55%. Every external evaluation returns AUROC exactly 1.0000 with zero false negatives — the signature of a deterministic confound rather than a diagnostic model.

The corrected architecture comparison is a tie within noise, and correctly so: the data cannot distinguish the models because the task, as posed, does not require them. We therefore report this as a negative result and release the audit and re-evaluation code. Establishing what these architectures can genuinely do for pancreatic tumour detection requires a corpus with patient-level identifiers, uniform acquisition and rendering across classes, verified de-duplication, and a shortcut baseline reported alongside every headline number.

---

# Reproducibility

The audit and correction are implemented in two standalone scripts released with this work: `reevaluate_fixed.py`, which reloads the saved checkpoints, regenerates each split, verifies reproduction by accuracy fingerprint and recomputes aligned metrics; and `evaluate_external.py`, which evaluates all checkpoints on the external directory with and without the eight byte-identical overlaps. Corrected per-seed metrics are written to `resnet50_results_CORRECTED.csv`, `inceptionv3_results_CORRECTED.csv` and `external_test_results.csv`. Seeds {42, 7, 21, 99, 123} are used throughout.

---

# References

[1] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, pp. 665–673, 2020.

[2] J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Medicine*, vol. 15, no. 11, e1002683, 2018.

[3] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, 100804, 2023.

[4] J. K. Winkler et al., "Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition," *JAMA Dermatology*, vol. 155, no. 10, pp. 1135–1141, 2019.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[6] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in *Proc. IEEE CVPR*, 2016, pp. 2818–2826.

[7] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE ICCV*, 2021, pp. 10012–10022.

[8] S. Mehta and M. Rastegari, "MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer," in *Proc. ICLR*, 2022.

[9] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.

[10] F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in *Proc. IEEE CVPR*, 2017, pp. 1251–1258.

[11] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998–6008.

[13] J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.

[14] L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Ré, "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging," in *Proc. ACM CHIL*, 2020, pp. 151–159.
