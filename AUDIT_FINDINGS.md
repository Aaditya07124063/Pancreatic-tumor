# Response to review comments — pancreatic CT classification

Status: full 6-model protocol re-running as of 2026-08-24 (see "Protocol now running").
This document answers each review point with the evidence behind it.

---

## 1. "Have you used pre-trained models for all of these?"

Yes — all four backbones are ImageNet-pretrained and fine-tuned end to end.

| Model | Weights source | Fine-tuning | Location |
|:--|:--|:--|:--|
| ResNet50 | `weights="imagenet"` (Keras) | all layers trainable | `resnet50_train.py:75` |
| InceptionV3 | `weights="imagenet"` (Keras) | all layers trainable | `inceptionv3_train.py:76` |
| MobileViT | `from_pretrained` (HF `apple/mobilevit-small`) | full fine-tune | `mobilevit_train.py:84` |
| Swin-Tiny | `from_pretrained` (HF `microsoft/swin-tiny-patch4-window7-224`) | full fine-tune | `swin_transformer_train.py:83` |

## 2. "The correct dataset splitting ratio should be 80:10:10, right?"

Agreed, and that is what is implemented — **stratified** 80/10/10, class balance preserved
in every partition (`data_utils.py:stratified_split`).

On the deduplicated corpus (n = 1,179) this gives:

```
Train: 943 (80.0%)  normal=496 | tumor=447
Val:   118 (10.0%)  normal=62  | tumor=56
Test:  118 (10.0%)  normal=62  | tumor=56
```

## 3. "It looks like the results shown are from Random Forest."

Two separate tracks exist in this repository; the Random Forest numbers belong to the second:

- **Track A — end-to-end fine-tuning** (`*_train.py`): the four pretrained backbones above.
  These are the headline architecture comparison.
- **Track B — frozen features + classical classifiers** (`feature_extraction_pipeline.py`):
  Xception / DenseNet121 as frozen extractors feeding SVM, **Random Forest**, AdaBoost, KNN,
  XGBoost, Bagging, ANN, LSTM, Bi-LSTM.

If a Random Forest row was read as a deep-model result, it came from Track B. The two tracks
are reported separately from here on.

## 4. "100% accuracy is not realistic — there may be data leakage."

**Correct, and the cause is now identified. It is worse than a split problem: stratified
splitting cannot fix it.** Three independent defects, in increasing order of severity:

### 4a. Exact duplicate files
Content-hash (MD5) audit of all 1,411 files:
- **196 byte-identical duplicate files** removed
- **18 image hashes carrying contradictory class labels** (the same image filed as both
  `normal` and `pancreatic_tumor`) — excluded entirely
- Corpus after dedup: **1,179 images** (620 normal / 559 tumor)

### 4b. Near-duplicate (slice-level) leakage — survives MD5 dedup
CT studies yield contiguous axial slices that are near-identical but not byte-identical, so
MD5 misses them. A 64-bit perceptual hash (dHash) on the *already deduplicated* corpus shows
the fraction of test images having a near-twin in their own training partition:

| dHash Hamming distance | test images with a near-duplicate in train |
|---:|---:|
| = 0 (identical hash) | **66.1%** |
| ≤ 2 | 96.4% |
| ≤ 5 | 98.5% |

Averaged over the 5 seeds. Random stratified splitting *guarantees* this, because it splits at
slice level rather than patient/study level. The standard remedy is patient- or study-level
partitioning, which this corpus carries no metadata to support.

### 4c. The decisive defect — the classes are separable by a rendering artefact
Seven **non-diagnostic** scalar statistics were extracted per image (mean intensity, intensity
SD, width, height, file size, fraction of pixels > 200, fraction < 10). None encodes lesion
morphology, location or texture.

One of them separates the classes perfectly:

| class | `fraction of pixels > 200` (min – max) |
|:--|:--|
| normal | 0.00001 – **0.00613** |
| pancreatic_tumor | **0.01003** – 0.13960 |

The ranges do not overlap. **A single threshold at ≈0.008 classifies all 1,179 images with
100% accuracy, on every seed, with zero variance** (majority-class baseline: 52.5%).

This means the two classes were windowed or rendered differently before assembly. Any
classifier — deep or trivial — reaches 100% by reading brightness, without ever looking at
the pancreas. Confirming this: an untrained 4-layer CNN written from scratch reaches
**100% test accuracy in one epoch, in 15 seconds**.

**Consequence:** the reported accuracies measure acquisition provenance, not pathology.
Increasing seeds, epochs, or split rigour cannot repair this, because the confound is inside
the images themselves.

---

## Protocol now running

Every review request is implemented; the numbers are being regenerated from scratch:

- deduplicated corpus (1,179 images), **stratified 80:10:10**
- **5 seeds** (42, 7, 21, 99, 123), mean ± SD reported
- **50 epochs** per seed, `PATIENCE=50` so early stopping cannot cut the budget short
  (best-validation weights still restored for evaluation)
- **6 models**: ResNet50, InceptionV3, MobileViT, Swin-Tiny (pretrained) plus
  **`ScratchCNN`** (from-scratch CNN) and **`VisionTransformer`** (from-scratch ViT: hand-written multi-head attention, patch embedding, CLS token — `new transformer scratch/models/`)
- the shortcut floor (100%) is reported in the same table, so every model is read against
  what a non-diagnostic rule already achieves — not against 50%

Reproduce with:

```
./run_all.sh
```

## Recommendation

The architecture comparison is **not identifiable from this corpus**. Before these numbers can
support any clinical claim, the dataset must be rebuilt: uniform windowing applied to all
images regardless of class, patient-level (not slice-level) partitioning, and provenance
recorded per image. Until then the honest reporting of this work is as a negative result plus
the audit protocol in `leakage_audit.py`, which detects all three defects in under a minute.
