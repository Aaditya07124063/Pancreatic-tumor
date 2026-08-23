"""
reevaluate_fixed.py
===================
Corrects the metric-misalignment bug in resnet50_train.py and inceptionv3_train.py.

THE BUG
-------
In both scripts, `make_dataset()` applies `.shuffle()` to EVERY split, including
the test split:

    ds = ds.shuffle(buffer_size=len(paths), seed=seed).batch(...)   # <-- line 81

`model.evaluate(test_ds)` is unaffected (accuracy is computed per-batch against
the labels carried inside the same batch), which is why test accuracy correctly
reads ~98%.

But then:

    y_prob = model.predict(test_ds).ravel()     # order = SHUFFLED
    prec   = precision_score(te_l, y_pred)      # order = ORIGINAL

`te_l` is the un-shuffled label array. So y_pred[i] is compared against the label
of a *different* image. The result is a chance-level score: with 58 positives in
a 100-image test set, random alignment yields precision ~= recall ~= 58%, and
Cohen's Kappa ~= 0. That is exactly what the original results show.

THE FIX
-------
Build the test pipeline with NO shuffle, and derive y_true from the same pipeline
that produced y_pred. Everything else (preprocessing, split, seeds) is identical
to the original script so the reloaded checkpoints see the exact same test images.

SELF-CHECK
----------
The script recomputes test accuracy from the aligned predictions and compares it
against the accuracy recorded in the original results CSV. If the two agree, the
train/val/test split was reproduced faithfully and the corrected precision /
recall / F1 / Kappa can be trusted. If they disagree, the script says so loudly
rather than silently emitting wrong numbers.

USAGE
-----
    python reevaluate_fixed.py                  # both models, all seeds
    python reevaluate_fixed.py --model resnet50
    python reevaluate_fixed.py --model inceptionv3 --seeds 42 7
"""

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inception

from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = "./train"
CLASSES = ["normal", "pancreatic_tumor"]   # label 0 = normal, 1 = tumor
SEEDS = [42, 7, 21, 99, 123]
BATCH_SIZE = 32

MODEL_SPECS = {
    "resnet50": {
        "img_size": (224, 224),
        "preprocess": pp_resnet,
        "outdir": "./resnet50_outputs",
        "ckpt": "resnet50_seed{seed}.keras",
        "orig_csv": "resnet50_results.csv",
        "pretty": "ResNet50",
    },
    "inceptionv3": {
        "img_size": (299, 299),
        "preprocess": pp_inception,
        "outdir": "./inceptionv3_outputs",
        "ckpt": "inceptionv3_seed{seed}.keras",
        "orig_csv": "inceptionv3_results.csv",
        "pretty": "InceptionV3",
    },
}


# ─── DATA (identical to the original scripts) ─────────────────────────────────
def load_dataset(data_dir, classes):
    """NOTE: os.listdir order is deliberately NOT sorted here, to match the
    original training scripts exactly. Sorting would produce a different split
    and the reloaded checkpoints would then be evaluated on images they were
    trained on."""
    paths, labels = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return np.array(paths), np.array(labels)


def load_images(paths, img_size):
    return np.array([img_to_array(load_img(p, target_size=img_size)) for p in paths],
                    dtype="float32")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_test_split(seed, all_paths, all_labels):
    """Reproduces the 80/10/10 stratified split from the training scripts."""
    set_seed(seed)
    _, tmp_p, _, tmp_l = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=seed)
    _, te_p, _, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed)
    return te_p, te_l


# ─── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate_seed(spec, seed, all_paths, all_labels):
    te_p, te_l = get_test_split(seed, all_paths, all_labels)

    ckpt = os.path.join(spec["outdir"], spec["ckpt"].format(seed=seed))
    if not os.path.exists(ckpt):
        print(f"    [SKIP] checkpoint not found: {ckpt}")
        return None

    model = tf.keras.models.load_model(ckpt, compile=False)

    X = spec["preprocess"](load_images(te_p, spec["img_size"]))

    # THE FIX: no shuffle, no tf.data reordering. Straight array in, array out.
    y_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = te_l.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec_score = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "seed": seed,
        "n_test": len(y_true),
        "n_pos": int(y_true.sum()),
        "test_acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec_score,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def check_against_original(spec, rows):
    """Confirms the split was reproduced by comparing recomputed test accuracy
    against the accuracy logged by the original (buggy-metrics) run. Accuracy
    was NOT affected by the bug, so it is a valid fingerprint of the split."""
    path = os.path.join(spec["outdir"], spec["orig_csv"])
    if not os.path.exists(path):
        print("    [WARN] original results CSV not found; cannot verify split.")
        return
    orig = pd.read_csv(path)
    orig = orig[orig["seed"].astype(str) != "AVG"]
    orig["seed"] = orig["seed"].astype(int)
    lookup = dict(zip(orig["seed"], orig["test_acc"]))

    print(f"\n  Split-reproduction check ({spec['pretty']}):")
    ok = True
    for r in rows:
        was = lookup.get(r["seed"])
        if was is None:
            continue
        match = abs(was - r["test_acc"]) < 1e-6
        ok &= match
        print(f"    seed {r['seed']:>4}: original acc={was:.4f}  recomputed acc={r['test_acc']:.4f}  "
              f"{'MATCH' if match else '*** MISMATCH ***'}")
    if ok:
        print("    -> Split reproduced exactly. Corrected metrics are trustworthy.")
    else:
        print("    -> WARNING: split did not reproduce (os.listdir order likely differs).")
        print("       The corrected metrics below may be evaluated on images the model")
        print("       saw during training. Re-run the original training script instead.")


def summarise(spec, rows):
    df = pd.DataFrame(rows)
    metric_cols = ["test_acc", "balanced_acc", "precision", "recall", "specificity",
                   "f1", "kappa", "auroc"]

    print(f"\n{'='*104}")
    print(f"  {spec['pretty']} — CORRECTED test metrics (predictions aligned with labels)")
    print(f"{'='*104}")
    hdr = (f"  {'Seed':<6}{'Acc':>9}{'BalAcc':>9}{'Prec':>9}{'Recall':>9}"
           f"{'Spec':>9}{'F1':>9}{'Kappa':>9}{'AUROC':>9}   {'TN/FP/FN/TP':>16}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"  {int(r['seed']):<6}{r['test_acc']:>9.4f}{r['balanced_acc']:>9.4f}"
              f"{r['precision']:>9.4f}{r['recall']:>9.4f}{r['specificity']:>9.4f}"
              f"{r['f1']:>9.4f}{r['kappa']:>9.4f}{r['auroc']:>9.4f}   "
              f"{int(r['tn'])}/{int(r['fp'])}/{int(r['fn'])}/{int(r['tp']):<8}")
    print("-" * len(hdr))
    means = df[metric_cols].mean()
    stds = df[metric_cols].std(ddof=0)
    print(f"  {'MEAN':<6}" + "".join(f"{means[c]:>9.4f}" for c in metric_cols))
    print(f"  {'SD':<6}" + "".join(f"{stds[c]:>9.4f}" for c in metric_cols))
    print("=" * len(hdr))

    out = df.copy()
    mean_row = {c: means[c] for c in metric_cols}
    mean_row.update({"seed": "MEAN", "n_test": df["n_test"].iloc[0],
                     "n_pos": df["n_pos"].iloc[0],
                     "tn": df["tn"].mean(), "fp": df["fp"].mean(),
                     "fn": df["fn"].mean(), "tp": df["tp"].mean()})
    sd_row = {c: stds[c] for c in metric_cols}
    sd_row.update({"seed": "SD", "n_test": "", "n_pos": "",
                   "tn": "", "fp": "", "fn": "", "tp": ""})
    out = pd.concat([out, pd.DataFrame([mean_row, sd_row])], ignore_index=True)

    dest = os.path.join(spec["outdir"], f"{spec['pretty'].lower()}_results_CORRECTED.csv")
    out.to_csv(dest, index=False)
    print(f"\n  Saved -> {dest}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS) + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    all_paths, all_labels = load_dataset(DATA_DIR, CLASSES)
    print("=" * 104)
    print("  CORRECTED RE-EVALUATION — fixing the shuffled-test-set metric bug")
    print("=" * 104)
    print(f"  Source pool: {len(all_paths)} images from {DATA_DIR}  "
          f"({int((all_labels == 0).sum())} normal / {int((all_labels == 1).sum())} tumour)")

    targets = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for name in targets:
        spec = MODEL_SPECS[name]
        print(f"\n\n### {spec['pretty']} ###")
        rows = []
        for seed in args.seeds:
            print(f"  evaluating seed {seed} ...")
            r = evaluate_seed(spec, seed, all_paths, all_labels)
            if r:
                rows.append(r)
        if not rows:
            print("  nothing evaluated.")
            continue
        check_against_original(spec, rows)
        summarise(spec, rows)


if __name__ == "__main__":
    main()
