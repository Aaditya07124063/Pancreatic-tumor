"""
reevaluate_pytorch.py
=====================
Independently recomputes the test metrics for the saved MobileViT and Swin
Transformer checkpoints, so that every number reported in the write-up comes
from a re-evaluation I ran myself rather than from the original training logs.

The PyTorch pipelines did not suffer the shuffled-test-set defect that affected
the two Keras pipelines (their test loader uses shuffle=False and they collect
predictions and labels in the same pass). This script therefore serves as an
independent confirmation rather than a correction: if the recomputed accuracy
matches the accuracy the original run logged, the original PyTorch numbers are
verified, and the additional metrics reported here (specificity, balanced
accuracy, AUROC) are new.

The model architectures are rebuilt from the default HuggingFace configs, which
correspond exactly to the checkpoints the training scripts fine-tuned:
    SwinConfig()      -> microsoft/swin-tiny-patch4-window7-224
    MobileViTConfig() -> apple/mobilevit-small
Both state dicts load with zero missing and zero unexpected keys, confirming the
architectures match.

Preprocessing replicates the training script's torchvision transform chain
exactly - Resize((224,224)) bilinear, ToTensor, Normalize(ImageNet stats) -
implemented directly on PIL so that torchvision is not required.

Usage:
    python reevaluate_pytorch.py
    python reevaluate_pytorch.py --model swin --seeds 42
"""

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

import torch
from transformers import (MobileViTConfig, MobileViTForImageClassification,
                          SwinConfig, SwinForImageClassification)

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DATA_DIR = "./train"
CLASSES = ["normal", "pancreatic_tumor"]
IMG_SIZE = 224
SEEDS = [42, 7, 21, 99, 123]
BATCH = 16

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPECS = {
    "swin": dict(pretty="Swin-Tiny", outdir="./swin_outputs", ckpt="swin_seed{seed}.pt",
                 orig_csv="swin_results.csv",
                 build=lambda: SwinForImageClassification(SwinConfig(num_labels=2))),
    "mobilevit": dict(pretty="MobileViT", outdir="./mobilevit_outputs", ckpt="mobilevit_seed{seed}.pt",
                      orig_csv="mobilevit_results.csv",
                      build=lambda: MobileViTForImageClassification(MobileViTConfig(num_labels=2))),
}


def load_dataset(data_dir, classes):
    """Unsorted os.listdir, matching the training scripts exactly. Sorting here
    would produce a different split and the checkpoints would then be scored on
    images they were trained on."""
    paths, labels = [], []
    for label, cls in enumerate(classes):
        d = os.path.join(data_dir, cls)
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(d, f))
                labels.append(label)
    return np.array(paths), np.array(labels)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_test_split(seed, paths, labels):
    set_seed(seed)
    _, tmp_p, _, tmp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=seed)
    _, te_p, _, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed)
    return te_p, te_l


def preprocess(path):
    """Replicates transforms.Compose([Resize((224,224)), ToTensor(), Normalize(...)])"""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # ToTensor scaling
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD               # Normalize
    return np.transpose(arr, (2, 0, 1))                      # HWC -> CHW


@torch.no_grad()
def predict(model, paths):
    model.eval()
    probs = []
    for i in range(0, len(paths), BATCH):
        batch = np.stack([preprocess(p) for p in paths[i:i + BATCH]])
        logits = model(pixel_values=torch.from_numpy(batch)).logits
        probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(probs)


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        test_acc=accuracy_score(y_true, y_pred),
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        f1=f1_score(y_true, y_pred, zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        auroc=roc_auc_score(y_true, y_prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPECS) + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    paths, labels = load_dataset(DATA_DIR, CLASSES)
    print("=" * 104)
    print("  INDEPENDENT RE-EVALUATION OF THE PYTORCH CHECKPOINTS")
    print("=" * 104)
    print(f"  Source pool: {len(paths)} images  "
          f"({int((labels == 0).sum())} normal / {int(labels.sum())} tumour)\n")

    for name in (list(SPECS) if args.model == "all" else [args.model]):
        spec = SPECS[name]
        print(f"\n### {spec['pretty']} ###")

        orig = pd.read_csv(os.path.join(spec["outdir"], spec["orig_csv"]))
        orig = orig[orig["seed"].astype(str) != "AVG"]
        orig["seed"] = orig["seed"].astype(int)
        logged = dict(zip(orig["seed"], orig["test_acc"]))

        rows = []
        for seed in args.seeds:
            ckpt = os.path.join(spec["outdir"], spec["ckpt"].format(seed=seed))
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt}")
                continue

            te_p, te_l = get_test_split(seed, paths, labels)
            model = spec["build"]()
            report = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            assert not report.missing_keys and not report.unexpected_keys, \
                f"architecture mismatch for {ckpt}"

            prob = predict(model, te_p)
            pred = (prob >= 0.5).astype(int)
            m = metrics(te_l.astype(int), pred, prob)
            m["seed"] = seed

            was = logged.get(seed)
            ok = was is not None and abs(was - m["test_acc"]) < 1e-6
            m["matches_original_log"] = bool(ok)
            rows.append(m)
            print(f"  seed {seed:>4} | acc={m['test_acc']:.4f} (logged {was:.4f} -> "
                  f"{'MATCH' if ok else '*** MISMATCH ***'}) | prec={m['precision']:.4f} "
                  f"rec={m['recall']:.4f} spec={m['specificity']:.4f} F1={m['f1']:.4f} "
                  f"kappa={m['kappa']:.4f} auroc={m['auroc']:.4f} | "
                  f"TN/FP/FN/TP={m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")
            del model

        if not rows:
            continue

        df = pd.DataFrame(rows)
        cols = ["test_acc", "balanced_acc", "precision", "recall",
                "specificity", "f1", "kappa", "auroc"]
        mean = {c: df[c].mean() for c in cols}
        sd = {c: df[c].std(ddof=0) for c in cols}

        print(f"\n  MEAN  " + "".join(f"{mean[c]*100:>9.2f}" for c in cols))
        print(f"  SD    " + "".join(f"{sd[c]*100:>9.2f}" for c in cols))
        verdict = ("all seeds reproduce the originally logged accuracy"
                   if df["matches_original_log"].all()
                   else "WARNING: some seeds do not match the original log")
        print(f"  ({verdict})")

        out = pd.concat([df,
                         pd.DataFrame([{**mean, "seed": "MEAN"}, {**sd, "seed": "SD"}])],
                        ignore_index=True)
        dest = os.path.join(spec["outdir"], f"{name}_results_RECOMPUTED.csv")
        out.to_csv(dest, index=False)
        print(f"  Saved -> {dest}")


if __name__ == "__main__":
    main()
