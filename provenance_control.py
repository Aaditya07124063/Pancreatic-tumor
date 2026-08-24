"""
Provenance-matched control experiment.

Two defects make the headline accuracies uninterpretable:
  * the classes are perfectly separable by a brightness artefact -- the fraction
    of pixels brighter than 200 spans 0.00001-0.00613 for normal and
    0.01003-0.13960 for tumour, with no overlap;
  * 66% of test images share an identical 64-bit dHash with a training image.

This script neutralises each defect and measures what accuracy survives.

Interventions
  NORM   per-image rank equalisation. Every image is remapped to an exactly
         uniform intensity histogram, so mean, SD, and the saturated- and
         dark-pixel fractions become constant across the corpus and carry no
         class information by construction.
  GROUP  partition by exact-dHash cluster rather than by image, so near-identical
         slices cannot straddle the train/test boundary. 606 clusters, largest
         1.3% of the corpus, none mixing classes.

Arms (5 seeds each)
  A  raw  + random split   reproduces the reported ~100%
  B  norm + random split   isolates the brightness shortcut
  C  raw  + group  split   isolates near-duplicate leakage
  D  norm + group  split   both removed -- the only interpretable number
"""
import argparse
import collections
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

import data_utils as du
from cnn_scratch_train import (DEVICE, IMG_SIZE, BATCH_SIZE, LR,
                               PancreaticDataset, ScratchCNN, set_seed)
from leakage_audit import dhash, shortcut_features

FEATURE_NAMES = ["mean", "sd", "width", "height", "filesize", "frac>200", "frac<10"]


class ProvenanceMatch:
    """Remap an image to an exactly uniform intensity histogram."""

    def __call__(self, img):
        g = np.asarray(img.convert("L"))
        flat = g.ravel()
        order = flat.argsort(kind="stable")
        ranks = np.empty(flat.size, dtype=np.float64)
        ranks[order] = np.arange(flat.size)
        eq = (ranks * (255.0 / max(flat.size - 1, 1))).astype(np.uint8)
        return Image.fromarray(eq.reshape(g.shape)).convert("RGB")


def get_transforms(normalise, augment):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    steps = [ProvenanceMatch()] if normalise else []
    if augment:
        steps += [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(15)]
    else:
        steps += [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    return transforms.Compose(steps + [transforms.ToTensor(), norm])


def exact_dhash_groups(paths):
    H = np.stack([dhash(p) for p in paths])
    keys = [h.tobytes() for h in H]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([index[k] for k in keys])


def group_stratified_split(labels, groups, seed):
    """80/10/10 by dHash cluster, stratified on cluster label."""
    uniq = np.unique(groups)
    glab = np.array([labels[groups == g][0] for g in uniq])
    tr_g, tmp_g, tr_y, tmp_y = train_test_split(
        uniq, glab, test_size=0.2, stratify=glab, random_state=seed)
    va_g, te_g = train_test_split(
        tmp_g, test_size=0.5, stratify=tmp_y, random_state=seed)
    sel = lambda gs: np.nonzero(np.isin(groups, gs))[0]
    return sel(tr_g), sel(va_g), sel(te_g)


# ---------------------------------------------------------------- probe -----
def run_probe(paths, labels, groups):
    """Confirm the interventions actually remove what they target."""
    print("=" * 74)
    print("  PROBE 1 -- does rank equalisation destroy the brightness shortcut?")
    print("=" * 74)

    raw = np.array([shortcut_features(p) for p in paths])
    match = ProvenanceMatch()
    eq = []
    for p in paths:
        img = match(Image.open(p))
        g = np.asarray(img.convert("L"), dtype=np.float32)
        w, h = img.size
        eq.append([g.mean(), g.std(), w, h, 0.0, (g > 200).mean(), (g < 10).mean()])
    eq = np.array(eq)

    print(f"\n  {'feature':>10} {'RAW normal':>13} {'RAW tumour':>13} "
          f"{'EQ normal':>13} {'EQ tumour':>13}")
    for j, nm in enumerate(FEATURE_NAMES):
        print(f"  {nm:>10} {raw[labels==0,j].mean():>13.5f} {raw[labels==1,j].mean():>13.5f} "
              f"{eq[labels==0,j].mean():>13.5f} {eq[labels==1,j].mean():>13.5f}")

    for tag, X in [("RAW", raw), ("EQUALISED", eq)]:
        accs, base = [], []
        for seed in du.SEEDS:
            tr, va, te, trl, val, tel = du.stratified_split(
                np.arange(len(paths)), labels, seed)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=seed))
            clf.fit(X[tr], trl)
            accs.append(clf.score(X[te], tel))
            base.append(DummyClassifier(strategy="most_frequent")
                        .fit(X[tr], trl).score(X[te], tel))
        print(f"\n  {tag:>10} shortcut accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
              f"   (majority {np.mean(base):.4f})")

    print("\n" + "=" * 74)
    print("  PROBE 2 -- does group splitting remove near-duplicate leakage?")
    print("=" * 74)
    H = np.stack([dhash(p) for p in paths])
    for tag, splitter in [("random", "rand"), ("group", "grp")]:
        rates = []
        for seed in du.SEEDS:
            if splitter == "rand":
                tr, va, te, *_ = du.stratified_split(np.arange(len(paths)), labels, seed)
            else:
                tr, va, te = group_stratified_split(labels, groups, seed)
            d = np.unpackbits((H[te][:, None, :] ^ H[tr][None, :, :]).reshape(-1, 8),
                              axis=1).sum(1).reshape(len(te), len(tr))
            rates.append(100 * (d.min(1) == 0).mean())
        print(f"  {tag:>7} split: {np.mean(rates):>5.1f}% of test images have an "
              f"identical-dHash twin in train")


# ------------------------------------------------------------- training -----
def evaluate(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            yp.extend(out.argmax(1).cpu().numpy())
            yt.extend(y.numpy())
    yt, yp = np.array(yt), np.array(yp)
    return (yt == yp).mean(), f1_score(yt, yp, zero_division=0), cohen_kappa_score(yt, yp)


def train_arm(name, normalise, use_groups, paths, labels, groups, epochs):
    print(f"\n{'='*74}\n  ARM {name}  |  normalise={normalise}  group_split={use_groups}\n{'='*74}")
    rows = []
    for seed in du.SEEDS:
        set_seed(seed)
        if use_groups:
            tr, va, te = group_stratified_split(labels, groups, seed)
        else:
            tr, va, te, *_ = du.stratified_split(np.arange(len(paths)), labels, seed)

        mk = lambda idx, aug: DataLoader(
            PancreaticDataset(paths[idx], labels[idx], get_transforms(normalise, aug)),
            batch_size=BATCH_SIZE, shuffle=aug, num_workers=0)
        tr_ld, va_ld, te_ld = mk(tr, True), mk(va, False), mk(te, False)

        model = ScratchCNN(num_classes=len(du.CLASSES)).to(DEVICE)
        crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LR)

        best_va, best_state = -1.0, None
        for _ in range(epochs):
            model.train()
            for x, y in tr_ld:
                opt.zero_grad()
                loss = crit(model(x.to(DEVICE)), y.to(DEVICE))
                loss.backward()
                opt.step()
            va_acc, _, _ = evaluate(model, va_ld)
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        acc, f1, kappa = evaluate(model, te_ld)
        rows.append(dict(arm=name, seed=seed, n_train=len(tr), n_test=len(te),
                         val_acc=best_va, test_acc=acc, f1=f1, kappa=kappa))
        print(f"  seed {seed:>3}: n_train={len(tr):>4} n_test={len(te):>3} | "
              f"test_acc={acc:.4f} F1={f1:.4f} kappa={kappa:.4f}")

    df = pd.DataFrame(rows)
    print(f"  MEAN test_acc={df.test_acc.mean():.4f} +/- {df.test_acc.std():.4f} | "
          f"kappa={df.kappa.mean():.4f} +/- {df.kappa.std():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=du.EPOCHS)
    args = ap.parse_args()

    paths, labels = du.load_merged_dataset(dedupe=True)
    groups = exact_dhash_groups(paths)
    print(f"Corpus: {len(paths)} images | {len(np.unique(groups))} exact-dHash clusters\n")

    run_probe(paths, labels, groups)
    if args.probe_only:
        return

    arms = [("A raw+random", False, False), ("B norm+random", True, False),
            ("C raw+group", False, True), ("D norm+group", True, True)]
    out = pd.concat([train_arm(n, nz, g, paths, labels, groups, args.epochs)
                     for n, nz, g in arms], ignore_index=True)
    out.to_csv("provenance_control_results.csv", index=False)

    print(f"\n{'='*74}\n  SUMMARY ({args.epochs} epochs, {len(du.SEEDS)} seeds, ScratchCNN)\n{'='*74}")
    s = out.groupby("arm").agg(test_acc=("test_acc", "mean"), sd=("test_acc", "std"),
                               kappa=("kappa", "mean")).reset_index()
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  Arm D is the number that reflects pathology rather than provenance.")
    print("  Results written to provenance_control_results.csv")


if __name__ == "__main__":
    main()
