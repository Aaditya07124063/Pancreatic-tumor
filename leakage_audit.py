"""
Pre-training integrity audit for the pancreatic CT corpus.

Three checks, all cheap, run before any model training:
  1. Near-duplicate audit (perceptual dHash) -- exact MD5 dedup only removes
     byte-identical files; adjacent CT slices survive it and leak across splits.
  2. Residual split contamination -- how many test images have a near-duplicate
     sitting in the training partition, per seed.
  3. Shortcut baseline -- accuracy reachable from seven NON-diagnostic scalar
     statistics under the identical 5-seed stratified 80/10/10 protocol.
     Any deep model must be read against this floor, not against 50%.
"""
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

import data_utils as du


def dhash(path, size=8):
    """64-bit difference hash: robust to re-encoding/resizing, unlike MD5."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def shortcut_features(path):
    """Seven statistics that encode acquisition provenance, never pathology."""
    img = Image.open(path)
    w, h = img.size
    g = np.asarray(img.convert("L"), dtype=np.float32)
    import os
    return [g.mean(), g.std(), w, h, os.path.getsize(path),
            (g > 200).mean(), (g < 10).mean()]


def main():
    raw = du._collect_entries(du.DATA_DIRS, du.CLASSES)
    paths, labels = du.load_merged_dataset(dedupe=True, use_cache=False)
    print(f"Raw files: {len(raw)}  ->  after MD5 dedup: {len(paths)}\n")

    # ---- 1. near-duplicate grouping -------------------------------------
    print("Computing perceptual hashes...")
    H = np.stack([dhash(p) for p in paths])
    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    THRESH = 5  # Hamming distance over 64 bits
    for i in range(n):
        d = np.unpackbits(H[i] ^ H[i + 1:], axis=1).sum(axis=1)
        for off in np.nonzero(d <= THRESH)[0]:
            union(i, i + 1 + int(off))

    groups = np.array([find(i) for i in range(n)])
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(counts)
    in_multi = int((counts[inv] > 1).sum())
    print(f"Near-duplicate groups (dHash Hamming <= {THRESH}): {n_groups} "
          f"distinct scenes among {n} images")
    print(f"  images sharing a group with another image: {in_multi} "
          f"({100*in_multi/n:.1f}%)")
    print(f"  largest group: {counts.max()} images\n")

    # ---- 2. residual contamination under plain stratified split ---------
    print("Residual test-partition contamination (plain stratified split):")
    rates = []
    for seed in du.SEEDS:
        idx = np.arange(n)
        tr_i, va_i, te_i, _, _, _ = du.stratified_split(idx, labels, seed)
        tr_groups = set(groups[tr_i])
        contaminated = sum(1 for i in te_i if groups[i] in tr_groups)
        rate = 100 * contaminated / len(te_i)
        rates.append(rate)
        print(f"  seed {seed:>3}: {contaminated:>3}/{len(te_i)} test images "
              f"({rate:.1f}%) have a near-duplicate in train")
    print(f"  mean contamination: {np.mean(rates):.1f}%\n")

    # ---- 3. shortcut baseline -------------------------------------------
    print("Shortcut baseline (7 non-diagnostic statistics, same protocol):")
    X = np.array([shortcut_features(p) for p in paths], dtype=np.float64)
    accs, base = [], []
    for seed in du.SEEDS:
        tr_i, va_i, te_i, tr_l, va_l, te_l = du.stratified_split(
            np.arange(n), labels, seed)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr_i], tr_l)
        acc = clf.score(X[te_i], te_l)
        d = DummyClassifier(strategy="most_frequent").fit(X[tr_i], tr_l)
        accs.append(acc)
        base.append(d.score(X[te_i], te_l))
        print(f"  seed {seed:>3}: shortcut acc = {acc:.4f} | majority = {base[-1]:.4f}")
    print(f"\n  SHORTCUT FLOOR: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Majority baseline: {np.mean(base):.4f}")
    print("\n  A deep model scoring at or below the shortcut floor has "
          "demonstrated no pathology-specific learning.")


if __name__ == "__main__":
    main()
